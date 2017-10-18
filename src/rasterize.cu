/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef glm::vec3 FragmentAttributeColor;
	typedef float     FragmentAttributeIntensity;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own
		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, 
		  		//cuz normal will go wrong after perspective transformation
		// glm::vec3 col;

		 VertexAttributeTexcoord texcoord0;
		 // don't free this resource  because it is a copy of the 
		 // resource
		 TextureData* dev_diffuseTex = NULL;
		 int diffuseTexWidth, diffuseTexHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;
		// VertexAttributeTexcoord texcoord0;
		// TextureData* dev_diffuseTex;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static dim3 numThreadsPerBlock(128);
static int * dev_depth = NULL;	// you might need this buffer when doing depth test
// A point light source for global illumination
// numLights number of light source
static constexpr int numLights {2};
// these two represent the direction of the light source in world space
static constexpr FragmentAttributeIntensity sourceIntensity[] {2.0f, 2.0f};
// light is pointing down and away from the camera.
static VertexAttributePosition lights[] { glm::vec3(0.0f, -3.0f, 10.0f),
	                                                   glm::vec3(-1.0f,  - 1.0f, 0.0f)};
static FragmentAttributeColor    lightColor[]  { glm::vec3(1.0f, 1.0f, 1.0f),
	                                                       glm::vec3(0.5, 0.5, 0.0f)};
// light source in world coordinates
static   VertexAttributePosition* dev_lightDirection;
static   FragmentAttributeColor*  dev_lightColor;
// holds the current light direction after transformation
static   VertexAttributePosition* dev_currentLightDirection;
static constexpr int BytesPerPixel {3};


/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, 
		 int numLights, const VertexAttributePosition* lightSource, 
		 const FragmentAttributeColor*  lightColor) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);
    
    if (x < w && y < h) {
	 glm::vec3 color(0.f);
	    // negative here because the light contributes when negative
	    // of the light direction is aligned with the normal
	 for (int i{0}; i < 1; ++i) {
		float dot { -glm::dot(fragmentBuffer[index].eyeNor, lightSource[i])};
		dot = clamp(dot);
		color += fragmentBuffer[index].color *dot * lightColor[i];
	 }
	 framebuffer[index] = color;
   }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));
// 4 Malloc for the light direction
	{
		cudaMalloc(&dev_lightDirection, numLights * sizeof(VertexAttributePosition));
		cudaMalloc(&dev_currentLightDirection, numLights * sizeof(VertexAttributePosition));
		cudaMalloc(&dev_lightColor, numLights * sizeof(FragmentAttributeColor));
		// normalize direction and update color with intensity
		for (int i {0}; i < numLights; ++i) {
			lights[i] = glm::normalize(lights[i]);
			lightColor[i] *= sourceIntensity[i];
		}
		cudaMemcpy(dev_lightDirection, (void *) lights,
			numLights * sizeof(VertexAttributePosition), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_lightColor, (void *) lightColor, 
			numLights *  sizeof(FragmentAttributeColor), cudaMemcpyHostToDevice);
		checkCUDAError("Add Light Direction");
	}

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, 
						std::vector<PrimitiveDevBufPointers>>(mesh.name,
							std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) 
{

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) 
	{
		glm::vec4 pos = MV * glm::vec4(primitive.dev_position[vid], 1.0f);
		primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
		pos /= pos.w;
		// eyePos is the vertex position in the camera view but not in
		// clip coordinates
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(pos);
		pos = MVP * pos;
		// turn the viewpos into NDC
		pos /= pos.w;
		// now scale into pixel space
		//pos = glm::vec4(primitive.dev_position[vid], 1.0f);
		// OpenGL has pixel 0, 0 is in the upper right as in 
		// the pathtracer
		pos.x = 0.5 * width *  (-pos.x + 1);
		pos.y = 0.5 * height * (-pos.y + 1);
		primitive.dev_verticesOut[vid].pos = pos;
		primitive.dev_verticesOut[vid].texcoord0        = primitive.dev_texcoord0[vid];
		primitive.dev_verticesOut[vid].dev_diffuseTex   = primitive.dev_diffuseTex;
		primitive.dev_verticesOut[vid].diffuseTexWidth  = primitive.diffuseTexWidth;
		primitive.dev_verticesOut[vid].diffuseTexHeight = primitive.diffuseTexHeight;
		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		
	}
}

__global__ void transformLights(int numLights, const glm::mat3 MV, 
		VertexAttributePosition* dev_currentLightDirection,
		const VertexAttributePosition* dev_lightDirection)
{
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (iid < numLights) 
	{
		dev_currentLightDirection[iid] = MV * dev_lightDirection[iid];
	}
}


static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}
	
}
// for now assume tex is 4 bytes in the order rgba-- a not yet used
__host__ __device__ FragmentAttributeColor textureToFragmentColor(TextureData* tex) 
{
	FragmentAttributeColor allcolors;
	// convert blue color
	float maxf { 1/(float) 0xFF};
	allcolors.b = (float) tex[2]* maxf;
	// convert the green color
	allcolors.g = (float) tex[1] * maxf;
	// convert the red color
	allcolors.r = (float) tex[0] * maxf;
	return allcolors;
}
// scale the float of an image from 0 to 1 to 0 in pixels to width)
// will not overflow beyond 0 to width -1
__host__ __device__ int ScaleImageFloat(float val, int width)
{
	const float delta{ 0.0001f };
	const float delta2 { 0.0002f};
	val = delta + ((float)width - delta2) * val;
	return int(val);
}
// based on uv coordinate it looks up the color in the uv Texture map and returns it as a FragmentAttributeColor
__host__ __device__ FragmentAttributeColor uvColor(int texWidth, int texHeight, VertexAttributeTexcoord * uv,
		     TextureData* dev_diffuseTex)
{
	
	// assume the texture field is BytesPerPixel wide:
	//assume that uv (0f, 0f) is lower left corner.
	// assume that in the texturepixel (0,0) is upper left
	int pixelWidth = ScaleImageFloat(uv->x, texWidth);
	int pixelHeight = ScaleImageFloat(uv->y, texHeight);
	int texturepixel = BytesPerPixel* (pixelWidth + texWidth  * pixelHeight);
	return textureToFragmentColor(dev_diffuseTex + texturepixel);
	//float scale = 1.f / 255.f;
	//return scale * glm::vec3(dev_diffuseTex[texturepixel],
//		dev_diffuseTex[texturepixel + 1],
//		dev_diffuseTex[texturepixel + 2]);
	//return textureToFragmentColor(dev_diffuseTex + texturepixel);
	//int x = 0.5f + (texWidth - 1.f) * uv->x;
	//int y = 0.5f + (texHeight - 1.f) * uv->y;
	//int x = texWidth * uv->x;
	//int y = texHeight * uv->y;
	//float scale = 1.f / 255.f;
	//int index = x + y * texWidth;
	//return scale * glm::vec3(dev_diffuseTex[index * 3],
	//	dev_diffuseTex[index * 3 + 1],
	//	dev_diffuseTex[index * 3 + 2]);
}
// update fragmentBuffer only if the current depth is less than the depth already stored
// this is an atomic operation.
__device__ float updateFragmentClosestDepth(Fragment* fragmentBuffer, const Fragment * currentFragment, 
		                            int * dev_depth, float newDepth)
{
    const int * newVal = (int *) &newDepth;
	int old = *dev_depth, assumed;
	// old always has the current *dev_depth. if atomicCas 1
	// fails old is *dev_depth.  if the first succeeeds and 
	// the second one fails (another
	// thread wrote to the buffer) then old is *dev_depth. If they
	// both succeed then *dev_depth will not change in the second call.
	do {
	      float * current { (float*) &old};
	      assumed = old;
	      // fragment fails the depth test
	      if ( *current < newDepth) {
		      return *current;
	      }
	      // this failure means another thread wrote to the
	      // buffer
	      old = atomicCAS(dev_depth, assumed, *newVal);
	      // No one else wrote to the depth and this is the
	      // closest fragment
	      if ( old == assumed) {
		      // update the fragmentBuffer because no other thread wrote 
			  // to the depth buffer
		      *fragmentBuffer= *currentFragment;
		      // check if no one else wrote to depth buffer
	      }
	}while (old != assumed);
	return newDepth;
}

// returns the world space barycentric coordinates as a vec3 given the pixelSpace barycentric coordinates
// , the triangle in pixel space and the zpixel of the desired pixel.
// One way to see this is that 1/zPixel is proportional to z world and pixB[0] is proportional to 
// xpix, ypix which are proportional to xreal/zworld.  In the end each coefficient is proportional to
// xworld, yworld as it would be in barycentric coordinates.
__host__ __device__ glm::vec3  worldSpaceBarycentricCoordinates(const glm::vec3 pixB, 
		const glm::vec3 tri[3], float zPixel)
{
              
	      if ( zPixel > -.00001 && zPixel < .00001) {
		      zPixel = -.01;
	      }
	      // zbuffer is negative because there is a sign reversal in getZAtCoordinate
	      // That reversal is there so that the smallest depth is the closest. Here though
	      // we want the real zvalue because those are compared with tri[1].z
	      float invZ = - 1/zPixel;
	      return glm::vec3(  invZ * pixB[0] * tri[0].z, invZ * pixB[1] * tri[1].z, 
			         invZ * pixB[2] * tri[2].z);

}
// baryCentric Avg with vec3 
__host__ __device__ glm::vec3 baryCentricAvg(const glm::vec3 bC, const glm::vec3 v0, const glm::vec3 v1,
		                             const glm::vec3 v2)
{
	return v0 * bC.x + v1 * bC.y + v2 * bC.z;
}
// baryCentric Avg with vec2 
__host__ __device__ glm::vec2 baryCentricAvg(const glm::vec3 bC, const glm::vec2 v0, const glm::vec2 v1,
		                             const glm::vec2 v2)
{
	return v0 * bC.x + v1 * bC.y + v2 * bC.z;
}
// perform rasterization on the triangles
__global__  void rasterizeTriangles (int numTriangles, Fragment* fragmentBuffer, Primitive * dev_primitives,  
		int * dev_depth, 
		int width, int height)
{

	// index id
	int indx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( indx < numTriangles) 
	{
		Primitive& prim { dev_primitives[indx]};
		const glm::vec3 tri[3] { glm::vec3(prim.v[0].pos), 
			               glm::vec3(prim.v[1].pos),
			               glm::vec3(prim.v[2].pos)};
		AABB box = getAABBForTriangle(tri);
		for (int i {0}; i < width; ++i)
		{
		   if (i < box.min.x || i > box.max.x) 
		   {
		          continue;
		   }
		   for (int j{ 0 }; j < height; ++j)
		   {
			   if (j < box.min.y || j > box.max.y) {
				   continue;
			   }
			   glm::vec2 point(i, j);
			   // calculate the barycentric coordinates in pixel space.
			   // these have no dependence on the zpixel space as they should
			   glm::vec3 bC = calculateBarycentricCoordinate(tri, point);
			   // the zdepth in pixel space
			   if (isBarycentricCoordInBounds(bC)) {
				   int pix = i + width * j;
				   // Barycentric UV coordinate
				   // no texture
				   // printf("%d %d %d\n", prim.v[0].diffuseTexWidth,
				   //	   prim.v[0].dev_diffuseTex);
				   if(prim.v[0].diffuseTexWidth == 0) {
					   // default color is red
					   fragmentBuffer[pix].color = glm::vec3(0.8f, 0.0f, 0.0f);
				   }
				   else {
					float fragmentdepth =  getZAtCoordinate(bC, tri);
					// convert to Barycentric coordinates in world Space
					bC = worldSpaceBarycentricCoordinates(bC, tri, fragmentdepth);
					VertexAttributeTexcoord uvPoint {baryCentricAvg(bC, 
							prim.v[0].texcoord0, prim.v[1].texcoord0, 
							prim.v[2].texcoord0)};
					Fragment fragbuffer;
					fragbuffer.color = uvColor(prim.v[0].diffuseTexWidth,
						   prim.v[0].diffuseTexHeight,
						   &uvPoint, prim.v[0].dev_diffuseTex);
					fragbuffer.eyePos = baryCentricAvg(bC, prim.v[0].eyePos, prim.v[1].eyePos,
							      prim.v[2].eyePos);
					fragbuffer.eyeNor = baryCentricAvg(bC, prim.v[0].eyeNor, prim.v[1].eyeNor,
							                   prim.v[2].eyeNor);
					// this will update the depth only when this pixel wins the depth buffer test
					// in case two threads decide to write to the same fragment buffer at the same time
					// only one will update the fragment.
					fragmentdepth = updateFragmentClosestDepth(fragmentBuffer + pix, &fragbuffer,
						dev_depth + pix, fragmentdepth);
				   }
			   }
		   }
			 
		}

	}
}
dim3 gridSize(int num_paths) 
{
	return (num_paths + numThreadsPerBlock.x - 1) /
		numThreadsPerBlock.x;
}


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >
					   (p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");
				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
        {
		// transform the lights to Eye Space so that the normal is correct
		dim3 lightgrid { gridSize(numLights)};
		transformLights<<<lightgrid, numThreadsPerBlock>>>(numLights,
			      glm::mat3(), dev_currentLightDirection,
				dev_lightDirection);
		checkCUDAError("Transform Lights to Eye Space");
	}
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	// TODO: rasterize
	{
		curPrimitiveBeginId = 0;

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForTriangles{ gridSize(p->numPrimitives) };
				rasterizeTriangles << <numBlocksForTriangles, numThreadsPerBlock >> > (
					p->numPrimitives, dev_fragmentBuffer, dev_primitives + 
					   curPrimitiveBeginId, dev_depth, width, height);
				checkCUDAError("Rasterizer");
				curPrimitiveBeginId += p->numPrimitives;
			}
		}
		checkCUDAError("Rasterizer Final");
	}

	

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer,
			numLights, dev_currentLightDirection, dev_lightColor);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

	cudaFree(dev_framebuffer);
	dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;
	cudaFree(dev_lightColor);
	dev_lightColor = NULL;
	cudaFree(dev_lightDirection);
	dev_lightDirection = NULL;
	cudaFree(dev_currentLightDirection);
	dev_currentLightDirection = NULL;
	checkCUDAError("rasterize Free");
}
