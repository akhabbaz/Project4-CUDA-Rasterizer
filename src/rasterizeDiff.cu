diff --git a/src/rasterize.cu b/src/rasterize.cu
index 72dd60f..14b4521 100644
--- a/src/rasterize.cu
+++ b/src/rasterize.cu
@@ -114,7 +114,7 @@ static Primitive *dev_primitives = NULL;
 static Fragment *dev_fragmentBuffer = NULL;
 static glm::vec3 *dev_framebuffer = NULL;
 static dim3 numThreadsPerBlock(128);
-static int * dev_depth = NULL;	// you might need this buffer when doing depth test
+static float * dev_depth = NULL;	// you might need this buffer when doing depth test
 // A point light source for global illumination
 // numLights number of light source
 static constexpr int numLights {2};
@@ -173,7 +173,7 @@ void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer,
 	 for (int i{0}; i < 1; ++i) {
 		float dot { -glm::dot(fragmentBuffer[index].eyeNor, lightSource[i])};
 		dot = clamp(dot);
-		color += fragmentBuffer[index].color *dot * lightColor[i];
+		color += fragmentBuffer[index].color; // *dot * lightColor[i];
 	 }
 	 framebuffer[index] = color;
    }
@@ -193,7 +193,7 @@ void rasterizeInit(int w, int h) {
     cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
     
 	cudaFree(dev_depth);
-	cudaMalloc(&dev_depth, width * height * sizeof(int));
+	cudaMalloc(&dev_depth, width * height * sizeof(float));
 // 4 Malloc for the light direction
 	{
 		cudaMalloc(&dev_lightDirection, numLights * sizeof(VertexAttributePosition));
@@ -215,7 +215,7 @@ void rasterizeInit(int w, int h) {
 }
 
 __global__
-void initDepth(int w, int h, int * depth)
+void initDepth(int w, int h, float * depth)
 {
 	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
 	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
@@ -223,7 +223,9 @@ void initDepth(int w, int h, int * depth)
 	if (x < w && y < h)
 	{
 		int index = x + (y * w);
-		depth[index] = INT_MAX;
+		float max = 100000.f;
+		int * maxptr = (int *)(&max);
+		depth[index] = max;
 	}
 }
 
@@ -688,7 +690,7 @@ void _vertexTransformAndAssembly(
 		// eyePos is the vertex position in the camera view but not in
 		// clip coordinates
 		primitive.dev_verticesOut[vid].eyePos = glm::vec3(pos);
-		pos = MVP * pos;
+		pos = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
 		// turn the viewpos into NDC
 		pos /= pos.w;
 		// now scale into pixel space
@@ -802,36 +804,61 @@ __host__ __device__ FragmentAttributeColor uvColor(int texWidth, int texHeight,
 // update fragmentBuffer only if the current depth is less than the depth already stored
 // this is an atomic operation.
 __device__ float updateFragmentClosestDepth(Fragment* fragmentBuffer, const Fragment * currentFragment, 
-		                            int * dev_depth, float newDepth)
+		                            float * addr, float value)
 {
-    const int * newVal = (int *) &newDepth;
-	int old = *dev_depth, assumed;
-	// old always has the current *dev_depth. if atomicCas 1
-	// fails old is *dev_depth.  if the first succeeeds and 
-	// the second one fails (another
-	// thread wrote to the buffer) then old is *dev_depth. If they
-	// both succeed then *dev_depth will not change in the second call.
-	do {
-	      float * current { (float*) &old};
-	      assumed = old;
-	      // fragment fails the depth test
-	      if ( *current < newDepth) {
-		      return *current;
-	      }
-	      // this failure means another thread wrote to the
-	      // buffer
-	      old = atomicCAS(dev_depth, assumed, *newVal);
-	      // No one else wrote to the depth and this is the
-	      // closest fragment
-	      if ( old == assumed) {
-		      // update the fragmentBuffer because no other thread wrote 
-			  // to the depth buffer
-		      *fragmentBuffer= *currentFragment;
-		      // check if no one else wrote to depth buffer
-	      }
-	}while (old != assumed);
-	return newDepth;
+   
+        float old = *addr, assumed;
+
+        if(old <= value) return old;
+
+        do
+
+        {
+
+                assumed = old;
+
+                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
+		if ( old == assumed) {
+			*fragmentBuffer = *currentFragment;
+		}
+
+        }while(old!=assumed);
+
+        return old; 
 }
+// update fragmentBuffer only if the current depth is less than the depth already stored
+// this is an atomic operation.
+//__device__ float updateFragmentClosestDepth(Fragment* fragmentBuffer, const Fragment * currentFragment, 
+//		                            int * dev_depth, float newDepth)
+//{
+//    const int * newVal = (int *) &newDepth;
+//	int old = *dev_depth, assumed;
+//	// old always has the current *dev_depth. if atomicCas 1
+//	// fails old is *dev_depth.  if the first succeeeds and 
+//	// the second one fails (another
+//	// thread wrote to the buffer) then old is *dev_depth. If they
+//	// both succeed then *dev_depth will not change in the second call.
+//	do {
+//	      float * current { (float*) &old};
+//	      assumed = old;
+//	      // fragment fails the depth test
+//	      if ( *current > newDepth) {
+//		      return *current;
+//	      }
+//	      // this failure means another thread wrote to the
+//	      // buffer
+//	      old = atomicCAS(dev_depth, assumed, *newVal);
+//	      // No one else wrote to the depth and this is the
+//	      // closest fragment
+//	      if ( old == assumed) {
+//		      // update the fragmentBuffer because no other thread wrote 
+//			  // to the depth buffer
+//		      *fragmentBuffer= *currentFragment;
+//		      // check if no one else wrote to depth buffer
+//	      }
+//	}while (old != assumed);
+//	return newDepth;
+//}
 
 // returns the world space barycentric coordinates as a vec3 given the pixelSpace barycentric coordinates
 // , the triangle in pixel space and the zpixel of the desired pixel.
@@ -867,7 +894,7 @@ __host__ __device__ glm::vec2 baryCentricAvg(const glm::vec3 bC, const glm::vec2
 }
 // perform rasterization on the triangles
 __global__  void rasterizeTriangles (int numTriangles, Fragment* fragmentBuffer, Primitive * dev_primitives,  
-		int * dev_depth, 
+		float * dev_depth, 
 		int width, int height)
 {
 
@@ -926,6 +953,7 @@ __global__  void rasterizeTriangles (int numTriangles, Fragment* fragmentBuffer,
 					// only one will update the fragment.
 					fragmentdepth = updateFragmentClosestDepth(fragmentBuffer + pix, &fragbuffer,
 						dev_depth + pix, fragmentdepth);
+					fragmentBuffer[pix] = fragbuffer;
 				   }
 			   }
 		   }
@@ -968,41 +996,19 @@ void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const g
 				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
 
 				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >
-					   (p->numVertices, *p, MVP, MV, MV_normal, width, height);
+					(p->numVertices, *p, MVP, MV, MV_normal, width, height);
 				checkCUDAError("Vertex Processing");
 				cudaDeviceSynchronize();
 				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
-					(p->numIndices, 
-					curPrimitiveBeginId, 
-					dev_primitives, 
-					*p);
+					(p->numIndices,
+						curPrimitiveBeginId,
+						dev_primitives,
+						*p);
 				checkCUDAError("Primitive Assembly");
-				curPrimitiveBeginId += p->numPrimitives;
 			}
-		}
-
-		checkCUDAError("Vertex Processing and Primitive Assembly");
-	}
-        {
-		// transform the lights to Eye Space so that the normal is correct
-		dim3 lightgrid { gridSize(numLights)};
-		transformLights<<<lightgrid, numThreadsPerBlock>>>(numLights,
-			      glm::mat3(), dev_currentLightDirection,
-				dev_lightDirection);
-		checkCUDAError("Transform Lights to Eye Space");
-	}
-	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
-	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
-	// TODO: rasterize
-	{
-		curPrimitiveBeginId = 0;
-
-		auto it = mesh2PrimitivesMap.begin();
-		auto itEnd = mesh2PrimitivesMap.end();
-
-		for (; it != itEnd; ++it) {
-			auto p = (it->second).begin();	// each primitive
-			auto pEnd = (it->second).end();
+			curPrimitiveBeginId = 0;
+			p = (it->second).begin();
+			pEnd = (it->second).end();
 			for (; p != pEnd; ++p) {
 				dim3 numBlocksForTriangles{ gridSize(p->numPrimitives) };
 				rasterizeTriangles << <numBlocksForTriangles, numThreadsPerBlock >> > (
