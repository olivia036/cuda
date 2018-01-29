#include <iostream>
#include <time.h>
#include <math.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <helper_cuda.h>
//#include <helper_timer.h>

//#include <math_functions.h>
using namespace std;

const int NUM_OF_PARTICLES = 80;
const int NUM_OF_DIMENSIONS = 26;
const int MAX_ITER = 300;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
__device__ float OMEGA = 0.5f;
__device__ float c1 = 1.5f;
__device__ float c2 = 1.5f;
__device__ float phi = 3.1415f;


void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest);
float host_fitness_function(float x[]);
//float getRandom(float low, float high);
//float getRandomClamped();

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__device__ float fitness_function(float x[]);

__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];
__device__ float fitness_function(float x[]);
//__device__ float generate(curandState* globalState, int ind);


__global__ void addKernel(int *c, const int *a, const int *b);
__global__ void kernelUpdateParticle(float *position, float *velocities, float *pBests, float *gBests, curandState* globalState1, curandState* globalState2);
__global__ void kernelUpdatePBest(float *positions, float *pBests, float *gBest);


