#include "kernel.cuh"
#include <crt\device_functions.h>


__device__ float fitness_function(float x[])
{
	float res = 0;
	float y1 = 1 + (x[0] - 1) / 4;
	float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;

	res += pow(sin(phi * y1), 2);

	for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++)
	{
		float y = 1 + (x[i] - 1) / 4;
		float yp = 1 + (x[i + 1] - 1) / 4;

		res += pow(y - 1, 2) * (1 + 10 * pow(sin(phi * yp), 2)) + pow(yn - 1, 2);

	}

	return res;
}



__global__ void kernelUpdateParticle(float *position, float *velocities, float *pBests, float *gBests, curandState* globalState1, curandState* globalState2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= NUM_OF_PARTICLES*NUM_OF_DIMENSIONS)
		return;

	curandState localState1 = globalState1[i];
	curandState localState2 = globalState2[i];
	//float rp = r1;
	//float rg = r2;
	float r1 = curand_uniform(&localState1);
	float r2 = curand_uniform(&localState1);
	globalState1[i] = localState1;
	globalState2[i] = localState2;
	//__syncthreads();
	velocities[i] = OMEGA*velocities[i] + c1*r1*(pBests[i] - position[i]) + c2*r2*(gBests[i] - position[i]);

	//Update position of particle
	position[i] += velocities[i];
}

__global__ void kernelUpdatePBest(float *positions, float *pBests, float *gBest)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= NUM_OF_PARTICLES*NUM_OF_DIMENSIONS || i%NUM_OF_DIMENSIONS != 0)
		return;

	for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
	{
		tempParticle1[i] = positions[i + j];
		tempParticle2[i] = pBests[i + j];
	}

	if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
	{
		for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
		{
			pBests[i + k] = positions[i + k];
		}
	}
}


__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}


void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest)
{
	int size = NUM_OF_PARTICLES*NUM_OF_DIMENSIONS;
	//int size_random = NUM_OF_PARTICLES*NUM_OF_DIMENSIONS * 2;
	curandState* devStates1;
	curandState* devStates2;

	float *devPos;
	float *devVel;
	float *devPBest;
	float *devGBest;

	float temp[NUM_OF_DIMENSIONS];

	//Memory allocation
	cudaMalloc((void**)&devPos, sizeof(float)*size);
	cudaMalloc((void**)&devVel, sizeof(float)*size);
	cudaMalloc((void**)&devPBest, sizeof(float)*size);
	cudaMalloc((void**)&devGBest, sizeof(float)*size);

	cudaMalloc(&devStates1, size * sizeof(curandState));
	cudaMalloc(&devStates2, size * sizeof(curandState));


	//Thread & Block number
	int threadsNum = 256;
	int blocksNum = NUM_OF_PARTICLES / threadsNum;

	//copy particle data from host to device
	cudaMemcpy(devPos, positions, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVel, positions, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(devPBest, positions, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(devGBest, positions, sizeof(float)*size, cudaMemcpyHostToDevice);



	//PSO main function
	for (int iter = 0; iter < MAX_ITER; iter++)
	{

		//initialize the random num
		setup_kernel << < blocksNum, threadsNum >> > (devStates1, time(NULL));
		setup_kernel << < blocksNum, threadsNum >> > (devStates2, time(NULL));

		//clock_t countBegin = clock();
		//Update position and velocity

		kernelUpdateParticle << <blocksNum, threadsNum >> > (devPos, devVel, devPBest, devGBest, devStates1, devStates2);

		//Update pBest
		kernelUpdatePBest << <blocksNum, threadsNum >> > (devPos, devPBest, devGBest);

		//Update gBest
		cudaMemcpy(pBests, devPBest, sizeof(float)*NUM_OF_PARTICLES*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);

		for (int i = 0; i < size; i += NUM_OF_DIMENSIONS)
		{
			for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
			{
				temp[k] = pBests[i + k];
			}

			if (host_fitness_function(temp) < host_fitness_function(gBest))
			{
				for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
				{
					gBest[k] = temp[k];
				}
			}
		}

		cudaMemcpy(devGBest, gBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);
		//clock_t countEnd = clock();
		//printf("The iter time consumption : %10.3lf ms\n", (double)(countEnd - countBegin) / CLOCKS_PER_SEC);
		//printf("The iter number is: %d\n", iter);
	}

	//Retrieve particle data from device to host
	cudaMemcpy(positions, devPos, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities, devVel, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pBests, devPBest, sizeof(float)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(gBest, devGBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);

	//clean up
	cudaFree(devPos);
	cudaFree(devVel);
	cudaFree(devPBest);
	cudaFree(devGBest);
	cudaFree(devStates1);
	cudaFree(devStates2);

}

