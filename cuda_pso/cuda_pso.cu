#include "kernel.cuh"

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


__global__ void kernelUpdateParticle(float *position, float *velocities, float *pBests, float *gBests, float r1, float r2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= NUM_OF_PARTICLES*NUM_OF_DIMENSIONS)
		return;

	//float rp = getRandomClamped();
	//float rg = getRandomClamped();

	float rp = r1;
	float rg = r2;

	velocities[i] = OMEGA*velocities[i] + c1*rp*(pBests[i] - position[i]) + c2*rg*(gBests[i] - position[i]);

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
//
//return random num between low and high
float getRandom(float low, float high)
{
	return low + float(((high - low) + 1)*rand() / (RAND_MAX + 1.0));
}

//return random num between 0.0f and 1.0f
float getRandomClamped()
{
	return (float)rand() / (float)RAND_MAX;
}

void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest)
{
	int size = NUM_OF_PARTICLES*NUM_OF_DIMENSIONS;

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
		//clock_t countBegin = clock();
		//Update position and velocity
		kernelUpdateParticle << <blocksNum, threadsNum >> > (devPos, devVel, devPBest, devGBest, getRandomClamped(), getRandomClamped());

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

}

