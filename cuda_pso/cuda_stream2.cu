#include "kernel.cuh"

//fitness_function define
__device__ float fitness_function(float x[]);

//setup random number
__global__ void setup_rand(curandState * state, unsigned long seed)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
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
	float r2 = curand_uniform(&localState2);
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



//cuda_pso define
void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest)
{
	int size = NUM_OF_PARTICLES*NUM_OF_DIMENSIONS;
	int memcpyCount = 1040;
	
	//setup a cuda stream  
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);

	//setup random number seed
	curandState* devStates1;
	curandState* devStates2;
	
	cudaMalloc(&devStates1, size * sizeof(curandState));
	cudaMalloc(&devStates2, size * sizeof(curandState));

	//device memory allocation
	float *devPos1;
	float *devVel1;
	float *devPBest1;
	float *devGBest1;

	float *devPos2;
	float *devVel2;
	float *devPBest2;
	float *devGBest2;
	
	float temp[NUM_OF_DIMENSIONS];
	//stream1
	cudaMalloc((void**)&devPos1, sizeof(float)*size/2);//GPU memory allocation
	cudaMalloc((void**)&devVel1, sizeof(float)*size/2);
	cudaMalloc((void**)&devPBest1, sizeof(float)*size/2);
	cudaMalloc((void**)&devGBest1, sizeof(float)*size/2);
	//stream2
	cudaMalloc((void**)&devPos2, sizeof(float)*size / 2);//GPU memory allocation
	cudaMalloc((void**)&devVel2, sizeof(float)*size / 2);
	cudaMalloc((void**)&devPBest2, sizeof(float)*size / 2);
	cudaMalloc((void**)&devGBest2, sizeof(float)*size / 2);
	//CPU memory allocation
	cudaHostAlloc((void**)&positions, sizeof(float)*size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&velocities, sizeof(float)*size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&pBests, sizeof(float)*size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&gBest, sizeof(float)*size, cudaHostAllocDefault);

	//set thread & block number
	int threadsNum = 256;
	int blocksNum = NUM_OF_PARTICLES / threadsNum;

	for (int i=0;i<size/2;i+=memcpyCount/2)
	{
		cudaMemcpyAsync(devPos1, positions + i, sizeof(float)*size/2, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(devVel1, positions + i, sizeof(float)*size/2, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(devPBest1, positions + i, sizeof(float)*size/2, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(devGBest1, positions + i, sizeof(float)*size/2, cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(devPos2, positions + i + memcpyCount, sizeof(float)*size / 2, cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(devVel2, positions + i + memcpyCount, sizeof(float)*size / 2, cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(devPBest2, positions + i + memcpyCount, sizeof(float)*size / 2, cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(devGBest2, positions + i + memcpyCount, sizeof(float)*size / 2, cudaMemcpyHostToDevice, stream2);
	}
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	//PSO main function
	for (int iter = 0; iter < MAX_ITER; iter++)
	{

		//initialize the random num
		setup_rand << < blocksNum, threadsNum >> > (devStates1, time(NULL));
		setup_rand << < blocksNum, threadsNum >> > (devStates2, time(NULL));

		//clock_t countBegin = clock();
		//Update position and velocity

		kernelUpdateParticle << <blocksNum, threadsNum >> > (devPos1, devVel1, devPBest1, devGBest1, devStates1, devStates2);
		kernelUpdateParticle << <blocksNum, threadsNum >> > (devPos2, devVel2, devPBest2, devGBest2, devStates2, devStates2);

		//Update pBest
		kernelUpdatePBest << <blocksNum, threadsNum >> > (devPos1, devPBest1, devGBest1);
		kernelUpdatePBest << <blocksNum, threadsNum >> > (devPos2, devPBest2, devGBest2);

		//Update gBest
		cudaMemcpyAsync(pBests, devPBest1, sizeof(float)*NUM_OF_PARTICLES*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost,stream1);
		cudaMemcpyAsync(pBests+memcpyCount, devPBest2, sizeof(float)*NUM_OF_PARTICLES*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost, stream1);

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

		cudaMemcpyAsync(devGBest1, gBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost,stream2);
		cudaMemcpyAsync(devGBest2, gBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost,stream2);

		//clock_t countEnd = clock();
		//printf("The iter time consumption : %10.3lf ms\n", (double)(countEnd - countBegin) / CLOCKS_PER_SEC);
		//printf("The iter number is: %d\n", iter);
	}

	for (int i = 0; i < size/2; i += memcpyCount/2)
	{
		cudaMemcpyAsync(positions + i, devPos1 + i, sizeof(float)*size, cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync(positions + i, devVel1 + i, sizeof(float)*size, cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync(positions + i, devPBest1 + i, sizeof(float)*size, cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync(positions + i, devGBest1 + i, sizeof(float)*size, cudaMemcpyDeviceToHost, stream1);

		cudaMemcpyAsync(positions + i + memcpyCount, devPos1 + i + memcpyCount, sizeof(float)*size, cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync(positions + i + memcpyCount, devVel1 + i + memcpyCount, sizeof(float)*size, cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync(positions + i + memcpyCount, devPBest1 + i + memcpyCount, sizeof(float)*size, cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync(positions + i + memcpyCount, devGBest1 + i + memcpyCount, sizeof(float)*size, cudaMemcpyDeviceToHost, stream2);

	}
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	//clean up
	cudaFree(devPos1);
	cudaFree(devVel1);
	cudaFree(devPBest1);
	cudaFree(devGBest1);

	cudaFree(devPos2);
	cudaFree(devVel2);
	cudaFree(devPBest2);
	cudaFree(devGBest2);

	cudaFree(devStates1);
	cudaFree(devStates2);
}



//fitness_function define
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

