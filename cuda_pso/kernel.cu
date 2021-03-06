#include "kernel.cuh"

int main(int argc, char** argv)
{

	const int arraySize = 1;
	const int a[arraySize] = { 1 };
	const int b[arraySize] = { 1 };
	int c[arraySize] = { 0 };
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//particles
	float position[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
	float velocities[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
	float pBests[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];

	//gBest
	float gBest[NUM_OF_DIMENSIONS];
	double ave_time = 0;
	srand((unsigned)time(NULL));
	for (int count=0;count<20;count++)
	{
		//Initialize particles
		for (int i = 0; i < NUM_OF_PARTICLES*NUM_OF_DIMENSIONS; i++)
		{
			position[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
			pBests[i] = position[i];
			velocities[i] = 0;
		}

		for (int k = 0; k < NUM_OF_DIMENSIONS - 1; k++)
		{
			gBest[k] = pBests[k];
		}

		clock_t begin = clock();

		//PSO main function
		cuda_pso(position, velocities, pBests, gBest);

		clock_t end = clock();

		printf("==================== GPU%d =======================\n",count+1);

		printf("Time consumption: %10.3lf ms\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);

		// gBest minimum
		//for (int i = 0; i < NUM_OF_DIMENSIONS; i++)
		//printf("x%d = %f\n", i, gBest[i]);

		//printf("Minimum = %f\n", host_fitness_function(gBest));

		// ======================== END OF GPU ====================== //
	}
		
	
	

	system("PAUSE");
	return 0;
}
