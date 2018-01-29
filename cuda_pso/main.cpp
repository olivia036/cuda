#include <iostream>
#include <math.h>
#include "kernel.h"

using namespace std;

int main(int argc, char** argv)
{
	//particles
	float position[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
	float velocities[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
	float pBests[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];

	//gBest
	float gBest[NUM_OF_DIMENSIONS];

	srand((unsigned)time(NULL));

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

	printf("==================== GPU =======================\n");

	printf("Time consumption: %10.3lf ms\n", (double)(end - begin) / CLOCKS_PER_SEC);

	// gBest berisi nilai minimum
	for (int i = 0; i < NUM_OF_DIMENSIONS; i++)
		printf("x%d = %f\n", i, gBest[i]);

	printf("Minimum = %f\n", host_fitness_function(gBest));

	// ======================== END OF GPU ====================== //

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
	pso(position, velocities, pBests, gBest);

	clock_t end = clock();

	printf("==================== CPU =======================\n");

	printf("Time consumption: %10.3lf ms\n", (double)(end - begin) / CLOCKS_PER_SEC);

	// gBest berisi nilai minimum
	for (int i = 0; i < NUM_OF_DIMENSIONS; i++)
		printf("x%d = %f\n", i, gBest[i]);

	printf("Minimum = %f\n", host_fitness_function(gBest));

	// ======================== END OF CPU ====================== //
	system("PAUSE");
	return 0;
}