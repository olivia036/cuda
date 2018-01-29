#include "kernel.cuh"

float host_fitness_function(float x[])
{
	float res = 0;
	float y1 = 1 + (x[0] - 1) / 4;
	float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;

	res += pow(sin(phi*y1), 2);

	for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++)
	{
		float y = 1 + (x[i] - 1) / 4;
		float yp = 1 + (x[i + 1] - 1) / 4;

		res += pow(y - 1, 2)*(1 + 10 * pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
	}

	return res;
}

