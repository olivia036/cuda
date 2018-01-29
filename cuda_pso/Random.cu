#include "kernel.cuh"
int n = 200;

__device__ float generate(curandState* globalState, int ind)
{
	//int ind = threadIdx.x;  
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void kernel(float* N, curandState* globalState, int n)
{
	// generate random numbers  
	for (int i = 0; i < 40000; i++)
	{
		int k = generate(globalState, i) * 100000;
		while (k > n*n - 1)
		{
			k -= (n*n - 1);
		}
		N[i] = k;
	}
}

int main()
{
	const int N = 10;

	curandState* devStates;
	cudaMalloc(&devStates, N * sizeof(curandState));

	// setup seeds  
	setup_kernel << < 1, N >> > (devStates, time(NULL));

	float N2[N];
	float* N3;
	cudaMalloc((void**)&N3, sizeof(float)*N);

	kernel << <1, 1 >> > (N3, devStates, n);

	cudaMemcpy(N2, N3, sizeof(float)*N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		cout << N2[i] << endl;
	}

    system("PAUSE");
	return 0;
	
}