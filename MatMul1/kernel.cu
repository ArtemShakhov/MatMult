
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE  32          
#define N           640

using namespace std;



cudaError_t addWithCuda(int *c, const int *a, const int *b, int time, int **ccpu);

__global__ void addKernel(const int *a, const int *b, int n, int * c)
{
	int   bx = blockIdx.x;     
	int   by = blockIdx.y;
	int   tx = threadIdx.x;        
	int   ty = threadIdx.y;
	float sum = 0;           
	int   ia = n * BLOCK_SIZE * by + n * ty;  
	int   ib = BLOCK_SIZE * bx + tx;

	
	for (int k = 0; k < n; k++)
		sum += a[ia + k] * b[ib + k * n];

	
	int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	c[ic + n * ty + tx] = sum;
}

int** CPU(int** a, int** b, int** c, int n) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < n; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	return c;
}

bool checkResult(int** a, int* b, int n) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (a[i][j] != b[N * i + j])
				return false;
	return true;
}


int main( int argc, char *  argv[] )
{
	setlocale(LC_ALL, "Russian");
	
   

	// allocate host memory
	int** acpu;
	int** bcpu;
	int** ccpu;
	acpu = new int*[N];
	for (int i = 0; i < N; i++)
		acpu[i] = new int[N];

	bcpu = new int*[N];
	for (int i = 0; i < N; i++)
		bcpu[i] = new int[N];

	ccpu = new int*[N];
	for (int i = 0; i < N; i++)
		ccpu[i] = new int[N];

	int * a = new int[N*N];
	int * b = new int[N*N];
	int * c = new int[N*N];


	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			int	k = N * i + j;
			a[k]= (rand()) % 10 + 1;
			b[k] = (rand()) % 10 + 1;
			acpu[i][j] = a[k];
			bcpu[i][j] = b[k];
			ccpu[i][j] = 0;
		}

	}
	clock_t time;
	time = clock();
	ccpu = CPU(acpu, bcpu, ccpu, N);
	time = clock() - time;
	


    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, time, ccpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    
	
	

    
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, int time, int **ccpu)
{
    int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	int numBytes = N * N * sizeof(int);
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        goto Error;
    }
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, numBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, numBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, numBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// asynchronously issue work to the GPU (all to stream 0)
	cudaEventRecord(start, 0);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<blocks, threads>>>(dev_a, dev_b, N, dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	
	
	if (checkResult(ccpu, c, N))
	{
		cout << "Âðåìÿ ðàáîòû íà CPU: " << time * 1000.0 / CLOCKS_PER_SEC << " ìèëëèñåêóíä" << endl;
		cout << "Âðåìÿ ðàáîòû íà GPU: " << gpuTime << " ìèëëèñåêóíä";
	}

Error:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

	delete a;
	delete b;
	delete c;

    
    return cudaStatus;
}
