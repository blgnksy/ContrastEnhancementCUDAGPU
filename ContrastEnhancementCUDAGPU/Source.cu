#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
#include <assert.h>
#include <math.h>
#include <windows.h>

// CUDA error checking Macro.
#define CUDA_CALL(x,y) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}\
  else{printf("CUDA Success at %d. (%s)\n",__LINE__,y); }}

//Global
#define DIM 256

// Function Prototypes.
Npp8u *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);

void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray);

__global__ void
MinMaxKernel(Npp8u * pSrc_Dev, Npp8u * pMin_Dev, Npp8u * pMax_Dev);

__global__ void
SubtractKernel(Npp8u * pDst_Dev, Npp8u * pSrc_Dev, Npp8u nMin_Dev);

__global__ void
MultiplyKernel(Npp8u * pDst_Dev, Npp8u nConstant, int normalizer);

void StartCounter();

double GetCounter();


int
main(int argc, char ** argv)
{
	// Host parameter declarations.	
	Npp8u * pSrc_Host, *pDst_Host;
	int   nWidth, nHeight, nMaxGray;

	// Load image to the host.
	std::cout << "Load PGM file." << std::endl;
	pSrc_Host = LoadPGM("C:\\Users\\blgnksy\\source\\repos\\CudaAssignment2\\ColorEnhancement\\lena_before.pgm", nWidth, nHeight, nMaxGray);
	pDst_Host = new Npp8u[nWidth * nHeight];

	// Device parameter declarations.
	Npp8u	 * pSrc_Dev, *pDst_Dev;
	Npp8u    * pMin_Dev, *pMax_Dev;
	Npp8u    nMin_Host[DIM], nMax_Host[DIM];
	int		 nSrcStep_Dev, nDstStep_Dev;

	//Start Counter.
	cudaEvent_t start, stop;
	float elapsed_time_ms;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Allocate and Copy the Device Variables
	pSrc_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nSrcStep_Dev);
	pDst_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nDstStep_Dev);

	// Device variables are copied to device. 
	CUDA_CALL(cudaMalloc(&pMin_Dev, DIM * sizeof(Npp8u)), "Memory allocated.");
	CUDA_CALL(cudaMalloc(&pMax_Dev, DIM * sizeof(Npp8u)), "Memory allocated.");
	CUDA_CALL(cudaMemcpy(pSrc_Dev, pSrc_Host, nWidth * nHeight * sizeof(Npp8u), cudaMemcpyHostToDevice), "Memory copied.(HostToDevice)");

	/*
	Defining Kernel Execution Paramaters.
	I defined two different block size to be able to find global minimum. During the First Max and Min kernels execution, they are only
	be able to find local minimum.
	*/
	dim3 dimGrid(nWidth);
	dim3 dimBlock2(nWidth / 2);
	dim3 dimBlock1(nWidth);

	//Minimum kernel and Maximum kernels are independent. So no need to put them into same stream.
	size_t sharedMemSize = nHeight *  nWidth * sizeof(Npp8u);

	// One kernel for both min and max.
	MinMaxKernel << <dimGrid, dimBlock2, sharedMemSize >> > (pSrc_Dev, pMin_Dev, pMax_Dev);
	MinMaxKernel << <1, dimBlock1, sharedMemSize >> > (pMin_Dev, pMin_Dev, pMax_Dev);

	// Minimum and maximum values are copied to host.
	CUDA_CALL(cudaMemcpy(&nMin_Host, pMin_Dev, sizeof(Npp8u), cudaMemcpyDeviceToHost), "Memory copied.(DeviceToHost)");
	CUDA_CALL(cudaMemcpy(&nMax_Host, pMax_Dev, sizeof(Npp8u), cudaMemcpyDeviceToHost), "Memory copied.(DeviceToHost)");

	//Just for control.
	assert(nMin_Host[0] = 92);
	assert(nMax_Host[0] = 202);

	// Subtracting the minimum
	SubtractKernel << <dimGrid, dimBlock1 >> > (pDst_Dev, pSrc_Dev, nMin_Host[0]);

	// Provided code from Original work.
	int nScaleFactor = 0;
	int nPower = 1;
	while (nPower * 255.0f / (nMax_Host[0] - nMin_Host[0]) < 255.0f)
	{
		nScaleFactor++;
		nPower *= 2;
	}
	Npp8u nConstant = static_cast<Npp8u>(255.0f / (nMax_Host[0] - nMin_Host[0]) * (nPower / 2));

	// CUDA Kernel doesn't support these calculation. So that I calculated it outside the kernel. 
	int normalizer = pow(2, (nScaleFactor - 1));

	// Multiply constant and 
	MultiplyKernel << <dimGrid, dimBlock1 >> > (pDst_Dev, nConstant, normalizer);

	CUDA_CALL(cudaMemcpy(pDst_Host, pDst_Dev, nWidth * nHeight * sizeof(Npp8u), cudaMemcpyDeviceToHost), "Memory copied.(DeviceToHost)");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("Time to calculate results(GPU Time): %f ms.\n", elapsed_time_ms);

	// Output the result image.
	std::cout << "Output the PGM file." << std::endl;
	WritePGM("C:\\Users\\blgnksy\\source\\repos\\CudaAssignment2\\ColorEnhancement\\lena_after_GPU.pgm", pDst_Host, nWidth, nHeight, nMaxGray);

	// Clean up.
	delete[] pSrc_Host;
	delete[] pDst_Host;

	nppiFree(pSrc_Dev);
	nppiFree(pDst_Dev);
	CUDA_CALL(cudaFree(pMin_Dev), "Memory Freed.");
	CUDA_CALL(cudaFree(pMax_Dev), "Memory Freed.");
	printf("All done. Press Any Key to Continue...");
	getchar();
	return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )

// Load PGM file.
Npp8u *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{
	char aLine[256];
	FILE * fInput = fopen(sFileName, "r");
	if (fInput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	// First line: version
	fgets(aLine, 256, fInput);
	std::cout << "\tVersion: " << aLine;
	// Second line: comment
	fgets(aLine, 256, fInput);
	std::cout << "\tComment: " << aLine;
	fseek(fInput, -1, SEEK_CUR);
	// Third line: size
	fscanf(fInput, "%d", &nWidth);
	std::cout << "\tWidth: " << nWidth;
	fscanf(fInput, "%d", &nHeight);
	std::cout << " Height: " << nHeight << std::endl;
	// Fourth line: max value
	fscanf(fInput, "%d", &nMaxGray);
	std::cout << "\tMax value: " << nMaxGray << std::endl;
	while (getc(fInput) != '\n');
	// Following lines: data
	Npp8u * pSrc_Host = new Npp8u[nWidth * nHeight];
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			pSrc_Host[i*nWidth + j] = fgetc(fInput);
	fclose(fInput);

	return pSrc_Host;
}

// Write PGM image.
void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray)
{
	FILE * fOutput = fopen(sFileName, "w+");
	if (fOutput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	char * aComment = "# Created by CUDA Assignment II";
	fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			fputc(pDst_Host[i*nWidth + j], fOutput);
	fclose(fOutput);
}

//
__global__ void
MinMaxKernel(Npp8u * pSrc_Dev, Npp8u * pMin_Dev, Npp8u * pMax_Dev)
{
	extern __shared__ Npp8u sMin[];
	extern __shared__ Npp8u sMax[];
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	if (gid < 512)
	{
		if (pSrc_Dev[gid] > pSrc_Dev[gid + blockDim.x])
		{
			sMin[tid] = pSrc_Dev[gid + blockDim.x];
			sMax[tid] = pSrc_Dev[gid];
		}
		else
		{
			sMin[tid] = pSrc_Dev[gid];
			sMax[tid] = pSrc_Dev[gid + blockDim.x];
		}
		__syncthreads();
	}
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
			if (sMin[tid] > sMin[tid + s]) sMin[tid] = sMin[tid + s];
		if (sMax[tid] < sMax[tid + s]) sMax[tid] = sMax[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		if (sMin[tid] > sMin[tid + 32]) sMin[tid] = sMin[tid + 32];
		if (sMin[tid] > sMin[tid + 16]) sMin[tid] = sMin[tid + 16];
		if (sMin[tid] > sMin[tid + 8]) sMin[tid] = sMin[tid + 8];
		if (sMin[tid] > sMin[tid + 4]) sMin[tid] = sMin[tid + 4];
		if (sMin[tid] > sMin[tid + 2]) sMin[tid] = sMin[tid + 2];
		if (sMin[tid] > sMin[tid + 1]) sMin[tid] = sMin[tid + 1];

		if (sMax[tid] < sMax[tid + 32]) sMax[tid] = sMax[tid + 32];
		if (sMax[tid] < sMax[tid + 16]) sMax[tid] = sMax[tid + 16];
		if (sMax[tid] < sMax[tid + 8]) sMax[tid] = sMax[tid + 8];
		if (sMax[tid] < sMax[tid + 4]) sMax[tid] = sMax[tid + 4];
		if (sMax[tid] < sMax[tid + 2]) sMax[tid] = sMax[tid + 2];
		if (sMax[tid] < sMax[tid + 1]) sMax[tid] = sMax[tid + 1];
	}
	if (tid == 0)
	{
		pMin_Dev[blockIdx.x] = sMin[0];
		pMax_Dev[blockIdx.x] = sMax[0];
	}
}

// Subtract Min from Source and set it to Destination
__global__ void
SubtractKernel(Npp8u * pDst_Dev, Npp8u * pSrc_Dev, Npp8u nMin_Dev)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	pDst_Dev[i] = pSrc_Dev[i] - nMin_Dev;
}

// multiply by nConstant and divide by 2 ^ nScaleFactor-1
__global__ void
MultiplyKernel(Npp8u * pDst_Dev, Npp8u nConstant, int normalizer)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	pDst_Dev[i] = static_cast<Npp8u>(pDst_Dev[i] * nConstant / normalizer);
}