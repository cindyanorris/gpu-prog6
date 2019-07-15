#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
//config.h defines the TILE_WIDTH 
//and the constants: SIMPLE, TILED, TILED2
//that indicate which kernel to launch
#include "config.h" 
#include "d_matMultiply.h"

//prototypes for kernels in this file
__global__ 
void d_matMultiplySimpleKernel(float * d_matrixM, float * d_matrixN, 
                               float * d_result, int width);

__global__ 
void d_matMultiplyTiledKernel(float * d_matrixM, float * d_matrixN, 
                              float * d_result, int width);

__global__ 
void d_matMultiplyTiled2Kernel(float * d_matrixM, float * d_matrixN, 
                               float * d_result, int width);

/*  d_matMultiply
    This function prepares and invokes a kernel to perform
    matrix multiplication (matrixM X matrixN) on the GPU.   
    The matrices have been linearized so each array is
    1D and contains width * width elements.
    Inputs:
    matrixM - points to matrixM data
    matrixN - points to matrixN data
    result - points to the matrix to hold the result
    width - width and height of the input and result matrices
    which - indicates which kernel to use (SIMPLE, TILED, TILED2)
*/
float d_matMultiply(float * matrixM, float * matrixN, float * result, 
                    int width, int which)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //Your work goes here
    //kernel calls provided but you need to write the code for the
    //memory allocations, etc. and define the grid and the block
    //Use TILE_SIZE (defined in config.h)
/*
    if (which == SIMPLE)
    {
        d_matMultiplySimpleKernel<<<grid, block>>>(d_matrixM, d_matrixN, 
                                                   d_result, width);
    }
    else if (which == TILED)
    {
        d_matMultiplyTiledKernel<<<grid, block>>>(d_matrixM, d_matrixN, 
                                                  d_result, width);
    }                                             
    else if (which == TILED2)
    {
        d_matMultiplyTiled2Kernel<<<grid, block>>>(d_matrixM, d_matrixN, 
    }
*/


    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;
}

/*  
    d_matMultiplySimpleKernel
    This kernel performs matrix multiplication of matrixM and matrixN
    (d_matrixM X d_matrixN) and stores the result in d_result.
    All three matrices are of size width by width and have been linearized.
    Each thread calculates one output element.  All of the elements
    needed for the dot-product calculation are accessed from global
    memory.
    Inputs:
    d_matrixM - pointer to the array containing matrixM
    d_matrixN - pointer to the array containing matrixN
    d_result - pointer to the array in the global memory to hold the result
               of the matrix multiply
    width - width and height of the matrices
*/
__global__ void d_matMultiplySimpleKernel(float * d_matrixM, float * d_matrixN,
                                          float * d_result, int width) 
{
}      

/*  
    d_matMultiplyTiledKernel
    This kernel performs matrix multiplication of matrixM and matrixN
    (d_matrixM X d_matrixN) and stores the result in d_result.
    All three matrices are of size width by width and have been linearized.
    Each thread calculates one output element.  Each thread in
    a block cooperates in loading a tile of matrixN and matrixM elements into 
    shared memory and then performs the dot-product calculation using
    the values in the shared memory.  When the threads are finished
    with the current tile, they then load the next tile that is needed.
    At the end, all threads in a block have calculated the results
    of TILE_SIZE by TILE_SIZE output elements.
    Inputs:
    d_matrixM - pointer to the array containing matrixM
    d_matrixN - pointer to the array containing matrixN
    d_result - pointer to the array in the global memory to hold the result
               of the matrix multiply
    width - width and height of the matrices
*/
__global__ 
void d_matMultiplyTiledKernel(float * d_matrixM, float * d_matrixN,
                              float * d_result, int width) 
{
}      

/*  
    d_matMultiplyTiled2Kernel
    This kernel performs matrix multiplication of matrixM and matrixN
    (d_matrixM X d_matrixN) and stores the result in d_result.
    All three matrices are of size width by width and have been linearized.
    Each thread in a block cooperates in loading a tile of matrixN and 
    matrixM elements into shared memory and then performs the dot-product 
    calculation using the values in the shared memory.  Every thread in 
    the thread block computes 2 results using the values in the shared
    memory.  At the end, all threads in a block have calculated the results
    for TILE_SIZE by TILE_SIZE output elements.
    This implementation is described on page 128 in the textbook.
    Inputs:
    d_matrixM - pointer to the array containing matrixM
    d_matrixN - pointer to the array containing matrixN
    d_result - pointer to the array in the global memory to hold the result
               of the matrix multiply
    width - width and height of the matrices
*/
__global__ 
void d_matMultiplyTiled2Kernel(float * d_matrixM, float * d_matrixN,
                               float * d_result, int width) 
{

}      

