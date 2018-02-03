// MP 1
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include	<wb.h>
#include "wbCheck.h"

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
    {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;
    float * hostIncpy1;
    float * hostIncpy2;

    int devCount = -1;
    int curDevice = -2;
    cudaGetDeviceCount(&devCount);
    cudaGetDevice(&curDevice);
    wbLog(TRACE, "device count=", devCount, "\t\tcurDevice=", curDevice);
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    hostIncpy1 = (float *)malloc(inputLength * sizeof(float));
    hostIncpy2 = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    int size = inputLength * sizeof(float);
    wbLog(TRACE, "inLength=", inputLength, " size=", size);
    wbCheck (cudaMalloc((void**)&deviceInput1, size));
    wbCheck(cudaMalloc((void**)&deviceInput2, size));
    wbCheck(cudaMalloc((void**)&deviceOutput, size));


    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int blockx = 256;
    //int numblks = ceil(inputLength / (float)blockx);
    int numblks = (inputLength + blockx - 1) / blockx;
    wbLog(TRACE, "blockx=", blockx, " numblks=", numblks);

    dim3 DimGrid(numblks, 1, 1);
    dim3 DimBlock(blockx, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    //@@ Launch the GPU Kernel here
    //vecAdd <<<numblks, blockx>>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
    vecAdd <<<DimGrid, DimBlock >>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostIncpy1, deviceInput1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostIncpy2, deviceInput2, size, cudaMemcpyDeviceToHost);

    for (int ix = 0; ix < inputLength; ix++)
    {
        cout << hostIncpy1[ix] << "\t" << hostIncpy2[ix] << "\t" << hostOutput[ix] << endl;
    }

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

