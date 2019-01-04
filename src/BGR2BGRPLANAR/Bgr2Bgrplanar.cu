#include<iostream>
#include<cstdio>
#include "opencv2/opencv.hpp"
#include "Utils/timer.h"

#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)

#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

#define LOG_CUDA "[cuda]   "

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
    if( retval == cudaSuccess)
        return cudaSuccess;
#endif

    //int activeDevice = -1;
    //cudaGetDevice(&activeDevice);

    //Log("[cuda]   device %i  -  %s\n", activeDevice, txt);

    printf(LOG_CUDA "%s\n", txt);


    if( retval != cudaSuccess )
    {
        printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
        printf(LOG_CUDA "   %s:%i\n", file, line);
    }

    return retval;
}

// Convert an image (unsigned char) from being interleaved by pixel, to planar band-sequential FP32 BGR
template <typename T> __global__ void gpuPackedToPlanarBGR(T* input, int iWidth, float* output, int oWidth, int oHeight)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = oWidth * oHeight;

    if( dx >= oWidth || dy >= oHeight )
        return;

    const float3 bgr = make_float3(input[(dy * iWidth + dx) * 3 + 0] , input[(dy * iWidth + dx) * 3 + 1] , input[(dy * iWidth + dx) * 3 + 2]);

//    output[n * 0 + dy * oWidth + dx] = (bgr.x);
//    output[n * 1 + dy * oWidth + dx] = (bgr.y);
//    output[n * 2 + dy * oWidth + dx] = (bgr.z);

    output[n * 0 + dy * oWidth + dx] = (bgr.x - 127.5) / 128;
    output[n * 1 + dy * oWidth + dx] = (bgr.y - 127.5) / 128;
    output[n * 2 + dy * oWidth + dx] = (bgr.z - 127.5) / 128;
}


// cudaPackedToPlanarBGR
extern "C" cudaError_t cudaPackedToPlanarBGR(float* input, size_t inputWidth, size_t inputHeight,
                                   float* output, size_t outputWidth, size_t outputHeight)
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPackedToPlanarBGR<float><<<gridDim, blockDim>>>(input, inputWidth, output, outputWidth, outputHeight);

    return CUDA(cudaGetLastError());
}

extern "C" void bgr2bgrplanarCuda(const cv::Mat& input, cv::Mat& output)
{
    //Calculate total number of bytes of input and output image
    const int inputBytes = input.cols * input.channels() * input.rows * sizeof(float);
    const int outputBytes = output.cols * input.channels() * output.rows * sizeof(float);

    float *d_input;
    float *d_output;

    //Allocate device memory
    SAFE_CALL(cudaMalloc<float>(&d_input, inputBytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_output, outputBytes), "CUDA Malloc Failed");

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input,input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

    GpuTimer timer;
    timer.Start();

    cudaPackedToPlanarBGR(d_input, input.cols, input.rows, d_output, output.cols, output.rows);

    timer.Stop();

    printf("Own Cuda code ran in: %f msecs.\n", timer.Elapsed());

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

    //Free the device memory
    SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}