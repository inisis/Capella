#include <iostream>
#include <cstdio>

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "Utils/timer.h"

using std::cout;
using std::endl;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a;

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void resizeCudaKernel( unsigned char* input,
                                  float* output,
                                  const int outputWidth,
                                  const int outputHeight,
                                  const int inputWidth,
                                  const int inputHeight,
                                  const float pixelGroupSizeX,
                                  const float pixelGroupSizeY,
                                  const int inputChannels)
{
    //2D Index of current thread
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    const int pitchInput = inputWidth * inputChannels;
    const int pitchOutput = outputWidth * inputChannels;

    if((dx < outputWidth) && (dy < outputHeight))
    {
        if(inputChannels==1) { // grayscale image
        } else if(inputChannels==3) { // RGB image

            double scale_x = (double) inputWidth / outputWidth;
            double scale_y = (double) inputHeight / outputHeight;

            int xmax = outputWidth;

            float fx = (float)((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx = fx - sx;

            int isx1 = sx;
            if (isx1 < 0) {
                fx = 0.0;
                isx1 = 0;
            }
            if (isx1 >= (inputWidth - 1)) {
                xmax = ::min( xmax, dy);
                fx = 0;
                isx1 = inputWidth - 1;
            }

            float2 cbufx;
            cbufx.x = (1.f - fx);
            cbufx.y = fx;

            float fy = (float)((dy + 0.5) * scale_y - 0.5);
            int sy = floor(fy);
            fy = fy - sy;

            int isy1 = clip(sy - 1 + 1 + 0, 0, inputHeight);
            int isy2 = clip(sy - 1 + 1 + 1, 0, inputHeight);

            float2 cbufy;
            cbufy.x = (1.f - fy);
            cbufy.y = fy;

            int isx2 = isx1 + 1;

            float3 d0;

            float3 s11 = make_float3(input[(isy1 * inputWidth + isx1) * inputChannels + 0] , input[(isy1 * inputWidth + isx1) * inputChannels + 1] , input[(isy1 * inputWidth + isx1) * inputChannels + 2]);
            float3 s12 = make_float3(input[(isy1 * inputWidth + isx2) * inputChannels + 0] , input[(isy1 * inputWidth + isx2) * inputChannels + 1] , input[(isy1 * inputWidth + isx2) * inputChannels + 2]);
            float3 s21 = make_float3(input[(isy2 * inputWidth + isx1) * inputChannels + 0] , input[(isy2 * inputWidth + isx1) * inputChannels + 1] , input[(isy2 * inputWidth + isx1) * inputChannels + 2]);
            float3 s22 = make_float3(input[(isy2 * inputWidth + isx2) * inputChannels + 0] , input[(isy2 * inputWidth + isx2) * inputChannels + 1] , input[(isy2 * inputWidth + isx2) * inputChannels + 2]);

            float h_rst00, h_rst01;
            // B
            if( dy > xmax - 1)
            {
                h_rst00 = s11.x;
                h_rst01 = s21.x;
            }
            else
            {
                h_rst00 = s11.x * cbufx.x + s12.x * cbufx.y;
                h_rst01 = s21.x * cbufx.x + s22.x * cbufx.y;
            }
            // d0.x = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.x = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

            // G
            if( dy > xmax - 1)
            {
                h_rst00 = s11.y;
                h_rst01 = s21.y;
            }
            else
            {
                h_rst00 = s11.y * cbufx.x + s12.y * cbufx.y;
                h_rst01 = s21.y * cbufx.x + s22.y * cbufx.y;
            }
            // d0.y = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.y = h_rst00 * cbufy.x + h_rst01 * cbufy.y;
            // R
            if( dy > xmax - 1)
            {
                h_rst00 = s11.z;
                h_rst01 = s21.z;
            }
            else
            {
                h_rst00 = s11.z * cbufx.x + s12.z * cbufx.y;
                h_rst01 = s21.z * cbufx.x + s22.z * cbufx.y;
            }
            // d0.z = h_rst00 * (1 - fy) + h_rst01 * fy;
            d0.z = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

            output[(dy*outputWidth + dx) * 3 + 0 ] = (d0.x); // R
            output[(dy*outputWidth + dx) * 3 + 1 ] = (d0.y); // G
            output[(dy*outputWidth + dx) * 3 + 2 ] = (d0.z); // B
        } else {

        }
    }
}


extern "C" void downscaleCuda(const cv::Mat& input, cv::Mat& output)
{
    //Calculate total number of bytes of input and output image
    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    unsigned char *d_input;
    float *d_output;

    //Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_output, outputBytes), "CUDA Malloc Failed");

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input,input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

    GpuTimer timer;
    timer.Start();

    //Specify a reasonable block size
    const dim3 block(16,16);

    //Calculate grid size to cover the whole image
    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    // Calculate how many pixels in the input image will be merged into one pixel in the output image
    const float pixelGroupSizeY = float(input.rows) / float(output.rows);
    const float pixelGroupSizeX = float(input.cols) / float(output.cols);

    //Launch the size conversion kernel
    resizeCudaKernel<<<grid,block>>>(d_input, d_output, output.cols, output.rows, input.cols, input.rows, pixelGroupSizeX, pixelGroupSizeY, input.channels());

    timer.Stop();

    printf("Own Cuda code ran in: %f msecs.\n", timer.Elapsed());

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

    //Free the device memory
    SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

    return;
}