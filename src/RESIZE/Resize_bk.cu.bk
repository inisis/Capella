#ifndef RESIZE_CUH_
#define BILATERAL_FILTER_CUH_

#include<iostream>
#include<cstdio>
#include<cuda_runtime.h>

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

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void resizeCudaKernel( unsigned char* input,
                                  unsigned char* output,
                                  const int outputWidth,
                                  const int outputHeight,
                                  const int inputWidthStep,
                                  const int outputWidthStep,
                                  const float pixelGroupSizeX,
                                  const float pixelGroupSizeY,
                                  const int inputChannels)
{
    //2D Index of current thread
    const int outputXIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputYIndex = blockIdx.y * blockDim.y + threadIdx.y;

    //Only valid threads perform memory I/O
    if((outputXIndex<outputWidth) && (outputYIndex<outputHeight))
    {
        // Starting location of current pixel in output
        int output_tid  = outputYIndex * outputWidthStep + (outputXIndex * inputChannels);

        // Compute the size of the area of pixels to be resized to a single pixel
        const float pixelGroupArea = pixelGroupSizeX * pixelGroupSizeY;

        // Compute the pixel group area in the input image
        const int intputXIndexStart = static_cast<int>(outputXIndex * pixelGroupSizeX + 0.5f);
        const int intputXIndexEnd = static_cast<int>(intputXIndexStart + pixelGroupSizeX + 0.5f);
        const int intputYIndexStart = static_cast<int>(outputYIndex * pixelGroupSizeY + 0.5f);
        const int intputYIndexEnd = static_cast<int>(intputYIndexStart + pixelGroupSizeY + 0.5f);

        if(inputChannels==1) { // grayscale image
            float channelSum = 0;
            for(int intputYIndex=intputYIndexStart; intputYIndex<intputYIndexEnd; ++intputYIndex) {
                for(int intputXIndex=intputXIndexStart; intputXIndex<intputXIndexEnd; ++intputXIndex) {
                    int input_tid = intputYIndex * inputWidthStep + intputXIndex;
                    channelSum += input[input_tid];
                }
            }
            output[output_tid] = static_cast<unsigned char>(channelSum / pixelGroupArea);
        } else if(inputChannels==3) { // RGB image
            float channel1stSum = 0;
            float channel2stSum = 0;
            float channel3stSum = 0;
            for(int intputYIndex=intputYIndexStart; intputYIndex<intputYIndexEnd; ++intputYIndex) {
                for(int intputXIndex=intputXIndexStart; intputXIndex<intputXIndexEnd; ++intputXIndex) {
                    // Starting location of current pixel in input
                    int input_tid = intputYIndex * inputWidthStep + intputXIndex * inputChannels;
                    channel1stSum += input[input_tid];
                    channel2stSum += input[input_tid+1];
                    channel3stSum += input[input_tid+2];
                }
            }
            output[output_tid] = static_cast<unsigned char>(channel1stSum / pixelGroupArea);
            output[output_tid+1] = static_cast<unsigned char>(channel2stSum / pixelGroupArea);
            output[output_tid+2] = static_cast<unsigned char>(channel3stSum / pixelGroupArea);
        } else { // arbitrary number of channels
            // Compute the pixel group area in the input image
            /*const float intputXIndexStart = outputXIndex * pixelGroupSizeX;
            const int intputXIndexStartFloor = floor(intputXIndexStart);
            const float intputXIndexEnd = intputXIndexStart + pixelGroupSizeX;
            const int intputXIndexEndCeil = ceil(intputXIndexEnd);

            const float intputYIndexStart = outputYIndex * pixelGroupSizeY;
            const int intputYIndexStartFloor = floor(intputYIndexStart);
            const float intputYIndexEnd = intputYIndexStart + pixelGroupSizeY;
            const int intputYIndexEndCeil = ceil(intputYIndexEnd);

            // Initialize accumulators for maximum pixel channel sums: (red, green, blue, alpha), (red, green, blue) or (gray)
            float channelSums[] = {0, 0, 0, 0};

            for(int intputYIndex=intputYIndexStartFloor; intputYIndex<intputYIndexEndCeil; ++intputYIndex) {
                for(int intputXIndex=intputXIndexStartFloor; intputXIndex<intputXIndexEndCeil; ++intputXIndex) {

                    // Compute the weight of the current pixel
                    //float weight = 1.0;
                    if(intputXIndex < intputXIndexStart) {
                        weight *= 1 - (intputXIndexStart - intputXIndex); // use only the part till next integer
                    } else if(intputXIndex>intputXIndexEnd) {
                        weight *= 1 - (intputXIndex - intputXIndexEnd); // use only the part from integer till intputXIndexEnd
                    }
                    if(intputYIndex<intputYIndexStart) {
                        weight *= 1 - (intputYIndexStart - intputYIndex); // use only the part till next integer
                    } else if(intputYIndex>intputYIndexEnd) {
                        weight *= 1 - (intputYIndex - intputYIndexEnd); // use only the part from integer till intputXIndexEnd
                    }

                    // Starting location of current pixel in input
                    int input_tid = intputYIndex * inputWidthStep + intputXIndex * inputChannels;
                    for(int channelIndex = 0; channelIndex<inputChannels; ++channelIndex, ++input_tid) {
                        // add each color channel of the current pixel to the sum
                        channelSums[channelIndex] += input[input_tid]; // * weight;
                    }
                }
            }*/
        }
    }
}

void downscaleCuda(const cv::Mat& input, cv::Mat& output)
{
    //Calculate total number of bytes of input and output image
    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    unsigned char *d_input, *d_output;

    //Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input,inputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output,outputBytes),"CUDA Malloc Failed");

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    GpuTimer timer;
    timer.Start();

    //Specify a reasonable block size
    const dim3 block(16,16);

    //Calculate grid size to cover the whole image
    const dim3 grid((output.cols + block.x - 1)/block.x, (output.rows + block.y - 1)/block.y);

    // Calculate how many pixels in the input image will be merged into one pixel in the output image
    const float pixelGroupSizeY = float(input.rows) / float(output.rows);
    const float pixelGroupSizeX = float(input.cols) / float(output.cols);

    //Launch the size conversion kernel
    resizeCudaKernel<<<grid,block>>>(d_input,d_output,output.cols,output.rows,input.step,output.step, pixelGroupSizeX, pixelGroupSizeY, input.channels());

    timer.Stop();
    printf("Own Cuda code ran in: %f msecs.\n", timer.Elapsed());

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(output.ptr(),d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    //Free the device memory
    SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
    //SAFE_CALL(cudaDeviceReset(),"CUDA Device Reset Failed");
}

#endif /* RESIZE_CUH_ */