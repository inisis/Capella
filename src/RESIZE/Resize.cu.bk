#ifndef RESIZE_CUH_
#define BILATERAL_FILTER_CUH_

#include<iostream>
#include<cstdio>

using std::cout;
using std::endl;

const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

#define alignSize(sz, n) ((sz + n-1) & -n)
#define clip(x, a, b) (x >= a ? (x < b ? x : b-1) : a)


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

__global__ void HResizeLinear(  unsigned char* input_first,
                                unsigned char* input_second,
                                unsigned char* output_first,
                                unsigned char* output_second,
                                const int count,
                                const int* xofs,
                                const int* alpha,
                                const int swidth,
                                const int dwidth,
                                const int cn,
                                const int xmin,
                                const int xmax)
{
    int dx, k, dx0 = 0;
    for( k = 0; k < count -2 ; k++)
    {
        for( dx = dx0; dx < xmax; dx++ )
        {
            int sx = xofs[dx];
            int a0 = alpha[dx*2];
            int a1 = alpha[dx*2+1];
            int t0 = input_first[sx]*a0 + input_first[sx + cn]*a1;
            int t1 = input_second[sx]*a0 + input_second[sx + cn]*a1;
            output_first[dx] = t0; output_second[dx] = t1;
        }

        for( ; dx < dwidth; dx++ )
        {
            int sx = xofs[dx];
            output_first[dx] = int(input_first[sx] * INTER_RESIZE_COEF_SCALE);
            output_second[dx] = int(input_second[sx] * INTER_RESIZE_COEF_SCALE);
        }
    }


    for( dx = 0; dx < xmax; dx++ )
    {
        int sx = xofs[dx];
        output_first[dx] = input_first[sx]*alpha[dx*2] + input_first[sx+cn]*alpha[dx*2+1];
    }

    for( ; dx < dwidth; dx++ )
        output_first[dx] = int(input_first[xofs[dx]]*INTER_RESIZE_COEF_SCALE);


    for( dx = 0; dx < xmax; dx++ )
    {
        int sx = xofs[dx];
        output_second[dx] = input_second[sx]*alpha[dx*2] + input_second[sx+cn]*alpha[dx*2+1];
    }

    for( ; dx < dwidth; dx++ )
        output_second[dx] = int(input_second[xofs[dx]]*INTER_RESIZE_COEF_SCALE);


}


__global__ void VResizeLinear(  unsigned char* input_first,
                                unsigned char* input_second,
                                unsigned char* output,
                                const int width,
                                const int* beta)
{
    int x = ((width >> 4) << 4);
    int b0 = beta[0];
    int b1 = beta[1];
    int ksize = 2;

    for( ; x <= width - 4; x += 4 )
    {
        output[x+0] = uchar(( ((b0 * (input_first[x+0] >> 4)) >> 16) + ((b1 * (input_second[x+0] >> 4)) >> 16) + 2)>>2);
        output[x+1] = uchar(( ((b0 * (input_first[x+1] >> 4)) >> 16) + ((b1 * (input_second[x+1] >> 4)) >> 16) + 2)>>2);
        output[x+2] = uchar(( ((b0 * (input_first[x+2] >> 4)) >> 16) + ((b1 * (input_second[x+2] >> 4)) >> 16) + 2)>>2);
        output[x+3] = uchar(( ((b0 * (input_first[x+3] >> 4)) >> 16) + ((b1 * (input_second[x+3] >> 4)) >> 16) + 2)>>2);
    }

    for( ; x < width; x++ )
        output[x] = uchar(( ((b0 * (input_first[x] >> 4)) >> 16) + ((b1 * (input_second[x] >> 4)) >> 16) + 2)>>2);
}

__global__ void ResizeGeneric(  unsigned char* input,
                                unsigned char* output,
                                unsigned char* buffer_,
                                unsigned char* buffer_generic,
                                const int buffer_generic_step,
                                const int outputWidth,
                                const int outputHeight,
                                const int inputWidth,
                                const int inputHeight,
                                const int inputChannels,
                                const int xmin,
                                const int xmax)
{
    int ksize = 2;
    int dy;
    int cn = inputChannels;
    int width = outputWidth * inputChannels;
    int* xofs = (int*)(unsigned char*)buffer_;
    int* yofs = xofs + width;
    short* ialpha = (short*)(yofs + outputHeight);
    short* ibeta = ialpha + width * ksize;

    int2 prev_sy;
    prev_sy.x = -1;
    prev_sy.y = -1;

    int *rows_first  = (int *)buffer_generic;
    int *rows_second = (int *)buffer_generic + buffer_generic_step;

    const short* beta = ibeta;

    for( dy = 0; dy < outputHeight; dy++, beta += ksize )
    {
        int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (int k = 0; k < ksize; k++)
        {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, inputHeight);
            for (k1 = ::max(k1, k); k1 < ksize; k1++) {
                int tmp = k1 % 2 == 0 ? prev_sy.x : prev_sy.y;
                if (k1 < 12 && sy == tmp) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        //memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = ::min(k0, k); // remember the first row that needs to be computed
            //srows[k] = src.template ptr<T>(sy);
            if(k == 0)
                prev_sy.x = sy;
            else
                prev_sy.y = sy;
        }

        //if (k0 < ksize)

    }
}

__global__ void resizeCudaKernel( unsigned char* input,
                                  unsigned char* output,
                                  unsigned char* buffer_,
                                  const int outputWidth,
                                  const int outputHeight,
                                  const int inputWidth,
                                  const int inputHeight,
                                  const float scale_x,
                                  const float scale_y,
                                  const int inputChannels)
{
    //2D Index of current thread
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    const int pitchInput = inputWidth * inputChannels;

    int k, sx, sy, width, ksize, cn, xmin, xmax;
    cn = inputChannels;
    ksize = 2;
    width = outputWidth * inputChannels;
    int* xofs = (int*)(unsigned char*)buffer_;
    int* yofs = xofs + width;
    float* alpha = (float*)(yofs + outputHeight);
    short* ialpha = (short*)alpha;
    float* beta = alpha + width * ksize;
    short* ibeta = ialpha + width * ksize;

    float2 cbuf;

    if((dx<outputWidth) && (dy<outputHeight))
    {
        // Starting location of current pixel in output

        if(inputChannels==1) { // grayscale image
        } else if(inputChannels==3) { // RGB image

            float fx = ((float) dx + 0.5f) * scale_x - 0.5f;
            float sx = floor(fx);
            fx = fx - sx;

            int isx1 = (int) sx;
            if (isx1 < 0) {
                xmin = dx + 1;
            }
            if (isx1 > (inputWidth - 1)) {
                xmax = ::min(xmax, dx);
            }

            for (k = 0, sx = sx * inputChannels; k < inputChannels; k++)
                xofs[dx * inputChannels + k] = sx + k;

            cbuf.x = 1.0f - fx;
            cbuf.y = fx;

            ialpha[dx * cn * ksize] = saturate(cbuf.x * INTER_RESIZE_COEF_SCALE);
            ialpha[dx * cn * ksize + 1] = saturate(cbuf.y * INTER_RESIZE_COEF_SCALE);

            for (; k < cn * ksize; k++)
                ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];

            float fy = ((float) dy + 0.5f) * scale_y - 0.5f;
            float sy = floor(fy);
            fy = fy - sy;

            yofs[dy] = sy;

            cbuf.x = 1.0f - fy;
            cbuf.y = fy;

            ibeta[dy * ksize] = saturate(cbuf.x * INTER_RESIZE_COEF_SCALE);
            ibeta[dy * ksize + 1] = saturate(cbuf.y * INTER_RESIZE_COEF_SCALE);
        }
    }
}

void downscaleCuda(const cv::Mat& input, cv::Mat& output)
{
    //Calculate total number of bytes of input and output image
    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;

    const int bufferBytes = (output.cols * output.channels() + output.rows) * (sizeof(int) + sizeof(float)* 2);

    const int bufstep = (int)alignSize(output.cols * output.channels(), 16);

    unsigned char *d_input, *d_output;

    unsigned char *buffer_;

    unsigned char *buffer_generic_;

    //Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&buffer_, bufferBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&buffer_generic_, bufstep*2),"CUDA Malloc Failed");

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
    resizeCudaKernel<<<grid,block>>>(d_input, d_output, buffer_, output.cols, output.rows, input.cols, input.rows, pixelGroupSizeX, pixelGroupSizeY, input.channels());

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
