/**
 * Created by desmond <desmond.yao@buaa.edu.cn> on 2018-11-18
 */

#include <iostream>
#include <string>
#include <stdio.h>
#include "./Utils/utils.h"
#include "./Utils/timer.h"

extern "C" void rgbaToGreyscaleCuda(unsigned char *d_rgbaImage, unsigned char* const d_greyImage,
                                    const size_t numRows, const size_t numCols);

void processUsingCuda(std::string input_file, std::string output_file);
void processUsingCvMat(std::string input_file, std::string output_file);

int main(int argc, char **argv) {

    // used for the allowed error between different implementations
    bool useEpsCheck = false; // set true to enable perPixelError and globalError
    double perPixelError = 3;
    double globalError   = 10;

    const std::string input_file = argc >= 2 ? argv[1] : "../data/10_1.jpg";
    const std::string output_file_OpenCvMat = argc >= 3 ? argv[2] : "../data/image_OpenCvCpu.jpg";
    const std::string output_file_Cuda = argc >= 4 ? argv[3] : "../data/image_Cuda.jpg";

    for (int i=0; i<1; ++i) {
        processUsingCvMat(input_file, output_file_OpenCvMat);
        processUsingCuda(input_file, output_file_Cuda);
    }

    // check if the generated images are the same
    compareImages(output_file_OpenCvMat, output_file_Cuda, useEpsCheck, perPixelError, globalError);

    return 0;
}

void processUsingCuda(std::string input_file, std::string output_file) {

    unsigned char *d_rgbImage;
    unsigned char *d_greyImage;

    cv::Mat image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        std::cerr << "Couldn't open file: " << input_file << std::endl;
        exit(1);
    }

    cv::Mat imageRGB;
    cv::Mat imageGrey;

    cv::cvtColor(image, imageRGB, CV_BGR2RGB);  // CV_BGR2GRAY

    imageGrey.create(image.rows, image.cols, CV_8UC1);

    if (!imageRGB.isContinuous() || !imageGrey.isContinuous()) {
        std::cerr << "Images aren't continuous!! Exiting." << std::endl;
        exit(1);
    }

    size_t numPixels = imageRGB.rows * imageRGB.cols * 3;
    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(&d_rgbImage, numPixels));
    checkCudaErrors(cudaMalloc(&d_greyImage, numPixels));

    checkCudaErrors(cudaMemcpy(d_rgbImage, imageRGB.ptr(), numPixels, cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.Start();

    rgbaToGreyscaleCuda(d_rgbImage, d_greyImage, imageRGB.rows, imageRGB.cols);

    timer.Stop();

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    printf("Implemented CUDA code ran in: %f msecs.\n", timer.Elapsed());

    checkCudaErrors(cudaMemcpy(imageGrey.ptr(), d_greyImage, numPixels / 3, cudaMemcpyDeviceToHost));

    cv::imwrite(output_file.c_str(), imageGrey);
}

void processUsingCvMat(std::string input_file, std::string output_file)
{
    cv::Mat image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);

    if (image.empty())
    {
        std::cerr << "Couldn't open file: " << input_file << std::endl;
        exit(1);
    }

    cv::Mat gray;
    GpuTimer timer;
    timer.Start();
    cv::cvtColor(image, gray, CV_BGR2GRAY);  // CV_BGR2GRAY

    timer.Stop();

    printf("OpenCV code ran in: %f msecs.\n", timer.Elapsed());

    cv::imwrite(output_file.c_str(), gray);
}
