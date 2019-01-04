/**
 * Created by desmond <desmond.yao@buaa.edu.cn> on 2018-11-18
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "./Utils/timer.h"
#include "./Utils/utils.h"

using namespace std;
using namespace cv;

#define HEIGHT  4
#define WIDTH   4
#define CHANNEL 3

extern "C" void bgr2bgrplanarCuda(const cv::Mat& input, cv::Mat& output);

extern "C" void downscaleCuda(const cv::Mat& input, cv::Mat& output);

void processUsingOpenCvCpu(std::string nput_file, std::string output_file);
//void processUsingOpenCvGpu(std::string input_file, std::string output_file);
void processUsingCuda(std::string input_file, std::string output_file);

int main(int argc, char **argv) {

    bool useEpsCheck = false; // set true to enable perPixelError and globalError
    double perPixelError = 3;
    double globalError   = 10;

    const string input_file = argc >= 2 ? argv[1] : "../../data/10_1.jpg";
    const string output_file_OpenCvCpu = argc >= 3 ? argv[2] : "../../data/image_OpenCvCpu.jpg";
    const string output_file_Cuda = argc >= 4 ? argv[3] : "../../data/image_Cuda.jpg";

    for (int i=0; i<1; ++i) {
        processUsingOpenCvCpu(input_file, output_file_OpenCvCpu);
        processUsingCuda(input_file, output_file_Cuda);
    }

    compareImages(output_file_OpenCvCpu, output_file_Cuda, useEpsCheck, perPixelError, globalError);

    return 0;
}

void processUsingOpenCvCpu(std::string input_file, std::string output_file) {
    //Read input image from the disk
    Mat input = imread(input_file, CV_LOAD_IMAGE_COLOR);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    Mat output;

    GpuTimer timer;
    timer.Start();
    resize(input, output, Size(), .25, 0.25, CV_INTER_LINEAR); // downscale 4x on both x and y

    cv::Mat output_float;
    output.convertTo(output_float, CV_32FC3);

    Mat bgr[3];   //destination array
    split(output_float, bgr);//split source

    //merge(bgr, 3, output_float);

    bgr[0] = (bgr[0] - 127.5) / 128;
    bgr[1] = (bgr[1] - 127.5) / 128;
    bgr[2] = (bgr[2] - 127.5) / 128;

    vconcat(bgr, 3, output_float);

    timer.Stop();
    printf("OpenCv Cpu code ran in: %f msecs.\n", timer.Elapsed());

    std::ofstream opencv_image_file("raw_opencv.txt");
    float *outputPtr = output_float.ptr<float>(0);

    for (size_t i = 0; i < output_float.rows * output_float.cols * output_float.channels(); ++i) {
        opencv_image_file << outputPtr[i];
        opencv_image_file << endl;
    }

    imwrite(output_file, output_float);
}
/*
void processUsingOpenCvGpu(std::string input_file, std::string output_file) {
    //Read input image from the disk
    Mat inputCpu = imread(input_file,CV_LOAD_IMAGE_COLOR);
    cuda::GpuMat input (inputCpu);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    //Create output image
    cuda::GpuMat output;

    GpuTimer timer;
    timer.Start();

    cuda::resize(input, output, Size(), .25, 0.25, CV_INTER_AREA); // downscale 4x on both x and y

    timer.Stop();
    printf("OpenCv Gpu code ran in: %f msecs.\n", timer.Elapsed());

    Mat outputCpu;
    output.download(outputCpu);
    imwrite(output_file, outputCpu);

    input.release();
    output.release();
}
*/
void processUsingCuda(std::string input_file, std::string output_file) {
    //Read input image from the disk
    cv::Mat input = cv::imread(input_file, CV_LOAD_IMAGE_UNCHANGED);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    //Create output image
    Size newSize( input.size().width / 4, input.size().height / 4 ); // downscale 4x on both x and y
    Mat output (newSize, CV_32FC3);

    Mat output_bgrplanar(newSize, CV_32FC3);

    downscaleCuda(input, output);

    bgr2bgrplanarCuda(output, output_bgrplanar);

    std::ofstream cuda_image_file("raw_cuda.txt");
    float *outputPtr = output_bgrplanar.ptr<float>(0);

    for (size_t i = 0; i < output_bgrplanar.rows * output_bgrplanar.cols * output_bgrplanar.channels(); ++i) {
        cuda_image_file << outputPtr[i];
        cuda_image_file << endl;
    }

    imwrite(output_file, output);
}