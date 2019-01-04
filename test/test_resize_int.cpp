/**
 * Created by desmond <desmond.yao@buaa.edu.cn> on 2018-11-18
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "fstream"
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

const Size newSize(64, 128); // downscale 4x on both x and y

void processUsingOpenCvCpu(std::string nput_file, std::string output_file);
void processUsingCuda(std::string input_file, std::string output_file);

extern "C" void bgr2bgrplanarCuda(const cv::Mat& input, cv::Mat& output);

extern "C" void downscaleCuda(const cv::Mat& input, cv::Mat& output);

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

    Mat input = imread(input_file, CV_LOAD_IMAGE_COLOR);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    char data[HEIGHT * WIDTH * CHANNEL];

    for (int i = 0; i < HEIGHT; ++i)
    {
        for(int j = 0; j < WIDTH; ++j)
        {
            data[i * WIDTH * CHANNEL + j*CHANNEL + 0] = i * WIDTH * CHANNEL + j*CHANNEL + 0;
            data[i * WIDTH * CHANNEL + j*CHANNEL + 1] = i * WIDTH * CHANNEL + j*CHANNEL + 1;
            data[i * WIDTH * CHANNEL + j*CHANNEL + 2] = i * WIDTH * CHANNEL + j*CHANNEL + 2;
        }
    }

    cv::Mat point_mat(HEIGHT, WIDTH, CV_8UC3, data);

    cv::Mat output_float;
    input.convertTo(output_float, CV_32FC3);

    GpuTimer timer;
    timer.Start();

    Mat output;

    resize(input, output, newSize, CV_INTER_LINEAR); // downscale 4x on both x and y

    timer.Stop();
    printf("OpenCv Cpu code ran in: %f msecs.\n", timer.Elapsed());

    std::ofstream opencv_image_file("raw_opencv.txt");
    uchar *outputPtr = output.ptr<uchar>(0);

    for (size_t i = 0; i < output.rows * output.cols * output.channels(); ++i) {
        opencv_image_file << (int)outputPtr[i];
        opencv_image_file << endl;
    }

    imwrite(output_file, output);
}

void processUsingCuda(std::string input_file, std::string output_file) {

    cv::Mat input = cv::imread(input_file, CV_LOAD_IMAGE_UNCHANGED);
    if(input.empty())
    {
        std::cout<<"Image Not Found: "<< input_file << std::endl;
        return;
    }

    char data[HEIGHT * WIDTH * CHANNEL];

    for (int i = 0; i < HEIGHT; ++i)
    {
        for(int j = 0; j < WIDTH; ++j)
        {
            data[i * WIDTH * CHANNEL + j*CHANNEL + 0] = i * WIDTH * CHANNEL + j*CHANNEL + 0;
            data[i * WIDTH * CHANNEL + j*CHANNEL + 1] = i * WIDTH * CHANNEL + j*CHANNEL + 1;
            data[i * WIDTH * CHANNEL + j*CHANNEL + 2] = i * WIDTH * CHANNEL + j*CHANNEL + 2;
        }
    }

    cv::Mat point_mat(HEIGHT, WIDTH, CV_8UC3, data);

    Mat output (newSize, CV_32FC3);

    downscaleCuda(input, output);

    std::ofstream cuda_image_file("raw_cuda.txt");
    float *outputPtr = output.ptr<float>(0);

    for (size_t i = 0; i < output.rows * output.cols * output.channels(); ++i) {
        cuda_image_file << outputPtr[i];
        cuda_image_file << endl;
    }

    imwrite(output_file, output);
}