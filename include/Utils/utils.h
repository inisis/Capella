/**
 * Created by desmond <desmond.yao@buaa.edu.cn> on 2018-11-18
 */

#ifndef _UTILS_H__
#define _UTILS_H__

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timer.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

template<typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
	//check that the GPU result matches the CPU result
	bool is_same = true;
	for (size_t i = 0; i < numElem; ++i) {
		if (ref[i] != gpu[i]) {
			std::cerr << "Difference at pos " << i << std::endl;
			std::cerr << "Reference: " << std::setprecision(17) << +ref[i] << "\nGPU      : " << +gpu[i] << std::endl;
			is_same = false;
		}
	}
	if(is_same)
		std::cerr << "Generated images are the same." << std::endl;
	else
		std::cerr << "Generated images are not the same." << std::endl;
}

template<typename T>
void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
	assert(eps1 >= 0 && eps2 >= 0);
	unsigned long long totalDiff = 0;
	unsigned numSmallDifferences = 0;
	for (size_t i = 0; i < numElem; ++i) {
		//subtract smaller from larger in case of unsigned types
		T smaller = std::min(ref[i], gpu[i]);
		T larger = std::max(ref[i], gpu[i]);
		T diff = larger - smaller;
		if (diff > 0 && diff <= eps1) {
			numSmallDifferences++;
		}
		else if (diff > eps1) {
			std::cerr << "Difference at pos " << +i << " exceeds tolerance of " << eps1 << std::endl;
			std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
					"\nGPU      : " << +gpu[i] << std::endl;
			exit(1);
		}
		totalDiff += diff * diff;
	}
	double percentSmallDifferences = (double)numSmallDifferences / (double)numElem;
	if (percentSmallDifferences > eps2) {
		std::cerr << "Total percentage of non-zero pixel difference between the two images exceeds " << 100.0 * eps2 << "%" << std::endl;
		std::cerr << "Percentage of non-zero pixel differences: " << 100.0 * percentSmallDifferences << "%" << std::endl;
		exit(1);
	}
}

//Uses the autodesk method of image comparison
//Note the the tolerance here is in PIXELS not a percentage of input pixels
template<typename T>
void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
{

	size_t numBadPixels = 0;
	for (size_t i = 0; i < numElem; ++i) {
		T smaller = std::min(ref[i], gpu[i]);
		T larger = std::max(ref[i], gpu[i]);
		T diff = larger - smaller;
		if (diff > variance)
			++numBadPixels;
	}

	if (numBadPixels > tolerance) {
		std::cerr << "Too many bad pixels in the image." << numBadPixels << "/" << tolerance << std::endl;
		exit(1);
	}
}

void compareImages(std::string reference_filename, std::string test_filename,
		bool useEpsCheck, double perPixelError, double globalError)
{
	cv::Mat reference = cv::imread(reference_filename, -1);
	cv::Mat test = cv::imread(test_filename, -1);

	cv::Mat diff = abs(reference - test);

	cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

	double minVal, maxVal;

	cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

	//now perform transform so that we bump values to the full range

	diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

	diff = diffSingleChannel.reshape(reference.channels(), 0);

	cv::imwrite("../data/output_difference.png", diff);
	//OK, now we can start comparing values...
	unsigned char *referencePtr = reference.ptr<unsigned char>(0);
	unsigned char *testPtr = test.ptr<unsigned char>(0);

	if (useEpsCheck) {
		checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), perPixelError, globalError);
	}
	else
	{
		checkResultsExact(referencePtr, testPtr, 2 * reference.cols * reference.channels());
	}
	return;
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
	cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

	cv::imwrite(output_filename, reference);

}

#endif
