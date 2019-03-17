//
// Created by desmond on 18-9-21.
//

#ifndef GPUJPEG_LION_H
#define GPUJPEG_LION_H

#include "string"
#include "iostream"
#include "opencv2/opencv.hpp"
#include "cuda_runtime_api.h"

#include <GPUJPEG/gpujpeg.h>
#include <GPUJPEG/gpujpeg_util.h>
#include <GPUJPEG/gpujpeg_common_internal.h> // TIMER
#include <GPUJPEG/gpujpeg_encoder_internal.h> // TIMER

#define SEGMENT_ALIGN(b) (((b) + 127) & ~127)

class Lion
{
    public:

        Lion();

        ~Lion();

    public:

        int init_encoder(int gpu_id, int width, int height, int verbose);

        int encode_bgr(uint8_t* bgr_ptr, cv::Size size, cv::Rect rect, void* jpg_ptr);

        int release_encoder();


    private:

        // Default coder parameters
        struct gpujpeg_parameters param_;

        // Default image parameters
        struct gpujpeg_image_parameters param_image_;

        // Original image parameters in conversion
        struct gpujpeg_image_parameters param_image_original_;

        //jpeg encoder
        struct gpujpeg_encoder* encoder_;

        // encoder input
        struct gpujpeg_encoder_input encoder_input_;

    private:

        uint8_t* image_compressed_ = NULL;
        int image_compressed_size_ = 0;

    };

#endif //GPUJPEG_LION_H
