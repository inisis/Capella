//
// Created by desmond on 18-9-22.
//

#include "iostream"

#include "GPUJPEG/Lion.h"

using namespace std;

int main(int argc, char *argv[])
{

    if (argc < 2) {
        std::cout << "Usage  : test_jpg_encoder input_file" << std::endl;
        std::cout << "Example: ./test_jpe_encoder ../data/avatar_1440x1080.rgb" << std::endl;
        return -1;
    }

    Lion *lion = new Lion();

    cv::Size size = {1440, 1080};  //输入图像尺寸

    lion->init_encoder(0, 1440, 1080, 1); //初始化尺寸

    FILE* file;
    long image_size;
    file = fopen(argv[1], "rb");

    if (!file)
    {
        fprintf(stderr, "[GPUJPEG] [Error] Failed open %s for reading!\n", argv[1]);
        return -1;
    }

    fseek(file, 0, SEEK_END);
    image_size = ftell(file);
    rewind(file);

    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, image_size * sizeof(uint8_t));

    if (image_size != fread(data, sizeof(uint8_t), image_size, file))
    {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image data [%d bytes] from file %s!\n", image_size, argv[1]);
        return -1;
    }

    fclose(file);

    void *bgr_data;
    cudaMalloc((void **) &bgr_data, size.width * size.height * 3);
    cudaMemcpy(bgr_data, data, size.width * size.height * 3, cudaMemcpyHostToDevice);

    cv::Rect rect;

    lion->encode_bgr((uint8_t *)bgr_data, size, rect, nullptr);

    lion->release_encoder();

    return 0;
}