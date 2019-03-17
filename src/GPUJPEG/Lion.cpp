//
// Created by desmond on 18-9-21.
//
#include "GPUJPEG/Lion.h"

using namespace std;

Lion::Lion()
{


}

Lion::~Lion()
{
    release_encoder();
}

int Lion::init_encoder(int gpu_id, int width, int height, int verbose)
{

    if (gpujpeg_init_device(gpu_id, verbose) != 0)
        return -1;

    gpujpeg_set_default_parameters(&param_);
    param_.verbose = verbose;
    param_.interleaved = true;

    gpujpeg_parameters_chroma_subsampling_420(&param_);

    gpujpeg_image_set_default_parameters(&param_image_);
    param_image_.width = width;
    param_image_.height = height;
    param_image_.comp_count = 3;
    param_image_.color_space = GPUJPEG_RGB;
    param_image_.pixel_format = GPUJPEG_444_U8_P012;

    gpujpeg_image_set_default_parameters(&param_image_original_);

    param_.restart_interval = 2;

    // Create encoder
    encoder_ = gpujpeg_encoder_create(NULL);
    if ( encoder_ == NULL ) {
        fprintf(stderr, "Failed to create encoder!\n");
        return -1;
    }

    for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
        if (gpujpeg_table_quantization_encoder_init(&encoder_->table_quantization[comp_type], (enum gpujpeg_component_type)comp_type, param_.quality) != 0) {
            return -1;
        }
    }
    gpujpeg_cuda_check_error("Quantization init", return -1);

    if (0 == gpujpeg_coder_init_image(&encoder_->coder, &param_, &param_image_, encoder_->stream)) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init image encoding!\n");
        return -1;
    }
}

int Lion::encode_bgr(uint8_t * bgr_ptr, cv::Size size, cv::Rect rect, void* jpg_ptr)
{

    GPUJPEG_TIMER_INIT();

    GPUJPEG_TIMER_START();

    param_image_.width = size.width;
    param_image_.height = size.height;
    encoder_->coder.data_width = size.width;
    encoder_->coder.data_height = size.height;
    encoder_->coder.param_image = param_image_;
    encoder_->coder.component->height = size.height;
    encoder_->coder.component->width = size.width;
    encoder_->coder.data_height = size.height;
    encoder_->coder.data_width = size.width;

    gpujpeg_encoder_input_set_gpu_image(&encoder_input_, bgr_ptr);

    encoder_->coder.d_data_raw = bgr_ptr;
    encoder_->coder.data_raw_size = size.height * size.width * 3;
    encoder_->coder.data_size =  size.height * size.width * 3;

    size_t allocated_gpu_memory_size = 0;

    struct gpujpeg_coder * coder = &encoder_->coder;

    // Set parameters
    coder->param_image = param_image_;
    coder->param = param_;

    allocated_gpu_memory_size += coder->component_allocated_size * sizeof(struct gpujpeg_component);

    // Calculate raw data size
    coder->data_raw_size = gpujpeg_image_calculate_size(&coder->param_image);

    // Initialize color components and compute maximum sampling factor to coder->sampling_factor
    coder->data_size = 0;
    coder->sampling_factor.horizontal = 0;
    coder->sampling_factor.vertical = 0;
    for (int comp = 0; comp < coder->param_image.comp_count; comp++) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Sampling factors
        assert(coder->param.sampling_factor[comp].horizontal >= 1 && coder->param.sampling_factor[comp].horizontal <= 15);
        assert(coder->param.sampling_factor[comp].vertical >= 1 && coder->param.sampling_factor[comp].vertical <= 15);
        component->sampling_factor = coder->param.sampling_factor[comp];
        if ( component->sampling_factor.horizontal > coder->sampling_factor.horizontal ) {
            coder->sampling_factor.horizontal = component->sampling_factor.horizontal;
        }
        if ( component->sampling_factor.vertical > coder->sampling_factor.vertical ) {
            coder->sampling_factor.vertical = component->sampling_factor.vertical;
        }

        // Set type
        component->type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;

        // Set proper color component sizes in pixels based on sampling factors
        int width = ((coder->param_image.width + coder->sampling_factor.horizontal - 1) / coder->sampling_factor.horizontal) * coder->sampling_factor.horizontal;
        int height = ((coder->param_image.height + coder->sampling_factor.vertical - 1) / coder->sampling_factor.vertical) * coder->sampling_factor.vertical;
        int samp_factor_h = component->sampling_factor.horizontal;
        int samp_factor_v = component->sampling_factor.vertical;
        component->width = (width * samp_factor_h) / coder->sampling_factor.horizontal;
        component->height = (height * samp_factor_v) / coder->sampling_factor.vertical;

        // Compute component MCU size
        component->mcu_size_x = GPUJPEG_BLOCK_SIZE;
        component->mcu_size_y = GPUJPEG_BLOCK_SIZE;
        if ( coder->param.interleaved == 1 ) {
            component->mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE * samp_factor_h * samp_factor_v;
            component->mcu_size_x *= samp_factor_h;
            component->mcu_size_y *= samp_factor_v;
        } else {
            component->mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE;
        }
        component->mcu_size = component->mcu_size_x * component->mcu_size_y;

        // Compute allocated data size
        component->data_width = gpujpeg_div_and_round_up(component->width, component->mcu_size_x) * component->mcu_size_x;
        component->data_height = gpujpeg_div_and_round_up(component->height, component->mcu_size_y) * component->mcu_size_y;
        component->data_size = component->data_width * component->data_height;
        // Increase total data size
        coder->data_size += component->data_size;

        // Compute component MCU count
        component->mcu_count_x = gpujpeg_div_and_round_up(component->data_width, component->mcu_size_x);
        component->mcu_count_y = gpujpeg_div_and_round_up(component->data_height, component->mcu_size_y);
        component->mcu_count = component->mcu_count_x * component->mcu_count_y;

        // Compute MCU count per segment
        component->segment_mcu_count = coder->param.restart_interval;
        if ( component->segment_mcu_count == 0 ) {
            // If restart interval is disabled, restart interval is equal MCU count
            component->segment_mcu_count = component->mcu_count;
        }

        // Calculate segment count
        component->segment_count = gpujpeg_div_and_round_up(component->mcu_count, component->segment_mcu_count);

        //printf("Subsampling %dx%d, Resolution %d, %d, mcu size %d, mcu count %d\n",
        //    coder->param.sampling_factor[comp].horizontal, coder->param.sampling_factor[comp].vertical,
        //    component->data_width, component->data_height,
        //    component->mcu_compressed_size, component->mcu_count
        //);
    }

    // Maximum component data size for allocated buffers
    coder->data_width = gpujpeg_div_and_round_up(coder->param_image.width, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    coder->data_height = gpujpeg_div_and_round_up(coder->param_image.height, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;

    // Compute MCU size, MCU count, segment count and compressed data allocation size
    coder->mcu_count = 0;
    coder->mcu_size = 0;
    coder->mcu_compressed_size = 0;
    coder->segment_count = 0;
    coder->data_compressed_size = 0;
    if ( coder->param.interleaved == 1 ) {
        assert(coder->param_image.comp_count > 0);
        coder->mcu_count = coder->component[0].mcu_count;
        coder->segment_count = coder->component[0].segment_count;
        coder->segment_mcu_count = coder->component[0].segment_mcu_count;
        for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
            assert(coder->mcu_count == component->mcu_count);
            assert(coder->segment_mcu_count == component->segment_mcu_count);
            coder->mcu_size += component->mcu_size;
            coder->mcu_compressed_size += component->mcu_compressed_size;
        }
    } else {
        assert(coder->param_image.comp_count > 0);
        coder->mcu_size = coder->component[0].mcu_size;
        coder->mcu_compressed_size = coder->component[0].mcu_compressed_size;
        coder->segment_mcu_count = 0;
        for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
            assert(coder->mcu_size == component->mcu_size);
            assert(coder->mcu_compressed_size == component->mcu_compressed_size);
            coder->mcu_count += component->mcu_count;
            coder->segment_count += component->segment_count;
        }
    }
    //printf("mcu size %d -> %d, mcu count %d, segment mcu count %d\n", coder->mcu_size, coder->mcu_compressed_size, coder->mcu_count, coder->segment_mcu_count);

    allocated_gpu_memory_size += coder->segment_allocated_size * sizeof(struct gpujpeg_segment);

    // Prepare segments
    // While preparing segments compute input size and compressed size
    int data_index = 0;
    int data_compressed_index = 0;
    // Prepare segments based on (non-)interleaved mode
    if ( coder->param.interleaved == 1 ) {
        // Prepare segments for encoding (only one scan for all color components)
        int mcu_index = 0;
        for ( int index = 0; index < coder->segment_count; index++ ) {
            // Prepare segment MCU count
            int mcu_count = coder->segment_mcu_count;
            if ( (mcu_index + mcu_count) >= coder->mcu_count )
                mcu_count = coder->mcu_count - mcu_index;
            // Set parameters for segment
            coder->segment[index].scan_index = 0;
            coder->segment[index].scan_segment_index = index;
            coder->segment[index].mcu_count = mcu_count;
            coder->segment[index].data_compressed_index = data_compressed_index;
            coder->segment[index].data_temp_index = data_compressed_index;
            coder->segment[index].data_compressed_size = 0;
            // Increase parameters for next segment
            data_index += mcu_count * coder->mcu_size;
            data_compressed_index += SEGMENT_ALIGN(mcu_count * coder->mcu_compressed_size);
            mcu_index += mcu_count;
        }
    }
    else {
        // Prepare segments for encoding (one scan for each color component)
        int index = 0;
        for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
            // Get component
            struct gpujpeg_component* component = &coder->component[comp];
            // Prepare component segments
            int mcu_index = 0;
            for ( int segment = 0; segment < component->segment_count; segment++ ) {
                // Prepare segment MCU count
                int mcu_count = component->segment_mcu_count;
                if ( (mcu_index + mcu_count) >= component->mcu_count )
                    mcu_count = component->mcu_count - mcu_index;
                // Set parameters for segment
                coder->segment[index].scan_index = comp;
                coder->segment[index].scan_segment_index = segment;
                coder->segment[index].mcu_count = mcu_count;
                coder->segment[index].data_compressed_index = data_compressed_index;
                coder->segment[index].data_temp_index = data_compressed_index;
                coder->segment[index].data_compressed_size = 0;
                // Increase parameters for next segment
                data_index += mcu_count * component->mcu_size;
                data_compressed_index += SEGMENT_ALIGN(mcu_count * component->mcu_compressed_size);
                mcu_index += mcu_count;
                index++;
            }
        }
    }
    // Check data size
    //printf("%d == %d\n", coder->data_size, data_index);
    assert(coder->data_size == data_index);
    // Set compressed size
    coder->data_compressed_size = data_compressed_index;
    //printf("Compressed size %d (segments %d)\n", coder->data_compressed_size, coder->segment_count);

//for idct we must add some memory - it rounds up the block count, computes all and the extra bytes are omitted
    if (coder->component[0].data_width <= 0) {
        fprintf(stderr, "Data width should be positive!\n");
        return 0;
    }
    int idct_overhead = (GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_Z / coder->component[0].data_width + 1)
                        * GPUJPEG_BLOCK_SIZE * coder->component[0].data_width;

    allocated_gpu_memory_size += coder->data_allocated_size * sizeof(uint8_t);
    allocated_gpu_memory_size += coder->data_allocated_size * sizeof(int16_t);

    // Set data buffer to color components
    uint8_t* d_comp_data = coder->d_data;
    int16_t* d_comp_data_quantized = coder->d_data_quantized;
    int16_t* comp_data_quantized = coder->data_quantized;
    unsigned int data_quantized_index = 0;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        struct gpujpeg_component* component = &coder->component[comp];
        component->d_data = d_comp_data;
        component->d_data_quantized = d_comp_data_quantized;
        component->data_quantized_index = data_quantized_index;
        component->data_quantized = comp_data_quantized;
        d_comp_data += component->data_width * component->data_height;
        d_comp_data_quantized += component->data_width * component->data_height;
        comp_data_quantized += component->data_width * component->data_height;
        data_quantized_index += component->data_width * component->data_height;
    }

    // Allocate compressed data
    int max_compressed_data_size = coder->data_compressed_size;
    max_compressed_data_size += GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    //max_compressed_data_size *= 2;

    allocated_gpu_memory_size += coder->data_compressed_allocated_size * sizeof(uint8_t);
    allocated_gpu_memory_size += coder->data_compressed_allocated_size * sizeof(uint8_t);

    // Allocate block lists in host memory
    coder->block_count = 0;
    for (int comp = 0; comp < coder->param_image.comp_count; comp++) {
        coder->block_count += (coder->component[comp].data_width * coder->component[comp].data_height) / (8 * 8);
    }

    allocated_gpu_memory_size += coder->block_allocated_size * sizeof(*coder->d_block_list);

    // Initialize block lists in host memory
    int block_idx = 0;
    int comp_count = 1;
    if ( coder->param.interleaved == 1 ) {
        comp_count = coder->param_image.comp_count;
    }
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    for (int segment_idx = 0; segment_idx < coder->segment_count; segment_idx++) {
        struct gpujpeg_segment* const segment = &coder->segment[segment_idx];
        segment->block_index_list_begin = block_idx;

        // Non-interleaving mode
        if ( comp_count == 1 ) {
            // Inspect MCUs in segment
            for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
                // Component for the scan
                struct gpujpeg_component* component = &coder->component[segment->scan_index];

                // Offset of component data for MCU
                uint64_t data_index = component->data_quantized_index + (segment->scan_segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size;
                uint64_t component_type = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00 : 0x80;
                uint64_t dc_index = segment->scan_index;
                coder->block_list[block_idx++] = dc_index | component_type | (data_index << 8);
            }
        }
            // Interleaving mode
        else {
            // Encode MCUs in segment
            for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
                //assert(segment->scan_index == 0);
                for ( int comp = 0; comp < comp_count; comp++ ) {
                    struct gpujpeg_component* component = &coder->component[comp];

                    // Prepare mcu indexes
                    int mcu_index_x = (segment->scan_segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                    int mcu_index_y = (segment->scan_segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                    // Compute base data index
                    int data_index_base = component->data_quantized_index + mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);

                    // For all vertical 8x8 blocks
                    for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                        // Compute base row data index
                        int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                        // For all horizontal 8x8 blocks
                        for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                            // Compute 8x8 block data index
                            uint64_t data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                            uint64_t component_type = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00 : 0x80;
                            uint64_t dc_index = comp;
                            coder->block_list[block_idx++] = dc_index | component_type | (data_index << 8);
                        }
                    }
                }
            }
        }
        segment->block_count = block_idx - segment->block_index_list_begin;
    }
    assert(block_idx == coder->block_count);

    // Copy components to device memory
    if (encoder_->stream != NULL) {
        cudaMemcpyAsync(coder->d_component, coder->component, coder->param_image.comp_count * sizeof(struct gpujpeg_component), cudaMemcpyHostToDevice, *(encoder_->stream));
    }
    else {
        cudaMemcpy(coder->d_component, coder->component, coder->param_image.comp_count * sizeof(struct gpujpeg_component), cudaMemcpyHostToDevice);
    }
    gpujpeg_cuda_check_error("Coder component copy", return 0);

    // Copy block lists to device memory
    if (encoder_->stream != NULL) {
        cudaMemcpyAsync(coder->d_block_list, coder->block_list, coder->block_count * sizeof(*coder->d_block_list), cudaMemcpyHostToDevice, *(encoder_->stream));
    }
    else {
        cudaMemcpy(coder->d_block_list, coder->block_list, coder->block_count * sizeof(*coder->d_block_list), cudaMemcpyHostToDevice);
    }
    gpujpeg_cuda_check_error("Coder block list copy", return 0);

    // Copy segments to device memory
    if (encoder_->stream) {
        cudaMemcpyAsync(coder->d_segment, coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice, *(encoder_->stream));
    }
    else {
        cudaMemcpy(coder->d_segment, coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice);
    }
    gpujpeg_cuda_check_error("Coder segment copy", return 0);


    if (gpujpeg_encoder_encode(encoder_, &param_, &param_image_, &encoder_input_, &image_compressed_, &image_compressed_size_) != 0 ) {
        fprintf(stderr, "Failed to encode image!\n");
        return -1;
    }

    GPUJPEG_TIMER_STOP();
    float duration = GPUJPEG_TIMER_DURATION();
    if ( param_.verbose ) {
        printf(" -Copy From Device:  %10.2f ms\n", encoder_->coder.duration_memory_from);
    }
    printf("Encode Image GPU:    %10.2f ms (only in-GPU processing)\n", encoder_->coder.duration_in_gpu);
    printf("Encode Image:        %10.2f ms\n", duration);

    printf("image_compressed_size_:    %d bytes \n", image_compressed_size_);

    // Save image
    if ( gpujpeg_image_save_to_file("output.jpg", image_compressed_, image_compressed_size_) != 0 ) {
        fprintf(stderr, "Failed to save image !\n");
        return -1;
    }

}


int Lion::release_encoder()
{
    gpujpeg_encoder_destroy(encoder_);

}