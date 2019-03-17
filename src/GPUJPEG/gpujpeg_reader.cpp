/**
 * Copyright (c) 2011, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>
#include "GPUJPEG/gpujpeg_reader.h"
#include "GPUJPEG/gpujpeg_util.h"

/** Documented at declaration */
struct gpujpeg_reader*
gpujpeg_reader_create()
{
    struct gpujpeg_reader* reader = (struct gpujpeg_reader*)
            malloc(sizeof(struct gpujpeg_reader));
    if ( reader == NULL )
        return NULL;
    reader->comp_count = 0;
    reader->scan_count = 0;
    reader->segment_count = 0;
    reader->segment_info_count = 0;
    reader->segment_info_size = 0;

    return reader;
}

/** Documented at declaration */
int
gpujpeg_reader_destroy(struct gpujpeg_reader* reader)
{
    assert(reader != NULL);
    free(reader);
    return 0;
}

/**
 * Read byte from image data
 *
 * @param image
 * @return byte
 */
#define gpujpeg_reader_read_byte(image) \
    (uint8_t)(*(image)++)

/**
 * Read two-bytes from image data
 *
 * @param image
 * @return 2 bytes
 */
#define gpujpeg_reader_read_2byte(image) \
    (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
    image += 2;

/**
 * Read marker from image data
 *
 * @param image
 * @return marker code or -1 if failed
 */
int
gpujpeg_reader_read_marker(uint8_t** image)
{
    uint8_t byte = gpujpeg_reader_read_byte(*image);
    if( byte != 0xFF ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to read marker from JPEG data (0xFF was expected but 0x%X was presented)\n", byte);
        return -1;
    }
    int marker = gpujpeg_reader_read_byte(*image);
    return marker;
}

/**
 * Skip marker content (read length and that much bytes - 2)
 *
 * @param image
 * @return void
 */
void
gpujpeg_reader_skip_marker_content(uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);

    *image += length - 2;
}

/**
 * Read application ifno block from image
 *
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_app0(uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length != 16 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker length should be 16 but %d was presented!\n", length);
        return -1;
    }

    char jfif[5];
    jfif[0] = gpujpeg_reader_read_byte(*image);
    jfif[1] = gpujpeg_reader_read_byte(*image);
    jfif[2] = gpujpeg_reader_read_byte(*image);
    jfif[3] = gpujpeg_reader_read_byte(*image);
    jfif[4] = gpujpeg_reader_read_byte(*image);
    if ( strcmp(jfif, "JFIF") != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker identifier should be 'JFIF' but '%s' was presented!\n", jfif);
        return -1;
    }

    int version_major = gpujpeg_reader_read_byte(*image);
    int version_minor = gpujpeg_reader_read_byte(*image);
    if ( version_major != 1 || version_minor != 1 ) {
        fprintf(stderr, "[GPUJPEG] [Error] APP0 marker version should be 1.1 but %d.%d was presented!\n", version_major, version_minor);
        return -1;
    }

    int pixel_units = gpujpeg_reader_read_byte(*image);
    int pixel_xdpu = gpujpeg_reader_read_2byte(*image);
    int pixel_ydpu = gpujpeg_reader_read_2byte(*image);
    int thumbnail_width = gpujpeg_reader_read_byte(*image);
    int thumbnail_height = gpujpeg_reader_read_byte(*image);

    return 0;
}

/**
 * Read start of frame block from image
 *
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_reader_read_sof0(struct gpujpeg_parameters * param, struct gpujpeg_image_parameters * param_image, uint8_t** image)
{
    int length = (int)gpujpeg_reader_read_2byte(*image);
    if ( length < 6 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker length should be greater than 6 but %d was presented!\n", length);
        return -1;
    }
    length -= 2;

    int precision = (int)gpujpeg_reader_read_byte(*image);
    if ( precision != 8 ) {
        fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker precision should be 8 but %d was presented!\n", precision);
        return -1;
    }

    param_image->height = (int)gpujpeg_reader_read_2byte(*image);
    param_image->width = (int)gpujpeg_reader_read_2byte(*image);
    param_image->comp_count = (int)gpujpeg_reader_read_byte(*image);
    length -= 6;

    for ( int comp = 0; comp < param_image->comp_count; comp++ ) {
        int index = (int)gpujpeg_reader_read_byte(*image);
        if ( index != (comp + 1) ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component %d id should be %d but %d was presented!\n", comp, comp + 1, index);
            return -1;
        }

        int sampling = (int)gpujpeg_reader_read_byte(*image);
        param->sampling_factor[comp].horizontal = (sampling >> 4) & 15;
        param->sampling_factor[comp].vertical = (sampling) & 15;

        int table_index = (int)gpujpeg_reader_read_byte(*image);
        if ( comp == 0 && table_index != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component Y should have quantization table index 0 but %d was presented!\n", table_index);
            return -1;
        }
        if ( (comp == 1 || comp == 2) && table_index != 1 ) {
            fprintf(stderr, "[GPUJPEG] [Error] SOF0 marker component Cb or Cr should have quantization table index 1 but %d was presented!\n", table_index);
            return -1;
        }
        length -= 3;
    }

    // Check length
    if ( length > 0 ) {
        fprintf(stderr, "[GPUJPEG] [Warning] SOF0 marker contains %d more bytes than needed!\n", length);
        *image += length;
    }

    return 0;
}

/** Documented at declaration */
int
gpujpeg_decoder_get_image_info(uint8_t* image, int image_size, struct gpujpeg_image_parameters * param_image)
{
    // Check first SOI marker
    int marker_soi = gpujpeg_reader_read_marker(&image);
    if (marker_soi != GPUJPEG_MARKER_SOI) {
        fprintf(stderr, "[GPUJPEG] [Error] JPEG data should begin with SOI marker, but marker %s was found!\n", gpujpeg_marker_name((enum gpujpeg_marker_code)marker_soi));
        return -1;
    }

    int eoi_presented = 0;
    while (eoi_presented == 0) {
        // Read marker
        int marker = gpujpeg_reader_read_marker(&image);
        if (marker == -1) {
            return -1;
        }

        // Read more info according to the marker
        switch (marker)
        {
        case GPUJPEG_MARKER_SOF0: // Baseline
        case GPUJPEG_MARKER_SOF1: // Extended sequential with Huffman coder
        {
            struct gpujpeg_parameters param;
            if (gpujpeg_reader_read_sof0(&param, param_image, &image) != 0) {
                return -1;
            }
            return 0;
        }
        case GPUJPEG_MARKER_EOI:
        {
            eoi_presented = 1;
            break;
        }
        default:
            gpujpeg_reader_skip_marker_content(&image);
            break;
        }
    }
    return 0;
}
