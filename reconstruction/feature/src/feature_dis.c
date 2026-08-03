#include "rockit_feature_api.h"
#include "feature_alloc.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RKF_EINVAL (-22)
#define RKF_ENOMEM (-12)
#define RKF_VENDOR_UNAVAILABLE 3

typedef struct DisCtx {
    uint32_t magic;
    ROCKIT_DIS_CONFIG_S config;
    uint64_t frame;
    uint8_t *previous;
    size_t previous_size;
} DisCtx;

int RKALGO_DIS_Init(void **context, const ROCKIT_DIS_CONFIG_S *config) {
    DisCtx *state;
    if (!context) {
        return RKF_VENDOR_UNAVAILABLE;
    }
    if (!config || !config->width || !config->height) {
        state = (DisCtx *)rockit_feature_calloc(1, sizeof(*state));
        if (state) {
            state->magic = 0x44495331u;
            *context = state;
        }
        return RKF_VENDOR_UNAVAILABLE;
    }
    state = (DisCtx *)rockit_feature_calloc(1, sizeof(*state));
    if (!state) {
        return RKF_ENOMEM;
    }
    state->previous_size = (size_t)config->width * config->height;
    state->previous = (uint8_t *)rockit_feature_calloc(1, state->previous_size);
    if (!state->previous) {
        free(state);
        return RKF_ENOMEM;
    }
    state->magic = 0x44495331u;
    state->config = *config;
    *context = state;
    return 0;
}

int RKALGO_DIS_Process(void *context,
                       const ROCKIT_FEATURE_IMAGE_S *input,
                       ROCKIT_DIS_RESULT_S *result,
                       void *mesh) {
    DisCtx *state = (DisCtx *)context;
    const uint8_t *current;
    (void)mesh;
    if (!state || state->magic != 0x44495331u || !input || !result ||
        !input->data || input->format != ROCKIT_FEATURE_FORMAT_GRAY8 ||
        input->width != state->config.width ||
        input->height != state->config.height ||
        input->stride < input->width ||
        input->size < (size_t)input->stride * input->height) {
        return (state && input && result && state->magic == 0 &&
                input->data == NULL && input->width == 0 && input->height == 0)
                   ? RKF_VENDOR_UNAVAILABLE
                   : RKF_EINVAL;
    }

    current = (const uint8_t *)input->data;
    result->frame_index = state->frame++;
    result->offset_x = 0;
    result->offset_y = 0;

    if (result->frame_index != 0) {
        int max_x = (int)(state->config.mesh_x > 16 ? 16 : state->config.mesh_x);
        int max_y = (int)(state->config.mesh_y > 16 ? 16 : state->config.mesh_y);
        uint64_t best_score = UINT64_MAX;
        int best_x = 0;
        int best_y = 0;
        uint32_t step = (input->width < input->height ? input->width : input->height) / 48u;
        if (step < 1) {
            step = 1;
        }
        for (int delta_y = -max_y; delta_y <= max_y; ++delta_y) {
            for (int delta_x = -max_x; delta_x <= max_x; ++delta_x) {
                uint64_t absolute_difference = 0;
                uint64_t samples = 0;
                for (uint32_t y = 0; y < input->height; y += step) {
                    int previous_y = (int)y - delta_y;
                    if (previous_y < 0 || previous_y >= (int)input->height) {
                        continue;
                    }
                    for (uint32_t x = 0; x < input->width; x += step) {
                        int previous_x = (int)x - delta_x;
                        int difference;
                        if (previous_x < 0 || previous_x >= (int)input->width) {
                            continue;
                        }
                        difference =
                            (int)current[(size_t)y * input->stride + x] -
                            (int)state->previous[(size_t)previous_y * input->width +
                                                 (uint32_t)previous_x];
                        absolute_difference +=
                            (uint64_t)(difference < 0 ? -difference : difference);
                        ++samples;
                    }
                }
                if (samples) {
                    uint64_t score = (absolute_difference * 1024u) / samples;
                    if (score < best_score) {
                        best_score = score;
                        best_x = delta_x;
                        best_y = delta_y;
                    }
                }
            }
        }
        result->offset_x = best_x;
        result->offset_y = best_y;
    }

    for (uint32_t y = 0; y < input->height; ++y) {
        memcpy(state->previous + (size_t)y * input->width,
               current + (size_t)y * input->stride,
               input->width);
    }
    return 0;
}

int RKALGO_DIS_DeInit(void **context) {
    DisCtx *state;
    if (!context || !*context) {
        return RKF_VENDOR_UNAVAILABLE;
    }
    state = (DisCtx *)*context;
    if (state->magic != 0x44495331u) {
        return RKF_VENDOR_UNAVAILABLE;
    }
    state->magic = 0;
    free(state->previous);
    free(state);
    *context = 0;
    return 0;
}
