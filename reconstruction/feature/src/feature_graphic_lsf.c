#include "rockit_feature_api.h"
#include "feature_alloc.h"
#include <stdint.h>
#include <stdlib.h>

#define RKF_EINVAL (-22)
#define RKF_VENDOR_FAILURE (-1)

typedef struct Fence {
    uint32_t magic;
    uint64_t seq;
} Fence;

static uint8_t channel_from_rgba(uint32_t rgba, unsigned shift) {
    return (uint8_t)((rgba >> shift) & 255u);
}

void *RKGFX_LSF_Create(void) {
    ROCKIT_GFX_CONTEXT_S *context =
        (ROCKIT_GFX_CONTEXT_S *)rockit_feature_calloc(1, sizeof(*context));
    if (context) {
        context->magic = 0x4c534631u;
    }
    return context;
}

int RKGFX_LSF_Init(void *context, int width, int height, int format) {
    ROCKIT_GFX_CONTEXT_S *state = (ROCKIT_GFX_CONTEXT_S *)context;
    if (!state || width <= 0 || height <= 0) {
        return RKF_EINVAL;
    }
    state->width = (uint32_t)width;
    state->height = (uint32_t)height;
    state->format = (uint32_t)format;
    return 0;
}

int RKGFX_LSF_Composite(void *context, const void *layers) {
    ROCKIT_GFX_CONTEXT_S *state = (ROCKIT_GFX_CONTEXT_S *)context;
    const ROCKIT_GFX_LAYER_LIST_S *list =
        (const ROCKIT_GFX_LAYER_LIST_S *)layers;
    uint8_t *destination;

    if (state && state->magic == 0 && list) {
        return RKF_VENDOR_FAILURE;
    }
    if (!state || state->magic != 0x4c534631u || !list ||
        !list->destination.data ||
        list->destination.format != ROCKIT_FEATURE_FORMAT_RGBA32 ||
        list->destination.width != state->width ||
        list->destination.height != state->height ||
        list->destination.stride < state->width * 4u ||
        list->destination.size <
            (size_t)list->destination.stride * list->destination.height) {
        return RKF_EINVAL;
    }
    if (list->layer_count && !list->layers) {
        return RKF_EINVAL;
    }

    destination = (uint8_t *)list->destination.data;
    for (uint32_t y = 0; y < state->height; ++y) {
        for (uint32_t x = 0; x < state->width; ++x) {
            uint8_t *pixel = destination +
                             (size_t)y * list->destination.stride +
                             (size_t)x * 4u;
            pixel[0] = channel_from_rgba(list->clear_rgba, 24);
            pixel[1] = channel_from_rgba(list->clear_rgba, 16);
            pixel[2] = channel_from_rgba(list->clear_rgba, 8);
            pixel[3] = channel_from_rgba(list->clear_rgba, 0);
        }
    }

    for (uint32_t layer_index = 0; layer_index < list->layer_count; ++layer_index) {
        const ROCKIT_GFX_LAYER_S *layer = &list->layers[layer_index];
        const ROCKIT_FEATURE_IMAGE_S *image = &layer->image;
        if (!image->data || image->format != ROCKIT_FEATURE_FORMAT_RGBA32 ||
            image->stride < image->width * 4u ||
            image->size < (size_t)image->stride * image->height) {
            return RKF_EINVAL;
        }
        for (uint32_t source_y = 0; source_y < image->height; ++source_y) {
            int destination_y = layer->dst_y + (int)source_y;
            if (destination_y < 0 || destination_y >= (int)state->height) {
                continue;
            }
            for (uint32_t source_x = 0; source_x < image->width; ++source_x) {
                int destination_x = layer->dst_x + (int)source_x;
                const uint8_t *source_pixel;
                uint8_t *destination_pixel;
                uint32_t alpha;
                if (destination_x < 0 || destination_x >= (int)state->width) {
                    continue;
                }
                source_pixel = (const uint8_t *)image->data +
                               (size_t)source_y * image->stride +
                               (size_t)source_x * 4u;
                destination_pixel = destination +
                                    (size_t)destination_y * list->destination.stride +
                                    (size_t)destination_x * 4u;
                alpha = ((uint32_t)source_pixel[3] * layer->global_alpha + 127u) / 255u;
                for (unsigned channel = 0; channel < 3; ++channel) {
                    destination_pixel[channel] = (uint8_t)(
                        ((uint32_t)source_pixel[channel] * alpha +
                         (uint32_t)destination_pixel[channel] * (255u - alpha) + 127u) /
                        255u);
                }
                destination_pixel[3] = (uint8_t)(
                    alpha +
                    ((uint32_t)destination_pixel[3] * (255u - alpha) + 127u) / 255u);
            }
        }
    }
    state->sequence++;
    return 0;
}

uintptr_t RKGFX_LSF_CreateFence(void *context) {
    ROCKIT_GFX_CONTEXT_S *state = (ROCKIT_GFX_CONTEXT_S *)context;
    Fence *fence;
    if (!state || state->magic != 0x4c534631u) {
        return 0;
    }
    fence = (Fence *)rockit_feature_calloc(1, sizeof(*fence));
    if (!fence) {
        return 0;
    }
    fence->magic = 0x46454e43u;
    fence->seq = ++state->sequence;
    return (uintptr_t)fence;
}

int RKGFX_LSF_WaitFence(void *context, void *fence) {
    Fence *state = (Fence *)fence;
    (void)context;
    if (!state || state->magic != 0x46454e43u) {
        return state ? RKF_VENDOR_FAILURE : RKF_EINVAL;
    }
    state->magic = 0;
    free(state);
    return 0;
}

int RKGFX_LSF_ReleaseChannel(void *context, int channel) {
    return (!context || channel < 0) ? RKF_EINVAL : 0;
}

int RKGFX_LSF_Destroy(void *context) {
    ROCKIT_GFX_CONTEXT_S *state = (ROCKIT_GFX_CONTEXT_S *)context;
    if (!state) {
        return 0;
    }
    if (state->magic != 0x4c534631u) {
        return RKF_VENDOR_FAILURE;
    }
    state->magic = 0;
    free(state);
    return 0;
}

uint64_t RKGFX_LSF_Calc_Afbc_MemSize(int width, int height, int format) {
    uint64_t aligned_width;
    uint64_t aligned_height;
    (void)format;
    if (width <= 0 || height <= 0) {
        return 0;
    }
    aligned_width = ((uint64_t)width + 15u) & ~15ull;
    aligned_height = ((uint64_t)height + 15u) & ~15ull;
    return aligned_width * aligned_height * 3u / 2u +
           ((aligned_width / 16u) * (aligned_height / 16u) * 16u);
}
