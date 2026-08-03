#include "rockit_feature_api.h"
#include "feature_alloc.h"
#include <stdint.h>
#include <stdlib.h>

#define RKF_EINVAL (-22)
#define RKF_ENOMEM (-12)
#define RKF_ESTATE (-5)
#define RKF_VENDOR_FAILED 2

typedef struct AvsState {
    uint32_t magic;
    uint64_t frames;
    ROCKIT_AVS_CONFIG_S config;
} AvsState;

static AvsState *g_avs;

static uint32_t channel_count(uint32_t format) {
    if (format == ROCKIT_FEATURE_FORMAT_GRAY8) {
        return 1;
    }
    if (format == ROCKIT_FEATURE_FORMAT_RGB24) {
        return 3;
    }
    return format == ROCKIT_FEATURE_FORMAT_RGBA32 ? 4u : 0u;
}

static int valid_config(const ROCKIT_AVS_CONFIG_S *config) {
    uint32_t channels;
    if (!config || !config->inputs || !config->input_count ||
        config->input_count > 16 || !config->output.data) {
        return 0;
    }
    channels = channel_count(config->output.format);
    if (!channels || config->output.stride < config->output.width * channels ||
        config->output.size <
            (size_t)config->output.stride * config->output.height) {
        return 0;
    }
    for (uint32_t index = 0; index < config->input_count; ++index) {
        const ROCKIT_FEATURE_IMAGE_S *image = &config->inputs[index];
        if (!image->data || image->format != config->output.format ||
            image->width != config->output.width ||
            image->height != config->output.height ||
            image->stride < image->width * channels ||
            image->size < (size_t)image->stride * image->height) {
            return 0;
        }
    }
    return 1;
}

int RKGFX_AVS_init(void *config) {
    if (g_avs) {
        return RKF_ESTATE;
    }
    if (!valid_config((const ROCKIT_AVS_CONFIG_S *)config)) {
        return RKF_VENDOR_FAILED;
    }
    g_avs = (AvsState *)rockit_feature_calloc(1, sizeof(*g_avs));
    if (!g_avs) {
        return RKF_ENOMEM;
    }
    g_avs->magic = 0x41565331u;
    g_avs->config = *(const ROCKIT_AVS_CONFIG_S *)config;
    return 0;
}

int RKGFX_AVS_resize(void *config) {
    if (!g_avs) {
        return config ? 0 : RKF_VENDOR_FAILED;
    }
    if (!valid_config((const ROCKIT_AVS_CONFIG_S *)config)) {
        return RKF_EINVAL;
    }
    g_avs->config = *(const ROCKIT_AVS_CONFIG_S *)config;
    return 0;
}

int RKGFX_AVS_AllProcess(void *config) {
    const ROCKIT_AVS_CONFIG_S *active;
    uint32_t channels;
    float total_weight = 0.0f;
    if (!g_avs) {
        return RKF_VENDOR_FAILED;
    }
    active = config ? (const ROCKIT_AVS_CONFIG_S *)config : &g_avs->config;
    if (!valid_config(active)) {
        return RKF_EINVAL;
    }
    channels = channel_count(active->output.format);
    for (uint32_t index = 0; index < active->input_count; ++index) {
        float weight = active->weights ? active->weights[index] : 1.0f;
        if (weight > 0.0f) {
            total_weight += weight;
        }
    }
    if (total_weight <= 0.0f) {
        return RKF_EINVAL;
    }
    for (uint32_t y = 0; y < active->output.height; ++y) {
        for (uint32_t x = 0; x < active->output.width; ++x) {
            for (uint32_t channel = 0; channel < channels; ++channel) {
                float sum = 0.0f;
                int rounded;
                for (uint32_t index = 0; index < active->input_count; ++index) {
                    float weight = active->weights ? active->weights[index] : 1.0f;
                    if (weight > 0.0f) {
                        const ROCKIT_FEATURE_IMAGE_S *image = &active->inputs[index];
                        sum += ((const uint8_t *)image->data)[
                                   (size_t)y * image->stride +
                                   (size_t)x * channels + channel] *
                               weight;
                    }
                }
                rounded = (int)(sum / total_weight + 0.5f);
                if (rounded < 0) {
                    rounded = 0;
                } else if (rounded > 255) {
                    rounded = 255;
                }
                ((uint8_t *)active->output.data)[
                    (size_t)y * active->output.stride +
                    (size_t)x * channels + channel] = (uint8_t)rounded;
            }
        }
    }
    g_avs->config = *active;
    g_avs->frames++;
    return 0;
}

uintptr_t RKGFX_AVS_create_sync(void) {
    uint64_t *sync;
    if (!g_avs) {
        return 0;
    }
    sync = (uint64_t *)rockit_feature_calloc(1, sizeof(*sync));
    if (!sync) {
        return 0;
    }
    *sync = g_avs ? g_avs->frames : 0;
    return (uintptr_t)sync;
}

int RKGFX_AVS_wait_sync(void *sync) {
    if (!g_avs) {
        return RKF_VENDOR_FAILED;
    }
    if (!sync) {
        return RKF_EINVAL;
    }
    free(sync);
    return 0;
}

int RKGFX_AVS_deinit(void) {
    if (!g_avs) {
        return 0;
    }
    free(g_avs);
    g_avs = 0;
    return 0;
}
