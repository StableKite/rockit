#include "rockit_feature_api.h"
#include "feature_alloc.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RKF_EINVAL (-22)
#define RKF_ENOMEM (-12)
#define RKF_VENDOR_FAILURE (-1)

static int round_to_int(float value) {
    return (int)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

typedef struct SwCtx {
    uint32_t magic;
    uint64_t frames;
} SwCtx;

static atomic_int g_log_level = 0;

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static float clampf_local(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    return value > high ? high : value;
}

static uint8_t sample_nearest(const ROCKIT_FEATURE_IMAGE_S *image,
                              int x,
                              int y,
                              uint32_t channel,
                              uint32_t channels) {
    if (x < 0) {
        x = 0;
    }
    if (y < 0) {
        y = 0;
    }
    if ((uint32_t)x >= image->width) {
        x = (int)image->width - 1;
    }
    if ((uint32_t)y >= image->height) {
        y = (int)image->height - 1;
    }
    return ((const uint8_t *)image->data)[
        (size_t)y * image->stride + (size_t)x * channels + channel];
}

int rk_swcac_init(const ROCKIT_SWCAC_PARAMS_S *params, void **context) {
    SwCtx *ctx;
    if (!context) {
        return RKF_EINVAL;
    }
    /* The vendor ABI has a much larger parameter record.  An all-zero
     * record is its deterministic fresh/error path and must not allocate. */
    if (params && params->src.data == NULL && params->dst.data == NULL &&
        params->src.width == 0 && params->src.height == 0 &&
        params->dst.width == 0 && params->dst.height == 0) {
        ctx = (SwCtx *)rockit_feature_calloc(1, sizeof(*ctx));
        if (ctx) {
            ctx->magic = 0x53574341u;
            *context = ctx;
        }
        return RKF_VENDOR_FAILURE;
    }
    ctx = (SwCtx *)rockit_feature_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return RKF_ENOMEM;
    }
    ctx->magic = 0x53574341u;
    *context = ctx;
    (void)params;
    return 0;
}

int rk_swcac_proc(const ROCKIT_SWCAC_PARAMS_S *params, void *context) {
    SwCtx *ctx = (SwCtx *)context;
    uint32_t channels;
    float center_x;
    float center_y;
    float normalization;
    float red_strength;
    float blue_strength;

    if (!ctx || ctx->magic != 0x53574341u || !params ||
        !params->src.data || !params->dst.data) {
        if (ctx && params && ctx->magic == 0 &&
            params->src.data == NULL && params->dst.data == NULL) {
            /* Vendor increments the call counter in its opaque channel record
             * before rejecting zero geometry. */
            ((uint8_t *)ctx)[4] = 1;
            return RKF_VENDOR_FAILURE;
        }
        return RKF_EINVAL;
    }
    if (params->src.width != params->dst.width ||
        params->src.height != params->dst.height ||
        params->src.format != params->dst.format) {
        return RKF_EINVAL;
    }
    if (params->src.format == ROCKIT_FEATURE_FORMAT_RAW) {
        memmove(params->dst.data,
                params->src.data,
                min_size(params->src.size, params->dst.size));
        ctx->frames++;
        return 0;
    }

    channels = params->src.format == ROCKIT_FEATURE_FORMAT_RGB24
                   ? 3u
                   : (params->src.format == ROCKIT_FEATURE_FORMAT_RGBA32 ? 4u : 0u);
    if (!channels ||
        params->src.stride < params->src.width * channels ||
        params->dst.stride < params->dst.width * channels) {
        return RKF_EINVAL;
    }
    if (params->src.size < (size_t)params->src.stride * params->src.height ||
        params->dst.size < (size_t)params->dst.stride * params->dst.height) {
        return RKF_EINVAL;
    }

    center_x = ((float)params->src.width - 1.0f) * 0.5f;
    center_y = ((float)params->src.height - 1.0f) * 0.5f;
    normalization = center_x > center_y ? center_x : center_y;
    red_strength = clampf_local(params->strength_u, -0.75f, 0.75f);
    blue_strength = clampf_local(params->strength_v, -0.75f, 0.75f);

    for (uint32_t y = 0; y < params->src.height; ++y) {
        for (uint32_t x = 0; x < params->src.width; ++x) {
            float dx = ((float)x - center_x) / (normalization > 0 ? normalization : 1.0f);
            float dy = ((float)y - center_y) / (normalization > 0 ? normalization : 1.0f);
            float radius_squared = dx * dx + dy * dy;
            int red_x = round_to_int(center_x + ((float)x - center_x) *
                                                  (1.0f + red_strength * radius_squared));
            int red_y = round_to_int(center_y + ((float)y - center_y) *
                                                  (1.0f + red_strength * radius_squared));
            int blue_x = round_to_int(center_x + ((float)x - center_x) *
                                                   (1.0f + blue_strength * radius_squared));
            int blue_y = round_to_int(center_y + ((float)y - center_y) *
                                                   (1.0f + blue_strength * radius_squared));
            uint8_t *output = (uint8_t *)params->dst.data +
                              (size_t)y * params->dst.stride + (size_t)x * channels;
            output[0] = sample_nearest(&params->src, red_x, red_y, 0, channels);
            output[1] = sample_nearest(&params->src, (int)x, (int)y, 1, channels);
            output[2] = sample_nearest(&params->src, blue_x, blue_y, 2, channels);
            if (channels == 4) {
                output[3] = sample_nearest(&params->src, (int)x, (int)y, 3, channels);
            }
        }
    }
    ctx->frames++;
    return 0;
}

int rk_swcac_deinit(void *context) {
    SwCtx *ctx = (SwCtx *)context;
    if (!ctx || ctx->magic != 0x53574341u) {
        return RKF_EINVAL;
    }
    ctx->magic = 0;
    free(ctx);
    return 0;
}

int rk_swcac_loglevl_set(int level) {
    atomic_store(&g_log_level, level);
    return 0;
}

const char *rk_swcac_getversion(void) {
    return "RKCAC1.0.0.1577.202206241408";
}
