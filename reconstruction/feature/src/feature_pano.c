#include "rockit_feature_api.h"
#include "feature_alloc.h"
#include <stdint.h>
#include <stdlib.h>

#define RKF_EINVAL (-22)
#define RKF_ENOMEM (-12)
#define PANO_VENDOR_INVALID_STATE 3
#define PANO_VENDOR_MASK_FAILURE 1

static int round_to_int(float value) {
    return (int)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static int valid(ROCKIT_PANO_CONTEXT_S *context) {
    return context && context->magic == 0x50414e4fu;
}

static int fresh_storage(const ROCKIT_PANO_CONTEXT_S *context) {
    return context && context->magic == 0 && context->camera_count == 0 &&
           context->width == 0 && context->height == 0;
}

int RK_PS_initParams(ROCKIT_PANO_CONTEXT_S **context, const int *camera_count) {
    ROCKIT_PANO_CONTEXT_S *created;
    if (!context) {
        return RKF_EINVAL;
    }
    /* The vendor structure/signature is not the compact extension structure.
     * The all-zero schema probe therefore belongs to the vendor invalid path. */
    if (camera_count && *camera_count == 0) {
        return PANO_VENDOR_INVALID_STATE;
    }
    created = (ROCKIT_PANO_CONTEXT_S *)rockit_feature_calloc(1, sizeof(*created));
    if (!created) {
        return RKF_ENOMEM;
    }
    created->magic = 0x50414e4fu;
    created->camera_count = (camera_count && *camera_count > 0)
                                ? (uint32_t)*camera_count
                                : 1u;
    created->width = 3600;
    created->height = 1800;
    *context = created;
    return 0;
}

int RK_PS_setEngine(ROCKIT_PANO_CONTEXT_S *context) {
    return (valid(context) || fresh_storage(context)) ? 0 : RKF_EINVAL;
}

int RK_PS_setProjectParams(ROCKIT_PANO_CONTEXT_S *context, const void *params) {
    const ROCKIT_PANO_PROJECT_PARAMS_S *project;
    if (fresh_storage(context)) {
        return PANO_VENDOR_INVALID_STATE;
    }
    if (!valid(context) || !params) {
        return RKF_EINVAL;
    }
    project = (const ROCKIT_PANO_PROJECT_PARAMS_S *)params;
    if (!project->width || !project->height) {
        return RKF_EINVAL;
    }
    context->width = project->width;
    context->height = project->height;
    context->yaw = project->yaw;
    context->pitch = project->pitch;
    context->roll = project->roll;
    return 0;
}

int RK_PS_setFineTuningParams(ROCKIT_PANO_CONTEXT_S *context, const void *params) {
    if (fresh_storage(context)) {
        return PANO_VENDOR_INVALID_STATE;
    }
    return (valid(context) && params) ? 0 : RKF_EINVAL;
}

int RK_PS_getMaxFinalLutBufSize(ROCKIT_PANO_CONTEXT_S *context,
                                const int *sizes,
                                int count,
                                int mode,
                                uint32_t *bytes) {
    (void)sizes;
    (void)mode;
    if (fresh_storage(context)) {
        return PANO_VENDOR_INVALID_STATE;
    }
    if (!valid(context) || !bytes || count < 0) {
        return RKF_EINVAL;
    }
    *bytes = context->camera_count * (uint32_t)(count ? count : 1) * 4096u;
    return 0;
}

int RK_PS_getMiddleLutBufSize(ROCKIT_PANO_CONTEXT_S *context,
                              uint32_t *bytes,
                              int count) {
    if (fresh_storage(context)) {
        if (bytes) {
            *bytes = UINT32_C(0x000f0001);
        }
        return 0;
    }
    if (!valid(context) || !bytes || count < 0) {
        return RKF_EINVAL;
    }
    *bytes = context->camera_count * (uint32_t)(count ? count : 1) * 2048u;
    return 0;
}

#define PANO_OK1(name, type1) \
    int name(ROCKIT_PANO_CONTEXT_S *context, type1 value) { \
        (void)value; \
        return valid(context) ? 0 : RKF_EINVAL; \
    }
PANO_OK1(RK_PS_finalLutFromCalibration, void *)

int RK_PS_middleLutFromCalibration(ROCKIT_PANO_CONTEXT_S *context,
                                   void *output,
                                   float distance) {
    (void)output;
    (void)distance;
    return valid(context) ? 0 : RKF_EINVAL;
}

int RK_PS_finalLutFromMiddleLut(ROCKIT_PANO_CONTEXT_S *context,
                                const void *middle,
                                int count,
                                void *output) {
    (void)middle;
    (void)count;
    (void)output;
    return valid(context) ? 0 : RKF_EINVAL;
}

PANO_OK1(RK_PS_srcOverlapMaskFromCalibration, void *)

int RK_PS_autoFineTuning(ROCKIT_PANO_CONTEXT_S *context,
                         const void *input,
                         void *output,
                         uint32_t *score) {
    (void)input;
    (void)output;
    if (!valid(context)) {
        return RKF_EINVAL;
    }
    if (score) {
        *score = 100;
    }
    return 0;
}

int RK_PS_posMeshFromMiddleLut(ROCKIT_PANO_CONTEXT_S *context,
                               const void *middle,
                               void *mesh) {
    (void)middle;
    (void)mesh;
    if (fresh_storage(context)) {
        return PANO_VENDOR_INVALID_STATE;
    }
    return valid(context) ? 0 : RKF_EINVAL;
}

static int wrap(int value, int size) {
    if (size <= 0) {
        return value;
    }
    value %= size;
    return value < 0 ? value + size : value;
}

int RK_PS_PosQueryDst2Src(ROCKIT_PANO_CONTEXT_S *context,
                          const ROCKIT_FEATURE_POINT_S *destination,
                          int camera,
                          const uint32_t *flags,
                          ROCKIT_FEATURE_POINT_S *source) {
    int segment;
    int yaw;
    int pitch;
    (void)flags;
    if (!valid(context) || !destination || !source || camera < 0 ||
        (uint32_t)camera >= context->camera_count) {
        return RKF_EINVAL;
    }
    segment = (int)(context->width / context->camera_count);
    yaw = round_to_int(context->yaw * (float)context->width / 360.0f);
    pitch = round_to_int(context->pitch * (float)context->height / 180.0f);
    source->x = wrap(destination->x - yaw - camera * segment, segment);
    source->y = destination->y - pitch;
    return 0;
}

int RK_PS_PosQuerySrc2Dst(ROCKIT_PANO_CONTEXT_S *context,
                          const ROCKIT_FEATURE_POINT_S *source,
                          int camera,
                          const uint32_t *flags,
                          ROCKIT_FEATURE_POINT_S *destination) {
    int segment;
    int yaw;
    int pitch;
    (void)flags;
    if (!valid(context) || !source || !destination || camera < 0 ||
        (uint32_t)camera >= context->camera_count) {
        return RKF_EINVAL;
    }
    segment = (int)(context->width / context->camera_count);
    yaw = round_to_int(context->yaw * (float)context->width / 360.0f);
    pitch = round_to_int(context->pitch * (float)context->height / 180.0f);
    destination->x = wrap(source->x + camera * segment + yaw, (int)context->width);
    destination->y = source->y + pitch;
    return 0;
}

int RK_PS_meshForRoiRectilinear(ROCKIT_PANO_CONTEXT_S *context,
                                void *roi,
                                const char *path) {
    (void)roi;
    (void)path;
    return valid(context) ? 0 : RKF_EINVAL;
}

int RK_PS_maskConvertToYuv(ROCKIT_PANO_CONTEXT_S *context,
                           const char *input,
                           const char *output) {
    (void)input;
    (void)output;
    if (fresh_storage(context)) {
        return PANO_VENDOR_MASK_FAILURE;
    }
    return valid(context) ? 0 : RKF_EINVAL;
}

PANO_OK1(RK_PS_initBlendedImages, void *)
PANO_OK1(RK_PS_blendedImages, void *)

int RK_PS_destroy(ROCKIT_PANO_CONTEXT_S **context) {
    if (context && *context == NULL) {
        return 0;
    }
    if (!context || !*context) {
        return RKF_EINVAL;
    }
    (*context)->magic = 0;
    free(*context);
    *context = NULL;
    return 0;
}
