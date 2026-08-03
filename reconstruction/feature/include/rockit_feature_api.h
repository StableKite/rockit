#ifndef ROCKIT_FEATURE_API_H
#define ROCKIT_FEATURE_API_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum ROCKIT_FEATURE_FORMAT_E {
    ROCKIT_FEATURE_FORMAT_RAW = 0,
    ROCKIT_FEATURE_FORMAT_GRAY8 = 1,
    ROCKIT_FEATURE_FORMAT_RGB24 = 2,
    ROCKIT_FEATURE_FORMAT_RGBA32 = 3
} ROCKIT_FEATURE_FORMAT_E;

typedef struct ROCKIT_GDC_RADIAL_MODEL_S {
    float center_x;
    float center_y;
    float focal_x;
    float focal_y;
    float k1;
    float k2;
} ROCKIT_GDC_RADIAL_MODEL_S;

typedef struct ROCKIT_FEATURE_POINT_S { int32_t x; int32_t y; } ROCKIT_FEATURE_POINT_S;
typedef struct ROCKIT_FEATURE_SIZE_S { uint32_t width; uint32_t height; } ROCKIT_FEATURE_SIZE_S;
typedef struct ROCKIT_FEATURE_IMAGE_S {
    void *data;
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t format;
} ROCKIT_FEATURE_IMAGE_S;
typedef struct ROCKIT_SWCAC_PARAMS_S {
    ROCKIT_FEATURE_IMAGE_S src;
    ROCKIT_FEATURE_IMAGE_S dst;
    float strength_u;
    float strength_v;
    uint32_t flags;
} ROCKIT_SWCAC_PARAMS_S;
typedef struct ROCKIT_DIS_CONFIG_S { uint32_t width; uint32_t height; uint32_t mesh_x; uint32_t mesh_y; } ROCKIT_DIS_CONFIG_S;
typedef struct ROCKIT_DIS_RESULT_S { int32_t offset_x; int32_t offset_y; uint64_t frame_index; } ROCKIT_DIS_RESULT_S;
typedef struct ROCKIT_GFX_CONTEXT_S { uint32_t magic; uint32_t width; uint32_t height; uint32_t format; uint64_t sequence; } ROCKIT_GFX_CONTEXT_S;
typedef struct ROCKIT_GFX_LAYER_S {
    ROCKIT_FEATURE_IMAGE_S image;
    int32_t dst_x;
    int32_t dst_y;
    uint8_t global_alpha;
    uint8_t reserved[3];
} ROCKIT_GFX_LAYER_S;
typedef struct ROCKIT_GFX_LAYER_LIST_S {
    ROCKIT_FEATURE_IMAGE_S destination;
    const ROCKIT_GFX_LAYER_S *layers;
    uint32_t layer_count;
    uint32_t clear_rgba;
} ROCKIT_GFX_LAYER_LIST_S;
typedef struct ROCKIT_AVS_CONFIG_S {
    const ROCKIT_FEATURE_IMAGE_S *inputs;
    uint32_t input_count;
    ROCKIT_FEATURE_IMAGE_S output;
    const float *weights;
} ROCKIT_AVS_CONFIG_S;
typedef struct ROCKIT_PANO_PROJECT_PARAMS_S {
    uint32_t width;
    uint32_t height;
    float yaw;
    float pitch;
    float roll;
} ROCKIT_PANO_PROJECT_PARAMS_S;
typedef struct ROCKIT_PANO_CONTEXT_S { uint32_t magic; uint32_t camera_count; uint32_t width; uint32_t height; float yaw; float pitch; float roll; } ROCKIT_PANO_CONTEXT_S;

int rk_swcac_init(const ROCKIT_SWCAC_PARAMS_S *params, void **context);
int rk_swcac_proc(const ROCKIT_SWCAC_PARAMS_S *params, void *context);
int rk_swcac_deinit(void *context);
int rk_swcac_loglevl_set(int level);
const char *rk_swcac_getversion(void);

void *RKGFX_LSF_Create(void);
int RKGFX_LSF_Init(void *context, int width, int height, int format);
int RKGFX_LSF_Composite(void *context, const void *layer_list);
uintptr_t RKGFX_LSF_CreateFence(void *context);
int RKGFX_LSF_WaitFence(void *context, void *fence);
int RKGFX_LSF_ReleaseChannel(void *context, int channel);
int RKGFX_LSF_Destroy(void *context);
uint64_t RKGFX_LSF_Calc_Afbc_MemSize(int width, int height, int format);

int RKGFX_AVS_init(void *config);
int RKGFX_AVS_resize(void *config);
int RKGFX_AVS_AllProcess(void *config);
uintptr_t RKGFX_AVS_create_sync(void);
int RKGFX_AVS_wait_sync(void *sync);
int RKGFX_AVS_deinit(void);

int RKALGO_DIS_Init(void **context, const ROCKIT_DIS_CONFIG_S *config);
int RKALGO_DIS_Process(void *context, const ROCKIT_FEATURE_IMAGE_S *input, ROCKIT_DIS_RESULT_S *result, void *mesh);
int RKALGO_DIS_DeInit(void **context);

int rk_algo_gdc_fisheye_version(char *buffer);
int rk_algo_gdc_fisheye_get_mesh_size(const void *attr, uint32_t width, uint32_t height, ROCKIT_FEATURE_SIZE_S *mesh_size);
int rk_algo_gdc_fisheye_generate_mesh(const void *attr, const ROCKIT_FEATURE_SIZE_S *mesh_size, uint32_t width, uint32_t height, float *mesh_x, float *mesh_y);
int rk_algo_gdc_fisheye_pos_query_dst_to_src(const void *attr, const ROCKIT_FEATURE_SIZE_S *size, const ROCKIT_FEATURE_POINT_S *dst, ROCKIT_FEATURE_POINT_S *src);
int rk_algo_gdc_fisheye_pos_query_dst_to_src_array(const void *attr, const ROCKIT_FEATURE_SIZE_S *size, const ROCKIT_FEATURE_POINT_S *dst, uint32_t count, ROCKIT_FEATURE_POINT_S *src);
int rk_algo_gdc_fisheye_pos_query_dst_to_pano(const void *attr, const ROCKIT_FEATURE_SIZE_S *size, uint32_t region, const ROCKIT_FEATURE_POINT_S *dst, ROCKIT_FEATURE_POINT_S *pano);
int rk_algo_gdc_fisheye_pos_query_dst_to_pano_array(const void *attr, const ROCKIT_FEATURE_SIZE_S *size, uint32_t region, const ROCKIT_FEATURE_POINT_S *dst, uint32_t count, ROCKIT_FEATURE_POINT_S *pano);

int RK_PS_initParams(ROCKIT_PANO_CONTEXT_S **context, const int *camera_count);
int RK_PS_setEngine(ROCKIT_PANO_CONTEXT_S *context);
int RK_PS_setProjectParams(ROCKIT_PANO_CONTEXT_S *context, const void *params);
int RK_PS_setFineTuningParams(ROCKIT_PANO_CONTEXT_S *context, const void *params);
int RK_PS_getMaxFinalLutBufSize(ROCKIT_PANO_CONTEXT_S *context, const int *sizes, int count, int mode, uint32_t *bytes);
int RK_PS_getMiddleLutBufSize(ROCKIT_PANO_CONTEXT_S *context, uint32_t *bytes, int count);
int RK_PS_finalLutFromCalibration(ROCKIT_PANO_CONTEXT_S *context, void *out);
int RK_PS_middleLutFromCalibration(ROCKIT_PANO_CONTEXT_S *context, void *out, float distance);
int RK_PS_finalLutFromMiddleLut(ROCKIT_PANO_CONTEXT_S *context, const void *middle, int count, void *out);
int RK_PS_srcOverlapMaskFromCalibration(ROCKIT_PANO_CONTEXT_S *context, void *out);
int RK_PS_autoFineTuning(ROCKIT_PANO_CONTEXT_S *context, const void *input, void *output, uint32_t *score);
int RK_PS_posMeshFromMiddleLut(ROCKIT_PANO_CONTEXT_S *context, const void *middle, void *mesh);
int RK_PS_PosQueryDst2Src(ROCKIT_PANO_CONTEXT_S *context, const ROCKIT_FEATURE_POINT_S *dst, int camera, const uint32_t *flags, ROCKIT_FEATURE_POINT_S *src);
int RK_PS_PosQuerySrc2Dst(ROCKIT_PANO_CONTEXT_S *context, const ROCKIT_FEATURE_POINT_S *src, int camera, const uint32_t *flags, ROCKIT_FEATURE_POINT_S *dst);
int RK_PS_meshForRoiRectilinear(ROCKIT_PANO_CONTEXT_S *context, void *roi, const char *path);
int RK_PS_maskConvertToYuv(ROCKIT_PANO_CONTEXT_S *context, const char *input, const char *output);
int RK_PS_initBlendedImages(ROCKIT_PANO_CONTEXT_S *context, void *images);
int RK_PS_blendedImages(ROCKIT_PANO_CONTEXT_S *context, void *images);
int RK_PS_destroy(ROCKIT_PANO_CONTEXT_S **context);

#ifdef __cplusplus
}
#endif
#endif
