#include "rockit_feature_api.h"
#include <stdint.h>
#include <string.h>

#define RKF_EINVAL (-22)
#define RKF_VENDOR_FAILURE (-1)

static int round_to_int(float value) {
    return (int)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static void map_point(const ROCKIT_GDC_RADIAL_MODEL_S *model,
                      float x,
                      float y,
                      float *output_x,
                      float *output_y) {
    float normalized_x;
    float normalized_y;
    float radius_squared;
    float factor;
    if (!model || model->focal_x == 0.0f || model->focal_y == 0.0f) {
        *output_x = x;
        *output_y = y;
        return;
    }
    normalized_x = (x - model->center_x) / model->focal_x;
    normalized_y = (y - model->center_y) / model->focal_y;
    radius_squared = normalized_x * normalized_x + normalized_y * normalized_y;
    factor = 1.0f + model->k1 * radius_squared +
             model->k2 * radius_squared * radius_squared;
    *output_x = model->center_x + normalized_x * factor * model->focal_x;
    *output_y = model->center_y + normalized_y * factor * model->focal_y;
}

int rk_algo_gdc_fisheye_version(char *buffer) {
    if (!buffer) {
        return RKF_VENDOR_FAILURE;
    }
    strcpy(buffer, "RK GDC Version: 0.0.10.58 date: 2022-07-22");
    return 0;
}

int rk_algo_gdc_fisheye_get_mesh_size(const void *attr,
                                      uint32_t width,
                                      uint32_t height,
                                      ROCKIT_FEATURE_SIZE_S *mesh_size) {
    if (!attr || !mesh_size || !width || !height) {
        return RKF_VENDOR_FAILURE;
    }
    mesh_size->width = (width + 15u) / 16u + 1u;
    mesh_size->height = (height + 15u) / 16u + 1u;
    return 0;
}

int rk_algo_gdc_fisheye_generate_mesh(const void *attr,
                                      const ROCKIT_FEATURE_SIZE_S *mesh_size,
                                      uint32_t width,
                                      uint32_t height,
                                      float *mesh_x,
                                      float *mesh_y) {
    const ROCKIT_GDC_RADIAL_MODEL_S *model =
        (const ROCKIT_GDC_RADIAL_MODEL_S *)attr;
    if (!mesh_size || !mesh_x || !mesh_y || mesh_size->width < 2 ||
        mesh_size->height < 2 || !width || !height) {
        return RKF_VENDOR_FAILURE;
    }
    for (uint32_t y = 0; y < mesh_size->height; ++y) {
        for (uint32_t x = 0; x < mesh_size->width; ++x) {
            size_t index = (size_t)y * mesh_size->width + x;
            float destination_x =
                (float)x * (float)(width - 1u) / (float)(mesh_size->width - 1u);
            float destination_y =
                (float)y * (float)(height - 1u) / (float)(mesh_size->height - 1u);
            map_point(model,
                      destination_x,
                      destination_y,
                      &mesh_x[index],
                      &mesh_y[index]);
        }
    }
    return 0;
}

static int map_array(const void *attr,
                     const ROCKIT_FEATURE_POINT_S *destination,
                     uint32_t count,
                     ROCKIT_FEATURE_POINT_S *output) {
    const ROCKIT_GDC_RADIAL_MODEL_S *model =
        (const ROCKIT_GDC_RADIAL_MODEL_S *)attr;
    if (!attr || !destination || !output || count == 0 ||
        model->focal_x == 0.0f || model->focal_y == 0.0f) {
        return RKF_VENDOR_FAILURE;
    }
    for (uint32_t index = 0; index < count; ++index) {
        float output_x;
        float output_y;
        map_point(model,
                  (float)destination[index].x,
                  (float)destination[index].y,
                  &output_x,
                  &output_y);
        output[index].x = (int32_t)round_to_int(output_x);
        output[index].y = (int32_t)round_to_int(output_y);
    }
    return 0;
}

int rk_algo_gdc_fisheye_pos_query_dst_to_src_array(
    const void *attr,
    const ROCKIT_FEATURE_SIZE_S *size,
    const ROCKIT_FEATURE_POINT_S *destination,
    uint32_t count,
    ROCKIT_FEATURE_POINT_S *source) {
    (void)size;
    return map_array(attr, destination, count, source);
}

int rk_algo_gdc_fisheye_pos_query_dst_to_src(
    const void *attr,
    const ROCKIT_FEATURE_SIZE_S *size,
    const ROCKIT_FEATURE_POINT_S *destination,
    ROCKIT_FEATURE_POINT_S *source) {
    return rk_algo_gdc_fisheye_pos_query_dst_to_src_array(
        attr, size, destination, 1, source);
}

int rk_algo_gdc_fisheye_pos_query_dst_to_pano_array(
    const void *attr,
    const ROCKIT_FEATURE_SIZE_S *size,
    uint32_t region,
    const ROCKIT_FEATURE_POINT_S *destination,
    uint32_t count,
    ROCKIT_FEATURE_POINT_S *pano) {
    (void)region;
    return rk_algo_gdc_fisheye_pos_query_dst_to_src_array(
        attr, size, destination, count, pano);
}

int rk_algo_gdc_fisheye_pos_query_dst_to_pano(
    const void *attr,
    const ROCKIT_FEATURE_SIZE_S *size,
    uint32_t region,
    const ROCKIT_FEATURE_POINT_S *destination,
    ROCKIT_FEATURE_POINT_S *pano) {
    return rk_algo_gdc_fisheye_pos_query_dst_to_pano_array(
        attr, size, region, destination, 1, pano);
}
