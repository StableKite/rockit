#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "rk_mpi_cal.h"

namespace {

constexpr RK_S32 kCalIllegalParam = static_cast<RK_S32>(0xA0028009U);
constexpr RK_S32 kCalNullPtr = static_cast<RK_S32>(0xA0028006U);
constexpr RK_S32 kProviderUnsupported = -10;

struct Ratio {
    RK_U32 numerator;
    RK_U32 denominator;
};

#define ROCKIT_EXTENDED_PIXEL_RATIO_CASES                                                          \
    case 65540:                                                                                    \
    case 65541:                                                                                    \
        *ratio = {3, 2};                                                                           \
        return true;                                                                               \
    case 65542:                                                                                    \
    case 65543:                                                                                    \
    case 65550:                                                                                    \
    case 65551:                                                                                    \
        *ratio = {3, 1};                                                                           \
        return true;                                                                               \
    case 65544:                                                                                    \
    case 65545:                                                                                    \
        *ratio = {15, 4};                                                                          \
        return true;                                                                               \
    case 65552:                                                                                    \
    case 65553:                                                                                    \
    case 65554:                                                                                    \
    case 65555:                                                                                    \
        *ratio = {4, 1};                                                                           \
        return true;                                                                               \
    case 131072:                                                                                   \
    case 131073:                                                                                   \
    case 131074:                                                                                   \
    case 131075:                                                                                   \
        *ratio = {1, 1};                                                                           \
        return true;                                                                               \
    case 131076:                                                                                   \
    case 131077:                                                                                   \
    case 131078:                                                                                   \
    case 131079:                                                                                   \
    case 131080:                                                                                   \
    case 131081:                                                                                   \
    case 131082:                                                                                   \
    case 131083:                                                                                   \
    case 131085:                                                                                   \
        *ratio = {2, 1};                                                                           \
        return true

constexpr RK_U32 align_up_pow2(RK_U32 value, RK_U32 alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

bool byte_ratio(PIXEL_FORMAT_E format, Ratio *ratio) {
    if (!ratio)
        return false;
    const RK_S32 value = static_cast<RK_S32>(format);
    if (value >= 65536 && value <= 65558) {
        *ratio = {2, 1};
    }
    switch (value) {
    case 0:
    case 2:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 15:
    case 16:
    case 17:
    case 18:
        *ratio = {1, 1};
        return true;
    case 1:
    case 3:
        *ratio = {5, 4};
        return true;
    case 14:
        *ratio = {3, 1};
        return true;
        ROCKIT_EXTENDED_PIXEL_RATIO_CASES;
    default:
        return value >= 65536 && value <= 65558;
    }
}

bool picture_ratio(PIXEL_FORMAT_E format, Ratio *ratio) {
    if (!ratio)
        return false;
    const RK_S32 value = static_cast<RK_S32>(format);
    if (value >= 65536 && value <= 65558) {
        *ratio = {2, 1};
    }
    switch (value) {
    case 0:
    case 4:
    case 5:
    case 6:
    case 13:
        *ratio = {3, 2};
        return true;
    case 1:
        *ratio = {15, 8};
        return true;
    case 2:
    case 7:
    case 8:
    case 9:
    case 10:
    case 12:
    case 17:
    case 18:
        *ratio = {2, 1};
        return true;
    case 3:
        *ratio = {5, 2};
        return true;
    case 11:
        *ratio = {1, 1};
        return true;
    case 14:
    case 15:
    case 16:
        *ratio = {3, 1};
        return true;
        ROCKIT_EXTENDED_PIXEL_RATIO_CASES;
    default:
        return value >= 65536 && value <= 65558;
    }
}

RK_U32 gl_width_alignment(PIXEL_FORMAT_E format) {
    switch (static_cast<RK_S32>(format)) {
    case 0:
    case 6:
        return 16;
    case 4:
    case 5:
        return 64;
    case 65536:
    case 65537:
        return 32;
    case 65542:
    case 65543:
        return 64;
    case 65552:
    case 65553:
    case 65554:
    case 65555:
        return 16;
    case 65546:
    case 65547:
    case 65556:
    case 65557:
        return 32;
    case 7:
    case 2:
    case 8:
        return 16;
    case 9:
    case 10:
    case 1:
        return 32;
    case 14:
        return 64;
    case 15:
        return 32;
    case 16:
        return 64;
    case 65548:
    case 65549:
        return 32;
    case 11:
        return 64;
    default:
        return 16;
    }
}

RK_U32 graphic_vir_width(RK_U32 width, PIXEL_FORMAT_E format) {
    return align_up_pow2(width, gl_width_alignment(format));
}

RK_U32 graphic_vir_height(RK_U32 height, PIXEL_FORMAT_E) {
    return height;
}

RK_U32 afbc_buffer_size(RK_U32 width, RK_U32 height, PIXEL_FORMAT_E format) {
    RK_U32 plane_width[2] = {align_up_pow2(width, 16), 0};
    RK_U32 plane_height[2] = {align_up_pow2(height, 16), 0};
    RK_U32 plane_bits[2] = {32, 0};
    RK_U32 plane_count = 1;

    switch (static_cast<RK_S32>(format)) {
    case 0:
    case 6:
        plane_count = 2;
        plane_bits[0] = 8;
        plane_bits[1] = 16;
        break;
    case 1:
        plane_count = 2;
        plane_bits[0] = 10;
        plane_bits[1] = 20;
        break;
    case 9:
    case 17:
    case 65536:
    case 65537:
    case 65546:
    case 65547:
    case 65556:
    case 65557:
        plane_bits[0] = 16;
        break;
    case 65542:
    case 65543:
        plane_bits[0] = 24;
        break;
    case 65552:
    case 65553:
    case 65554:
    case 65555:
        plane_bits[0] = 32;
        break;
    default:
        plane_bits[0] = 32;
        break;
    }

    if (plane_count == 2) {
        plane_width[1] = align_up_pow2(width / 2U, 16);
        plane_height[1] = align_up_pow2(height / 2U, 16);
    }

    RK_U32 total = 0;
    for (RK_U32 plane = 0; plane < plane_count; ++plane) {
        const RK_U32 pixels = plane_width[plane] * plane_height[plane];
        RK_U32 body = align_up_pow2(32U * plane_bits[plane], 128) * (pixels / 256U);
        if (plane_count == 2)
            body = align_up_pow2(body, 4096);
        const RK_U32 header = align_up_pow2(pixels / 16U, 4096);
        total += header + body;
    }
    return total;
}

#define ROCKIT_FINISH_PICTURE_BUFFER_SIZE(linear_size_)                                            \
    if (attribute->enCompMode == COMPRESS_AFBC_16x16) {                                            \
        calculated.u32MBSize = afbc_buffer_size(                                                   \
            calculated.u32VirWidth, calculated.u32VirHeight, attribute->enPixelFormat);            \
    } else {                                                                                       \
        Ratio ratio{};                                                                             \
        if (!picture_ratio(attribute->enPixelFormat, &ratio))                                      \
            return kProviderUnsupported;                                                           \
        calculated.u32MBSize = linear_size_;                                                       \
    }                                                                                              \
    if (result)                                                                                    \
        *result = calculated;                                                                      \
    return 0

RK_S32 common_buffer_size(const PIC_BUF_ATTR_S *attribute, MB_PIC_CAL_S *result) {
    if (!attribute)
        return -3;
    if (attribute->u32Width == 0 || attribute->u32Height == 0 ||
        static_cast<RK_U32>(attribute->enCompMode) > 1U)
        return -2;

    MB_PIC_CAL_S calculated{};
    calculated.u32VirWidth = align_up_pow2(attribute->u32Width, 16);
    calculated.u32VirHeight = align_up_pow2(attribute->u32Height, 16);
    ROCKIT_FINISH_PICTURE_BUFFER_SIZE(calculated.u32VirWidth * calculated.u32VirHeight *
                                      ratio.numerator / ratio.denominator);
}

RK_S32 graphic_buffer_size(const PIC_BUF_ATTR_S *attribute, MB_PIC_CAL_S *result) {
    if (!attribute)
        return -3;
    if (attribute->u32Width == 0 || attribute->u32Height == 0 ||
        static_cast<RK_U32>(attribute->enCompMode) > 1U)
        return -2;

    MB_PIC_CAL_S calculated{};
    calculated.u32VirWidth = graphic_vir_width(attribute->u32Width, attribute->enPixelFormat);
    calculated.u32VirHeight = graphic_vir_height(attribute->u32Height, attribute->enPixelFormat);
    ROCKIT_FINISH_PICTURE_BUFFER_SIZE(align_up_pow2(calculated.u32VirWidth, 16) *
                                      align_up_pow2(calculated.u32VirHeight, 16) * ratio.numerator /
                                      ratio.denominator);
}

bool rga2_enabled() {
    const char *value = std::getenv("rt_rga2_enable");
    if (!value || !*value)
        return false;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 0);
    return end != value && parsed != 0;
}

bool rga2_supports(PIXEL_FORMAT_E format) {
    switch (static_cast<RK_S32>(format)) {
    case 4:
    case 5:
    case 0:
    case 6:
    case 11:
    case 7:
    case 2:
    case 8:
    case 10:
    case 9:
    case 17:
    case 18:
    case 65536:
    case 65537:
    case 65542:
    case 65543:
    case 65552:
    case 65553:
    case 65554:
    case 65555:
        return true;
    default:
        return false;
    }
}

bool rga3_supports(PIXEL_FORMAT_E format) {
    switch (static_cast<RK_S32>(format)) {
    case 0:
    case 6:
    case 1:
    case 2:
    case 8:
    case 3:
    case 10:
    case 9:
    case 17:
    case 18:
    case 65536:
    case 65537:
    case 65542:
    case 65543:
    case 65555:
    case 65554:
    case 65552:
        return true;
    default:
        return false;
    }
}

RK_U32 rga3_width_alignment(PIXEL_FORMAT_E format) {
    switch (static_cast<RK_S32>(format)) {
    case 0:
    case 6:
    case 4:
    case 5:
        return 16;
    case 65536:
    case 65537:
        return 8;
    case 65542:
    case 65543:
        return 16;
    case 65552:
    case 65553:
    case 65554:
    case 65555:
        return 4;
    case 65556:
        return 2;
    case 7:
    case 2:
    case 8:
    case 9:
    case 10:
        return 16;
    case 1:
        return 64;
    case 11:
        return 16;
    default:
        return 1;
    }
}

RK_U32 rga_height_alignment(PIXEL_FORMAT_E format, COMPRESS_MODE_E compression) {
    switch (static_cast<RK_S32>(format)) {
    case 0:
    case 6:
    case 4:
    case 5:
    case 7:
    case 2:
    case 8:
    case 9:
    case 10:
    case 1:
    case 11:
        return compression == COMPRESS_AFBC_16x16 ? 16U : 2U;
    default:
        return 1;
    }
}

RK_S32 rga_buffer_size(const PIC_BUF_ATTR_S *attribute, MB_PIC_CAL_S *result) {
    if (!attribute)
        return -3;
    if (attribute->u32Width == 0 || attribute->u32Height == 0 ||
        static_cast<RK_U32>(attribute->enCompMode) > 1U)
        return -2;

    MB_PIC_CAL_S calculated{};
    calculated.u32VirWidth =
        align_up_pow2(attribute->u32Width, rga3_width_alignment(attribute->enPixelFormat));
    calculated.u32VirHeight =
        align_up_pow2(attribute->u32Height,
                      rga_height_alignment(attribute->enPixelFormat, attribute->enCompMode));
    ROCKIT_FINISH_PICTURE_BUFFER_SIZE(calculated.u32VirWidth * calculated.u32VirHeight *
                                      ratio.numerator / ratio.denominator);
}

#undef ROCKIT_FINISH_PICTURE_BUFFER_SIZE

RK_U32 vdec_vir_width(const VDEC_PIC_BUF_ATTR_S *attribute) {
    if (!attribute || static_cast<RK_U32>(attribute->enCodecType - 1) >= 0x3FFFFFFEU ||
        attribute->stPicBufAttr.u32Width == 0)
        return 0;

    const RK_U32 width = attribute->stPicBufAttr.u32Width;
    const RK_U32 height = attribute->stPicBufAttr.u32Height;
    const bool afbc = attribute->stPicBufAttr.enCompMode == COMPRESS_AFBC_16x16;
    const RK_S32 codec = static_cast<RK_S32>(attribute->enCodecType);

    if (afbc)
        return align_up_pow2(width, 64);
    if (codec == 11 || codec == 12)
        return ((width + 255U) & 0xFFFFFE00U) | 0x100U;
    if (codec == 8 && !(width <= 1920U && height <= 1088U))
        return ((width + 255U) & 0xFFFFFE00U) | 0x100U;
    return align_up_pow2(width, 16);
}

RK_U32 vdec_vir_height(const VDEC_PIC_BUF_ATTR_S *attribute) {
    if (!attribute || static_cast<RK_U32>(attribute->enCodecType - 1) >= 0x3FFFFFFEU ||
        attribute->stPicBufAttr.u32Height == 0)
        return 0;

    const RK_U32 height = attribute->stPicBufAttr.u32Height;
    if (attribute->stPicBufAttr.enCompMode == COMPRESS_AFBC_16x16)
        return align_up_pow2(height, 16) + 16U;
    if (static_cast<RK_S32>(attribute->enCodecType) == 12)
        return align_up_pow2(height, 8);
    return align_up_pow2(height, 16);
}

RK_S32 vdec_buffer_size(const VDEC_PIC_BUF_ATTR_S *attribute, MB_PIC_CAL_S *result) {
    MB_PIC_CAL_S calculated{};
    const RK_U32 virtual_width = vdec_vir_width(attribute);
    const RK_U32 virtual_height = vdec_vir_height(attribute);
    if (!virtual_width || !virtual_height)
        return -2;

    Ratio ratio{};
    if (!picture_ratio(attribute->stPicBufAttr.enPixelFormat, &ratio))
        return kProviderUnsupported;

    const RK_U32 area = virtual_width * virtual_height;
    RK_U32 payload = 0;
    RK_U32 header = 0;
    RK_U32 extra = 0;
    if (attribute->stPicBufAttr.enCompMode == COMPRESS_AFBC_16x16) {
        header = align_up_pow2(16U * ratio.numerator / ratio.denominator, 64) * (area >> 8U);
        payload = align_up_pow2((ratio.numerator << 8U) / ratio.denominator, 64) * (area >> 8U);
    } else {
        payload = virtual_width * ratio.numerator * virtual_height / ratio.denominator;
        extra = area / 2U;
    }

    calculated.u32MBSize = payload + header + extra;
    calculated.u32VirWidth = virtual_width;
    calculated.u32VirHeight = virtual_height;
    if (result)
        *result = calculated;
    return 0;
}

#undef ROCKIT_EXTENDED_PIXEL_RATIO_CASES

} // namespace

extern "C" RK_S32 RK_MPI_CAL_COMM_GetPicBufferSize(const PIC_BUF_ATTR_S *attribute,
                                                   MB_PIC_CAL_S *result) {
    if (!result)
        return kCalNullPtr;
    MB_PIC_CAL_S calculated{};
    const RK_S32 rc = common_buffer_size(attribute, &calculated);
    if (rc != 0)
        return kCalIllegalParam;
    *result = calculated;
    return RK_SUCCESS;
}

extern "C" RK_S32 RK_MPI_CAL_VGS_GetPicBufferSize(const PIC_BUF_ATTR_S *attribute,
                                                  MB_PIC_CAL_S *result) {
    if (!result)
        return kCalNullPtr;
    MB_PIC_CAL_S calculated{};
    const RK_S32 rc = graphic_buffer_size(attribute, &calculated);
    if (rc != 0)
        return kCalIllegalParam;
    *result = calculated;
    return RK_SUCCESS;
}

extern "C" RK_S32 RK_MPI_CAL_TDE_GetPicBufferSize(const PIC_BUF_ATTR_S *attribute,
                                                  MB_PIC_CAL_S *result) {
    if (!result)
        return kCalNullPtr;
    // The hardware TDE provider accepts an all-zero descriptor and publishes
    // an all-zero result before selecting an RGA generation.
    if (attribute && attribute->u32Width == 0 && attribute->u32Height == 0 &&
        attribute->enPixelFormat == static_cast<PIXEL_FORMAT_E>(0) &&
        attribute->enCompMode == static_cast<COMPRESS_MODE_E>(0)) {
        *result = MB_PIC_CAL_S{};
        return RK_SUCCESS;
    }
    const bool supported = rga2_enabled()
                               ? rga2_supports(attribute ? attribute->enPixelFormat : RK_FMT_BUTT)
                               : rga3_supports(attribute ? attribute->enPixelFormat : RK_FMT_BUTT);
    if (!supported)
        return RK_MPI_CAL_VGS_GetPicBufferSize(attribute, result);
    MB_PIC_CAL_S calculated{};
    const RK_S32 rc = rga_buffer_size(attribute, &calculated);
    if (rc != 0)
        return kCalIllegalParam;
    *result = calculated;
    return RK_SUCCESS;
}

extern "C" RK_S32 RK_MPI_CAL_VDEC_GetPicBufferSize(const VDEC_PIC_BUF_ATTR_S *attribute,
                                                   MB_PIC_CAL_S *result) {
    if (!result)
        return kCalNullPtr;
    MB_PIC_CAL_S calculated{};
    const RK_S32 rc = vdec_buffer_size(attribute, &calculated);
    if (rc != 0)
        return kCalIllegalParam;
    *result = calculated;
    return RK_SUCCESS;
}

extern "C" RK_S32 RK_MPI_CAL_AVS_GetFinalLutBufferSize(const PIC_BUF_ATTR_S *attribute,
                                                       MB_PIC_CAL_S *result) {
    if (!attribute || !result)
        return kCalNullPtr;
    const RK_U32 rows = ((attribute->u32Height + 7U) >> 3U) + 1U;
    result->u32VirWidth = rows;
    result->u32MBSize = 2U * (((((attribute->u32Width + 15U) >> 4U) + 2U) & ~1U) * rows + 6U);
    return RK_SUCCESS;
}

extern "C" RK_U32 RK_MPI_CAL_VGS_GetPicVirWidth(RK_U32 width, PIXEL_FORMAT_E format) {
    return graphic_vir_width(width, format);
}

extern "C" RK_U32 RK_MPI_CAL_VGS_GetPicVirHeight(RK_U32 height, PIXEL_FORMAT_E format) {
    return graphic_vir_height(height, format);
}

extern "C" RK_U32 RK_MPI_CAL_VDEC_GetVirWidth(const VDEC_PIC_BUF_ATTR_S *attribute) {
    return vdec_vir_width(attribute);
}

extern "C" RK_U32 RK_MPI_CAL_VDEC_GetVirHeight(const VDEC_PIC_BUF_ATTR_S *attribute) {
    return vdec_vir_height(attribute);
}

extern "C" RK_U32 RK_MPI_CAL_COMM_GetHorStride(RK_U32 width, PIXEL_FORMAT_E format) {
    Ratio ratio{};
    if (!byte_ratio(format, &ratio))
        return static_cast<RK_U32>(kProviderUnsupported);
    return width * ratio.numerator / ratio.denominator;
}

extern "C" RK_U32 RK_MPI_CAL_COMM_GetVirWidth(RK_U32 stride, PIXEL_FORMAT_E format) {
    Ratio ratio{};
    if (!byte_ratio(format, &ratio))
        return static_cast<RK_U32>(kProviderUnsupported);
    const RK_U32 integer_bytes = ratio.numerator / ratio.denominator;
    return integer_bytes ? stride / integer_bytes : static_cast<RK_U32>(kProviderUnsupported);
}
