#include "rockit_internal.hpp"
#include "rgba_pixel_cases.hpp"
#include "local_frame_cpu.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rk_defines.h"
#include "rk_mpi_avs.h"
#include "rk_mpi_cal.h"
#include "rk_mpi_gdc.h"
#include "rk_mpi_ivs.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_pvs.h"
#include "rk_mpi_rgn.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_tde.h"
#include "rk_mpi_vgs.h"

namespace {

template <typename T> void zero_value(T *v) {
    if (v)
        std::memset(v, 0, sizeof(T));
}
RK_U32 align_up(RK_U32 v, RK_U32 a) {
    return a ? ((v + a - 1U) / a) * a : v;
}

RK_U32 bits_per_pixel(PIXEL_FORMAT_E fmt) {
    switch (fmt) {
    case RK_FMT_YUV420SP:
    case RK_FMT_YUV420P:
    case RK_FMT_YUV420P_VU:
    case RK_FMT_YUV420SP_VU:
        return 12;
    case RK_FMT_YUV420SP_10BIT:
        return 15;
    case RK_FMT_YUV400SP:
    case RK_FMT_RGB_BAYER_SBGGR_8BPP:
    case RK_FMT_RGB_BAYER_SGBRG_8BPP:
    case RK_FMT_RGB_BAYER_SGRBG_8BPP:
    case RK_FMT_RGB_BAYER_SRGGB_8BPP:
        return 8;
    case RK_FMT_YUV422SP:
    case RK_FMT_YUV422P:
    case RK_FMT_YUV422SP_VU:
    case RK_FMT_YUV422_YUYV:
    case RK_FMT_YUV422_UYVY:
    case RK_FMT_YUV422_YVYU:
    case RK_FMT_YUV422_VYUY:
    case RK_FMT_RGB565:
    case RK_FMT_BGR565:
    case RK_FMT_RGB555:
    case RK_FMT_BGR555:
    case RK_FMT_RGB444:
    case RK_FMT_BGR444:
    case RK_FMT_ARGB1555:
    case RK_FMT_ABGR1555:
    case RK_FMT_ARGB4444:
    case RK_FMT_ABGR4444:
    case RK_FMT_RGBA5551:
    case RK_FMT_BGRA5551:
    case RK_FMT_BGRA4444:
    case RK_FMT_RGBA4444:
    case RK_FMT_RGB_BAYER_SBGGR_16BPP:
        return 16;
    case RK_FMT_YUV422SP_10BIT:
        return 20;
    case RK_FMT_RGB888:
    case RK_FMT_BGR888:
    case RK_FMT_ARGB8565:
    case RK_FMT_ABGR8565:
    case RK_FMT_YUV444:
    case RK_FMT_YUV444SP:
    case RK_FMT_YUV444P:
        return 24;
    case RK_FMT_RGB101010:
    case RK_FMT_BGR101010:
        return 30;
    case RK_FMT_ARGB8888:
    case RK_FMT_ABGR8888:
    case RK_FMT_BGRA8888:
    case RK_FMT_RGBA8888:
    case RK_FMT_XBGR8888:
        return 32;
    case RK_FMT_2BPP:
        return 2;
    case RK_FMT_RGB_BAYER_SBGGR_10BPP:
    case RK_FMT_RGB_BAYER_SGBRG_10BPP:
    case RK_FMT_RGB_BAYER_SGRBG_10BPP:
    case RK_FMT_RGB_BAYER_SRGGB_10BPP:
        return 10;
    case RK_FMT_RGB_BAYER_SBGGR_12BPP:
    case RK_FMT_RGB_BAYER_SGBRG_12BPP:
    case RK_FMT_RGB_BAYER_SGRBG_12BPP:
    case RK_FMT_RGB_BAYER_SRGGB_12BPP:
        return 12;
    case RK_FMT_RGB_BAYER_14BPP:
        return 14;
    default:
        return 0;
    }
}

bool align_u64(RK_U64 value, RK_U64 alignment, RK_U64 *aligned) {
    if (!aligned || alignment == 0 || value > UINT64_MAX - (alignment - 1U))
        return false;
    *aligned = ((value + alignment - 1U) / alignment) * alignment;
    return true;
}

#define ROCKIT_BYTE_STRIDE_YUV_CASES                                                               \
    case RK_FMT_YUV420SP:                                                                          \
    case RK_FMT_YUV420P:                                                                           \
    case RK_FMT_YUV420P_VU:                                                                        \
    case RK_FMT_YUV420SP_VU:                                                                       \
    case RK_FMT_YUV422SP:                                                                          \
    case RK_FMT_YUV422P:                                                                           \
    case RK_FMT_YUV422SP_VU:                                                                       \
    case RK_FMT_YUV400SP:                                                                          \
    case RK_FMT_YUV440SP:                                                                          \
    case RK_FMT_YUV411SP:                                                                          \
    case RK_FMT_YUV444SP:                                                                          \
    case RK_FMT_YUV444P:

bool row_stride_bytes(RK_U32 width, PIXEL_FORMAT_E format, RK_U64 *stride) {
    if (!stride || width == 0)
        return false;
    RK_U64 raw = 0;
    switch (format) {
        ROCKIT_BYTE_STRIDE_YUV_CASES
        raw = width;
        break;
    case RK_FMT_YUV420SP_10BIT:
    case RK_FMT_YUV422SP_10BIT:
        raw = (static_cast<RK_U64>(width) * 10U + 7U) / 8U;
        break;
    default: {
        const RK_U32 bits = bits_per_pixel(format);
        if (!bits)
            return false;
        raw = (static_cast<RK_U64>(width) * bits + 7U) / 8U;
        break;
    }
    }
    return align_u64(raw, 16, stride);
}

RK_U32 vir_width(RK_U32 width, PIXEL_FORMAT_E format) {
    RK_U64 stride = 0;
    if (!row_stride_bytes(width, format, &stride) || stride > UINT32_MAX)
        return 0;
    switch (format) {
        ROCKIT_BYTE_STRIDE_YUV_CASES
        return static_cast<RK_U32>(stride);
    case RK_FMT_YUV420SP_10BIT:
    case RK_FMT_YUV422SP_10BIT: {
        const RK_U64 pixels = (stride * 8U + 9U) / 10U;
        return static_cast<RK_U32>(std::min<RK_U64>(pixels, UINT32_MAX));
    }
    default: {
        const RK_U32 bits = bits_per_pixel(format);
        if (!bits)
            return 0;
        const RK_U64 pixels = (stride * 8U + bits - 1U) / bits;
        return static_cast<RK_U32>(std::min<RK_U64>(pixels, UINT32_MAX));
    }
    }
}

#undef ROCKIT_BYTE_STRIDE_YUV_CASES

bool add_plane(RK_U64 stride, RK_U64 rows, RK_U64 *total) {
    if (!total || (rows != 0 && stride > UINT64_MAX / rows))
        return false;
    const RK_U64 bytes = stride * rows;
    if (*total > UINT64_MAX - bytes)
        return false;
    *total += bytes;
    return true;
}

RK_S32 calc_pic(const PIC_BUF_ATTR_S *attribute, MB_PIC_CAL_S *calculation) {
    if (!attribute || !calculation || attribute->u32Width == 0 || attribute->u32Height == 0 ||
        attribute->enCompMode != COMPRESS_MODE_NONE) {
        return RK_FAILURE;
    }
    const RK_U32 virtualWidth = vir_width(attribute->u32Width, attribute->enPixelFormat);
    const RK_U32 virtualHeight = align_up(attribute->u32Height, 2);
    RK_U64 yStride = 0;
    if (virtualWidth == 0 || virtualHeight < attribute->u32Height ||
        !row_stride_bytes(attribute->u32Width, attribute->enPixelFormat, &yStride)) {
        return RK_FAILURE;
    }
    RK_U64 bytes = 0;
    const RK_U64 rows = virtualHeight;
    switch (attribute->enPixelFormat) {
    case RK_FMT_YUV420SP:
    case RK_FMT_YUV420SP_VU:
    case RK_FMT_YUV420SP_10BIT:
        if (!add_plane(yStride, rows, &bytes) || !add_plane(yStride, (rows + 1U) / 2U, &bytes))
            return RK_FAILURE;
        break;
    case RK_FMT_YUV420P:
    case RK_FMT_YUV420P_VU: {
        RK_U64 chromaStride = 0;
        if (!align_u64((virtualWidth + 1U) / 2U, 16, &chromaStride) ||
            !add_plane(yStride, rows, &bytes) ||
            !add_plane(chromaStride, (rows + 1U) / 2U, &bytes) ||
            !add_plane(chromaStride, (rows + 1U) / 2U, &bytes))
            return RK_FAILURE;
        break;
    }
    case RK_FMT_YUV422SP:
    case RK_FMT_YUV422SP_VU:
    case RK_FMT_YUV422SP_10BIT:
        if (!add_plane(yStride, rows, &bytes) || !add_plane(yStride, rows, &bytes))
            return RK_FAILURE;
        break;
    case RK_FMT_YUV422P: {
        RK_U64 chromaStride = 0;
        if (!align_u64((virtualWidth + 1U) / 2U, 16, &chromaStride) ||
            !add_plane(yStride, rows, &bytes) || !add_plane(chromaStride, rows, &bytes) ||
            !add_plane(chromaStride, rows, &bytes))
            return RK_FAILURE;
        break;
    }
    case RK_FMT_YUV440SP:
        if (!add_plane(yStride, rows, &bytes) || !add_plane(yStride * 2U, (rows + 1U) / 2U, &bytes))
            return RK_FAILURE;
        break;
    case RK_FMT_YUV411SP: {
        RK_U64 chromaStride = 0;
        if (!align_u64(((virtualWidth + 3U) / 4U) * 2U, 16, &chromaStride) ||
            !add_plane(yStride, rows, &bytes) || !add_plane(chromaStride, rows, &bytes))
            return RK_FAILURE;
        break;
    }
    case RK_FMT_YUV444SP:
        if (!add_plane(yStride, rows, &bytes) || !add_plane(yStride * 2U, rows, &bytes))
            return RK_FAILURE;
        break;
    case RK_FMT_YUV444P:
        if (!add_plane(yStride, rows, &bytes) || !add_plane(yStride, rows, &bytes) ||
            !add_plane(yStride, rows, &bytes))
            return RK_FAILURE;
        break;
    default:
        if (!bits_per_pixel(attribute->enPixelFormat) || !add_plane(yStride, rows, &bytes))
            return RK_FAILURE;
        break;
    }
    RK_U64 alignedSize = 0;
    if (!align_u64(bytes, 64, &alignedSize) || alignedSize > UINT32_MAX)
        return RK_FAILURE;
    calculation->u32VirWidth = virtualWidth;
    calculation->u32VirHeight = virtualHeight;
    calculation->u32MBSize = static_cast<RK_U32>(alignedSize);
    return RK_SUCCESS;
}

struct CpuSurface {
    RK_U8 *data{nullptr};
    RK_U32 width{0};
    RK_U32 height{0};
    RK_U32 stride{0};
    RK_U32 bytesPerPixel{0};
    PIXEL_FORMAT_E format{RK_FMT_BUTT};
    MB_BLK block{MB_INVALID_HANDLE};
};

RK_U32 cpu_bytes_per_pixel(PIXEL_FORMAT_E format) {
    switch (format) {
    case RK_FMT_YUV400SP:
        return 1;
    case RK_FMT_RGB888:
    case RK_FMT_BGR888:
        return 3;
    case RK_FMT_ARGB8888:
    case RK_FMT_ABGR8888:
    case RK_FMT_BGRA8888:
    case RK_FMT_RGBA8888:
    case RK_FMT_XBGR8888:
        return 4;
    default:
        return 0;
    }
}

bool cpu_surface_from_frame(const VIDEO_FRAME_INFO_S &frame, CpuSurface &surface) {
    const VIDEO_FRAME_S &video = frame.stVFrame;
    const RK_U32 bytesPerPixel = cpu_bytes_per_pixel(video.enPixelFormat);
    if (video.pMbBlk == MB_INVALID_HANDLE || bytesPerPixel == 0 || video.u32Width == 0 ||
        video.u32Height == 0 || video.enCompressMode != COMPRESS_MODE_NONE) {
        return false;
    }
    RK_U8 *data = static_cast<RK_U8 *>(RK_MPI_MB_Handle2VirAddr(video.pMbBlk));
    const RK_U64 size = RK_MPI_MB_GetSize(video.pMbBlk);
    const RK_U32 virtualWidth = video.u32VirWidth ? video.u32VirWidth : video.u32Width;
    const RK_U32 stride = virtualWidth * bytesPerPixel;
    const RK_U32 virtualHeight = video.u32VirHeight ? video.u32VirHeight : video.u32Height;
    if (!data || virtualWidth < video.u32Width || virtualHeight < video.u32Height ||
        static_cast<RK_U64>(stride) * virtualHeight > size) {
        return false;
    }
    surface = {data,
               video.u32Width,
               video.u32Height,
               stride,
               bytesPerPixel,
               video.enPixelFormat,
               video.pMbBlk};
    return true;
}

bool cpu_surface_from_tde(const TDE_SURFACE_S &input, CpuSurface &surface) {
    const RK_U32 bytesPerPixel = cpu_bytes_per_pixel(input.enColorFmt);
    if (input.pMbBlk == MB_INVALID_HANDLE || bytesPerPixel == 0 || input.u32Width == 0 ||
        input.u32Height == 0 || input.enComprocessMode != COMPRESS_MODE_NONE) {
        return false;
    }
    RK_U8 *data = static_cast<RK_U8 *>(RK_MPI_MB_Handle2VirAddr(input.pMbBlk));
    const RK_U32 stride = input.u32Width * bytesPerPixel;
    if (!data || static_cast<RK_U64>(stride) * input.u32Height > RK_MPI_MB_GetSize(input.pMbBlk)) {
        return false;
    }
    surface = {data,
               input.u32Width,
               input.u32Height,
               stride,
               bytesPerPixel,
               input.enColorFmt,
               input.pMbBlk};
    return true;
}

bool cpu_rect_valid(const CpuSurface &surface, RK_S32 x, RK_S32 y, RK_U32 width, RK_U32 height) {
    if (x < 0 || y < 0 || width == 0 || height == 0) {
        return false;
    }
    return static_cast<RK_U64>(static_cast<RK_U32>(x)) + width <= surface.width &&
           static_cast<RK_U64>(static_cast<RK_U32>(y)) + height <= surface.height;
}

void cpu_store_color(RK_U8 *pixel, PIXEL_FORMAT_E format, RK_U32 color) {
    const RK_U8 red = static_cast<RK_U8>((color >> 16) & 0xffU);
    const RK_U8 green = static_cast<RK_U8>((color >> 8) & 0xffU);
    const RK_U8 blue = static_cast<RK_U8>(color & 0xffU);
    switch (format) {
    case RK_FMT_YUV400SP:
        pixel[0] =
            static_cast<RK_U8>((static_cast<RK_U32>(red) * 77U + static_cast<RK_U32>(green) * 150U +
                                static_cast<RK_U32>(blue) * 29U + 128U) >>
                               8);
        break;
    case RK_FMT_RGB888:
        pixel[0] = red;
        pixel[1] = green;
        pixel[2] = blue;
        break;
    case RK_FMT_BGR888:
        pixel[0] = blue;
        pixel[1] = green;
        pixel[2] = red;
        break;
    case RK_FMT_RGBA8888:
        pixel[0] = red;
        pixel[1] = green;
        pixel[2] = blue;
        pixel[3] = 255;
        break;
    case RK_FMT_BGRA8888:
        pixel[0] = blue;
        pixel[1] = green;
        pixel[2] = red;
        pixel[3] = 255;
        break;
    case RK_FMT_ARGB8888:
        pixel[0] = 255;
        pixel[1] = red;
        pixel[2] = green;
        pixel[3] = blue;
        break;
    case RK_FMT_ABGR8888:
    case RK_FMT_XBGR8888:
        pixel[0] = 255;
        pixel[1] = blue;
        pixel[2] = green;
        pixel[3] = red;
        break;
    default:
        break;
    }
}

RK_S32 cpu_copy_scale(const CpuSurface &source,
                      RK_S32 sourceX,
                      RK_S32 sourceY,
                      RK_U32 sourceWidth,
                      RK_U32 sourceHeight,
                      CpuSurface &destination,
                      RK_S32 destinationX,
                      RK_S32 destinationY,
                      RK_U32 destinationWidth,
                      RK_U32 destinationHeight) {
    if (source.format != destination.format || source.bytesPerPixel != destination.bytesPerPixel ||
        !cpu_rect_valid(source, sourceX, sourceY, sourceWidth, sourceHeight) ||
        !cpu_rect_valid(
            destination, destinationX, destinationY, destinationWidth, destinationHeight)) {
        return RK_FAILURE;
    }
    const RK_U32 bpp = source.bytesPerPixel;
    std::vector<RK_U8> snapshot(static_cast<std::size_t>(sourceWidth) * sourceHeight * bpp);
    for (RK_U32 y = 0; y < sourceHeight; ++y) {
        const RK_U8 *row = source.data + (static_cast<RK_U32>(sourceY) + y) * source.stride +
                           static_cast<RK_U32>(sourceX) * bpp;
        std::memcpy(snapshot.data() + static_cast<std::size_t>(y) * sourceWidth * bpp,
                    row,
                    static_cast<std::size_t>(sourceWidth) * bpp);
    }
    for (RK_U32 y = 0; y < destinationHeight; ++y) {
        const RK_U32 sourceRow =
            static_cast<RK_U32>((static_cast<RK_U64>(y) * sourceHeight) / destinationHeight);
        RK_U8 *destinationRow = destination.data +
                                (static_cast<RK_U32>(destinationY) + y) * destination.stride +
                                static_cast<RK_U32>(destinationX) * bpp;
        for (RK_U32 x = 0; x < destinationWidth; ++x) {
            const RK_U32 sourceColumn =
                static_cast<RK_U32>((static_cast<RK_U64>(x) * sourceWidth) / destinationWidth);
            const RK_U8 *pixel =
                snapshot.data() +
                (static_cast<std::size_t>(sourceRow) * sourceWidth + sourceColumn) * bpp;
            std::memcpy(destinationRow + static_cast<std::size_t>(x) * bpp, pixel, bpp);
        }
    }
    return RK_SUCCESS;
}

RK_S32 cpu_rotate(const CpuSurface &source, CpuSurface &destination, ROTATION_E rotation) {
    if (source.format != destination.format || source.bytesPerPixel != destination.bytesPerPixel ||
        rotation < ROTATION_0 || rotation >= ROTATION_BUTT) {
        return RK_FAILURE;
    }
    const bool swapsDimensions = rotation == ROTATION_90 || rotation == ROTATION_270;
    if ((swapsDimensions &&
         (destination.width != source.height || destination.height != source.width)) ||
        (!swapsDimensions &&
         (destination.width != source.width || destination.height != source.height))) {
        return RK_FAILURE;
    }
    const RK_U32 bpp = source.bytesPerPixel;
    std::vector<RK_U8> snapshot(static_cast<std::size_t>(source.width) * source.height * bpp);
    for (RK_U32 y = 0; y < source.height; ++y) {
        std::memcpy(snapshot.data() + static_cast<std::size_t>(y) * source.width * bpp,
                    source.data + y * source.stride,
                    static_cast<std::size_t>(source.width) * bpp);
    }
    for (RK_U32 y = 0; y < destination.height; ++y) {
        for (RK_U32 x = 0; x < destination.width; ++x) {
            RK_U32 sourceX = x;
            RK_U32 sourceY = y;
            switch (rotation) {
            case ROTATION_0:
                break;
            case ROTATION_90:
                sourceX = y;
                sourceY = source.height - 1U - x;
                break;
            case ROTATION_180:
                sourceX = source.width - 1U - x;
                sourceY = source.height - 1U - y;
                break;
            case ROTATION_270:
                sourceX = source.width - 1U - y;
                sourceY = x;
                break;
            default:
                return RK_FAILURE;
            }
            const RK_U8 *pixel = snapshot.data() +
                                 (static_cast<std::size_t>(sourceY) * source.width + sourceX) * bpp;
            std::memcpy(destination.data + y * destination.stride +
                            static_cast<std::size_t>(x) * bpp,
                        pixel,
                        bpp);
        }
    }
    return RK_SUCCESS;
}

RK_S32
cpu_fill(CpuSurface &surface, RK_S32 x, RK_S32 y, RK_U32 width, RK_U32 height, RK_U32 color) {
    if (!cpu_rect_valid(surface, x, y, width, height)) {
        return RK_FAILURE;
    }
    for (RK_U32 row = 0; row < height; ++row) {
        for (RK_U32 column = 0; column < width; ++column) {
            RK_U8 *pixel = surface.data + (static_cast<RK_U32>(y) + row) * surface.stride +
                           (static_cast<RK_U32>(x) + column) * surface.bytesPerPixel;
            cpu_store_color(pixel, surface.format, color);
        }
    }
    return RK_SUCCESS;
}

RK_S32 cpu_draw_line(CpuSurface &surface, const VGS_DRAW_LINE_S &line) {
    RK_S32 x0 = line.stStartPoint.s32X;
    RK_S32 y0 = line.stStartPoint.s32Y;
    const RK_S32 x1 = line.stEndPoint.s32X;
    const RK_S32 y1 = line.stEndPoint.s32Y;
    const RK_S32 dx = std::abs(x1 - x0);
    const RK_S32 sx = x0 < x1 ? 1 : -1;
    const RK_S32 dy = -std::abs(y1 - y0);
    const RK_S32 sy = y0 < y1 ? 1 : -1;
    RK_S32 error = dx + dy;
    const RK_U32 thickness = std::max<RK_U32>(line.u32Thick, 1U);
    for (;;) {
        const RK_S32 radius = static_cast<RK_S32>(thickness / 2U);
        for (RK_S32 yy = y0 - radius; yy <= y0 + radius; ++yy) {
            for (RK_S32 xx = x0 - radius; xx <= x0 + radius; ++xx) {
                if (xx >= 0 && yy >= 0 && static_cast<RK_U32>(xx) < surface.width &&
                    static_cast<RK_U32>(yy) < surface.height) {
                    cpu_store_color(surface.data + static_cast<RK_U32>(yy) * surface.stride +
                                        static_cast<RK_U32>(xx) * surface.bytesPerPixel,
                                    surface.format,
                                    line.u32Color);
                }
            }
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const RK_S32 twiceError = 2 * error;
        if (twiceError >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twiceError <= dx) {
            error += dx;
            y0 += sy;
        }
    }
    return RK_SUCCESS;
}

RK_S32 cpu_mosaic(CpuSurface &surface, const VGS_MOSAIC_S &mosaic) {
    const RK_U32 blockSize = static_cast<RK_U32>(mosaic.enBlkSize);
    const RECT_S &rect = mosaic.stDstRect;
    if ((blockSize != 8U && blockSize != 16U && blockSize != 32U && blockSize != 64U) ||
        !cpu_rect_valid(surface, rect.s32X, rect.s32Y, rect.u32Width, rect.u32Height)) {
        return RK_FAILURE;
    }
    const RK_U32 bpp = surface.bytesPerPixel;
    for (RK_U32 by = 0; by < rect.u32Height; by += blockSize) {
        for (RK_U32 bx = 0; bx < rect.u32Width; bx += blockSize) {
            const RK_U32 blockWidth = std::min(blockSize, rect.u32Width - bx);
            const RK_U32 blockHeight = std::min(blockSize, rect.u32Height - by);
            std::array<RK_U64, 4> sums{};
            for (RK_U32 y = 0; y < blockHeight; ++y) {
                for (RK_U32 x = 0; x < blockWidth; ++x) {
                    const RK_U8 *pixel =
                        surface.data + (static_cast<RK_U32>(rect.s32Y) + by + y) * surface.stride +
                        (static_cast<RK_U32>(rect.s32X) + bx + x) * bpp;
                    for (RK_U32 channel = 0; channel < bpp; ++channel)
                        sums[channel] += pixel[channel];
                }
            }
            const RK_U64 count = static_cast<RK_U64>(blockWidth) * blockHeight;
            for (RK_U32 y = 0; y < blockHeight; ++y) {
                for (RK_U32 x = 0; x < blockWidth; ++x) {
                    RK_U8 *pixel = surface.data +
                                   (static_cast<RK_U32>(rect.s32Y) + by + y) * surface.stride +
                                   (static_cast<RK_U32>(rect.s32X) + bx + x) * bpp;
                    for (RK_U32 channel = 0; channel < bpp; ++channel) {
                        pixel[channel] = static_cast<RK_U8>((sums[channel] + count / 2U) / count);
                    }
                }
            }
        }
    }
    return RK_SUCCESS;
}

RK_S32
prepare_vgs_surface(const VGS_TASK_ATTR_S *task, CpuSurface &source, CpuSurface &destination) {
    if (!task || !cpu_surface_from_frame(task->stImgIn, source) ||
        !cpu_surface_from_frame(task->stImgOut, destination)) {
        return RK_ERR_VGS_ILLEGAL_PARAM;
    }
    return RK_SUCCESS;
}

struct JobRegistry {
    std::mutex mutex;
    RK_S32 next{1};
    std::unordered_map<RK_S32, RK_U32> jobs;
};
JobRegistry &vgs_jobs() {
    static JobRegistry s;
    return s;
}
JobRegistry &gdc_jobs() {
    static JobRegistry s;
    return s;
}
JobRegistry &tde_jobs() {
    static JobRegistry s;
    return s;
}
RK_S32 begin_job(JobRegistry &s, RK_S32 *out, RK_S32 nullErr) {
    if (!out)
        return nullErr;
    std::lock_guard<std::mutex> l(s.mutex);
    const RK_S32 h = s.next++;
    s.jobs[h] = 0;
    *out = h;
    return RK_SUCCESS;
}
RK_S32 finish_job(JobRegistry &s, RK_S32 h, RK_S32 invalid) {
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.jobs.find(h);
    if (i == s.jobs.end())
        return invalid;
    s.jobs.erase(i);
    return RK_SUCCESS;
}
RK_S32 add_task(JobRegistry &s, RK_S32 h, const void *p, RK_S32 nullErr, RK_S32 invalid) {
    if (!p)
        return nullErr;
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.jobs.find(h);
    if (i == s.jobs.end())
        return invalid;
    ++i->second;
    return RK_SUCCESS;
}
RK_S32
record_vgs_operation(VGS_HANDLE handle, const VGS_TASK_ATTR_S *task, RK_S32 operationResult) {
    if (operationResult != RK_SUCCESS)
        return operationResult;
    return add_task(vgs_jobs(), handle, task, RK_ERR_VGS_NULL_PTR, RK_ERR_VGS_ILLEGAL_PARAM);
}

bool generic_vgs_pair(const VGS_TASK_ATTR_S &task) {
    return rockit_local::cpu_frame_is_supported(task.stImgIn) &&
           rockit_local::cpu_frame_is_supported(task.stImgOut) &&
           task.stImgIn.stVFrame.enPixelFormat == task.stImgOut.stVFrame.enPixelFormat;
}

RK_S32 commit_vgs_frame(VGS_HANDLE handle,
                        const VGS_TASK_ATTR_S *task,
                        VIDEO_FRAME_INFO_S *generated,
                        RK_S32 operationResult) {
    if (operationResult == RK_SUCCESS) {
        operationResult = rockit_local::cpu_copy_frame_to_existing(
            *generated, const_cast<VIDEO_FRAME_INFO_S *>(&task->stImgOut));
    }
    if (generated && generated->stVFrame.pMbBlk != MB_INVALID_HANDLE) {
        (void)RK_MPI_MB_ReleaseMB(generated->stVFrame.pMbBlk);
        generated->stVFrame.pMbBlk = MB_INVALID_HANDLE;
    }
    return record_vgs_operation(
        handle, task, operationResult == RK_SUCCESS ? RK_SUCCESS : RK_ERR_VGS_ILLEGAL_PARAM);
}

RK_S32 create_vgs_scaled_frame(const VGS_TASK_ATTR_S &task, VIDEO_FRAME_INFO_S *generated) {
    rockit_local::FrameTransform transform{};
    transform.outputWidth = task.stImgOut.stVFrame.u32Width;
    transform.outputHeight = task.stImgOut.stVFrame.u32Height;
    return rockit_local::cpu_transform_frame(task.stImgIn, transform, nullptr, generated);
}

VIDEO_FRAME_INFO_S tde_frame(const TDE_SURFACE_S &surface) {
    VIDEO_FRAME_INFO_S frame{};
    frame.stVFrame.pMbBlk = surface.pMbBlk;
    frame.stVFrame.u32Width = surface.u32Width;
    frame.stVFrame.u32Height = surface.u32Height;
    frame.stVFrame.u32VirWidth = surface.u32Width;
    frame.stVFrame.u32VirHeight = surface.u32Height;
    frame.stVFrame.enPixelFormat = surface.enColorFmt;
    frame.stVFrame.enCompressMode = surface.enComprocessMode;
    return frame;
}

RECT_S tde_rect(const TDE_RECT_S &rect) {
    return {rect.s32Xpos, rect.s32Ypos, rect.u32Width, rect.u32Height};
}
bool retain_mb(MB_BLK b) {
    return b != MB_INVALID_HANDLE && RK_MPI_MB_AddUserCnt(b) > 0;
}
void release_mb(MB_BLK b) {
    if (b != MB_INVALID_HANDLE)
        (void)RK_MPI_MB_ReleaseMB(b);
}

template <class Pred>
bool wait_for_item(std::condition_variable &cv,
                   std::unique_lock<std::mutex> &l,
                   RK_S32 ms,
                   Pred p) {
    if (p())
        return true;
    if (ms == 0 || ms < -1)
        return false;
    if (ms == -1) {
        cv.wait(l, p);
        return p();
    }
    return cv.wait_for(l, std::chrono::milliseconds(ms), p);
}
struct VideoItem {
    VIDEO_FRAME_INFO_S frame{};
};
void clear_q(std::deque<VideoItem> &q) {
    for (auto &i : q)
        release_mb(i.frame.stVFrame.pMbBlk);
    q.clear();
}

struct RgnAttachKey {
    RK_S32 mod{}, dev{}, chn{};
    bool operator==(const RgnAttachKey &o) const {
        return mod == o.mod && dev == o.dev && chn == o.chn;
    }
};
struct RgnAttachHash {
    std::size_t operator()(const RgnAttachKey &key) const {
        return rockit_local::packed_triplet_hash<std::size_t>(key.mod, key.dev, key.chn);
    }
};
struct RgnEntry {
    RGN_ATTR_S attr{};
    MB_BLK canvas{MB_INVALID_HANDLE};
    std::unordered_map<RgnAttachKey, RGN_CHN_ATTR_S, RgnAttachHash> attachments;
};
struct RgnState {
    std::mutex mutex;
    std::unordered_map<RGN_HANDLE, RgnEntry> entries;
};
RgnState &rgn_state() {
    static RgnState s;
    return s;
}
RK_U32 rgn_canvas_size(const RGN_ATTR_S &a) {
    if (a.enType != OVERLAY_RGN && a.enType != OVERLAY_EX_RGN)
        return 4096;
    PIC_BUF_ATTR_S p{};
    p.u32Width = a.unAttr.stOverlay.stSize.u32Width;
    p.u32Height = a.unAttr.stOverlay.stSize.u32Height;
    p.enPixelFormat = a.unAttr.stOverlay.enPixelFmt;
    MB_PIC_CAL_S c{};
    return calc_pic(&p, &c) == RK_SUCCESS ? std::max<RK_U32>(c.u32MBSize, 1) : 4096;
}
RgnAttachKey key_of(const MPP_CHN_S &c) {
    return {static_cast<RK_S32>(c.enModId), c.s32DevId, c.s32ChnId};
}

struct IvsResultData {
    RK_U32 frameId{0};
    std::vector<RECT_S> rectangles;
    std::vector<RK_CHAR> map;
    RK_U32 square{0};
    RK_U32 odFlag{0};
    RK_U32 pixelSum{0};
};
struct IvsChannel {
    IVS_CHN_ATTR_S attr{};
    IVS_MD_ATTR_S md{};
    std::deque<IvsResultData> results;
    std::vector<RK_U8> previousLuma;
    RK_U32 previousWidth{0};
    RK_U32 previousHeight{0};
    RK_U32 seq{0};
};
struct IvsState {
    std::mutex mutex;
    std::condition_variable cv;
    bool initialized{false};
    std::unordered_map<IVS_CHN, IvsChannel> channels;
};
IvsState &ivs_state() {
    static IvsState s;
    return s;
}

struct AvsChannel {
    bool configured{false}, enabled{false};
    AVS_CHN_ATTR_S attr{};
    std::deque<VideoItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
};
struct AvsGroup {
    bool started{false};
    AVS_GRP_ATTR_S attr{};
    std::array<std::deque<VideoItem>, AVS_PIPE_NUM> pipes;
    std::array<std::unordered_map<MB_BLK, RK_U32>, AVS_PIPE_NUM> pipeOutstanding;
    std::array<VideoItem, AVS_PIPE_NUM> latest{};
    std::array<bool, AVS_PIPE_NUM> latestValid{};
    std::array<AvsChannel, AVS_MAX_CHN_NUM> channels;
};
struct AvsState {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<AVS_GRP, AvsGroup> groups;
    AVS_MOD_PARAM_S mod{};
};
AvsState &avs_state() {
    static AvsState s;
    return s;
}
void clear_avs(AvsGroup &g) {
    for (auto &q : g.pipes)
        clear_q(q);
    for (auto &m : g.pipeOutstanding) {
        for (auto &[b, n] : m)
            for (RK_U32 i = 0; i < n; ++i)
                release_mb(b);
        m.clear();
    }
    for (std::size_t i = 0; i < g.latest.size(); ++i) {
        if (g.latestValid[i])
            release_mb(g.latest[i].frame.stVFrame.pMbBlk);
        g.latestValid[i] = false;
        g.latest[i] = VideoItem{};
    }
    for (auto &c : g.channels) {
        clear_q(c.ready);
        for (auto &[b, n] : c.outstanding)
            for (RK_U32 i = 0; i < n; ++i)
                release_mb(b);
        c.outstanding.clear();
    }
}

struct PvsChannel {
    bool configured{false}, enabled{false};
    PVS_CHN_ATTR_S attr{};
    PVS_CHN_PARAM_S param{};
};
struct PvsDevice {
    bool configured{false}, enabled{false};
    PVS_DEV_ATTR_S attr{};
    VIDEO_PROC_DEV_TYPE_E proc{VIDEO_PROC_DEV_RGA};
    std::array<PvsChannel, PVS_MAX_CHN_NUM> channels;
    std::array<VideoItem, PVS_MAX_CHN_NUM> latest{};
    std::array<bool, PVS_MAX_CHN_NUM> latestValid{};
    std::deque<VideoItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
};
struct PvsState {
    std::mutex mutex;
    std::condition_variable cv;
    std::array<PvsDevice, PVS_MAX_DEV_NUM> devs;
};
PvsState &pvs_state() {
    static PvsState s;
    return s;
}

std::size_t avs_channel_capacity(const AvsChannel &channel) {
    if (channel.attr.u32Depth)
        return channel.attr.u32Depth;
    if (channel.attr.u32FrameBufCnt)
        return channel.attr.u32FrameBufCnt;
    return 4;
}

bool build_avs_output(const AvsGroup &group,
                      const AvsChannel &channel,
                      VIDEO_FRAME_INFO_S *output) {
    if (!output)
        return false;
    const RK_U32 pipeCount = std::clamp<RK_U32>(group.attr.u32PipeNum, 1, AVS_PIPE_NUM);
    std::vector<VIDEO_FRAME_INFO_S> inputs;
    inputs.reserve(pipeCount);
    for (RK_U32 pipe = 0; pipe < pipeCount; ++pipe) {
        if (!group.latestValid[pipe])
            return false;
        inputs.push_back(group.latest[pipe].frame);
    }
    const RK_U32 outputWidth = channel.attr.u32Width ? channel.attr.u32Width
                                                     : (group.attr.stOutAttr.stSize.u32Width
                                                            ? group.attr.stOutAttr.stSize.u32Width
                                                            : inputs.front().stVFrame.u32Width);
    const RK_U32 outputHeight =
        channel.attr.u32Height
            ? channel.attr.u32Height
            : (group.attr.stOutAttr.stSize.u32Height ? group.attr.stOutAttr.stSize.u32Height
                                                     : inputs.front().stVFrame.u32Height);
    if (outputWidth == 0 || outputHeight == 0)
        return false;
    std::vector<RECT_S> destinations;
    destinations.reserve(pipeCount);
    bool averageOverlap = false;
    for (RK_U32 pipe = 0; pipe < pipeCount; ++pipe) {
        RECT_S rectangle{};
        switch (group.attr.enMode) {
        case AVS_MODE_BLEND:
            rectangle.u32Width = outputWidth;
            rectangle.u32Height = outputHeight;
            averageOverlap = true;
            break;
        case AVS_MODE_NOBLEND_VER: {
            const RK_U32 begin = (outputHeight * pipe) / pipeCount;
            const RK_U32 end = (outputHeight * (pipe + 1U)) / pipeCount;
            rectangle.s32Y = static_cast<RK_S32>(begin);
            rectangle.u32Width = outputWidth;
            rectangle.u32Height = end - begin;
            break;
        }
        case AVS_MODE_NOBLEND_QR: {
            const RK_U32 columns = 2;
            const RK_U32 rows = 2;
            const RK_U32 column = pipe % columns;
            const RK_U32 row = pipe / columns;
            if (row >= rows)
                return false;
            const RK_U32 left = (outputWidth * column) / columns;
            const RK_U32 right = (outputWidth * (column + 1U)) / columns;
            const RK_U32 top = (outputHeight * row) / rows;
            const RK_U32 bottom = (outputHeight * (row + 1U)) / rows;
            rectangle.s32X = static_cast<RK_S32>(left);
            rectangle.s32Y = static_cast<RK_S32>(top);
            rectangle.u32Width = right - left;
            rectangle.u32Height = bottom - top;
            break;
        }
        case AVS_MODE_NOBLEND_HOR:
        default: {
            const RK_U32 begin = (outputWidth * pipe) / pipeCount;
            const RK_U32 end = (outputWidth * (pipe + 1U)) / pipeCount;
            rectangle.s32X = static_cast<RK_S32>(begin);
            rectangle.u32Width = end - begin;
            rectangle.u32Height = outputHeight;
            break;
        }
        }
        destinations.push_back(rectangle);
    }
    VIDEO_FRAME_INFO_S metadata = inputs.front();
    metadata.stVFrame.u32Width = outputWidth;
    metadata.stVFrame.u32Height = outputHeight;
    metadata.stVFrame.u32VirWidth = outputWidth;
    metadata.stVFrame.u32VirHeight = outputHeight;
    metadata.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    for (const auto &input : inputs)
        metadata.stVFrame.u64PTS = std::max(metadata.stVFrame.u64PTS, input.stVFrame.u64PTS);
    return rockit_local::cpu_compose_frames(
               inputs, destinations, metadata, averageOverlap, output) == RK_SUCCESS;
}

bool build_pvs_output(const PvsDevice &device, VIDEO_FRAME_INFO_S *output) {
    if (!output || device.attr.stSize.u32Width == 0 || device.attr.stSize.u32Height == 0)
        return false;
    std::vector<VIDEO_FRAME_INFO_S> inputs;
    std::vector<RECT_S> destinations;
    for (std::size_t channel = 0; channel < device.channels.size(); ++channel) {
        if (!device.channels[channel].enabled || !device.latestValid[channel])
            continue;
        const auto &pvsRect = device.channels[channel].attr.stRect;
        RECT_S rectangle{};
        rectangle.s32X = pvsRect.s32X;
        rectangle.s32Y = pvsRect.s32Y;
        rectangle.u32Width = pvsRect.u32Width;
        rectangle.u32Height = pvsRect.u32Height;
        inputs.push_back(device.latest[channel].frame);
        destinations.push_back(rectangle);
    }
    if (inputs.empty())
        return false;
    VIDEO_FRAME_INFO_S metadata = inputs.front();
    metadata.stVFrame.u32Width = device.attr.stSize.u32Width;
    metadata.stVFrame.u32Height = device.attr.stSize.u32Height;
    metadata.stVFrame.u32VirWidth = device.attr.stSize.u32Width;
    metadata.stVFrame.u32VirHeight = device.attr.stSize.u32Height;
    metadata.stVFrame.enPixelFormat = device.attr.enPixelFormat;
    metadata.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    for (const auto &input : inputs)
        metadata.stVFrame.u64PTS = std::max(metadata.stVFrame.u64PTS, input.stVFrame.u64PTS);
    return rockit_local::cpu_compose_frames(inputs, destinations, metadata, false, output) ==
           RK_SUCCESS;
}
} // namespace

#define ROCKIT_RETURN_VGS_TRANSFORM(task_, transform_)                                           \
    VIDEO_FRAME_INFO_S generated{};                                                               \
    return commit_vgs_frame(                                                                       \
        handle,                                                                                    \
        (task_),                                                                                   \
        &generated,                                                                                \
        rockit_local::cpu_transform_frame((task_)->stImgIn, (transform_), nullptr, &generated))

#define ROCKIT_RETURN_VGS_GENERIC_FRAME_TASK(task_, operation_)                                   \
    if (cpu_bytes_per_pixel((task_)->stImgIn.stVFrame.enPixelFormat) == 0 ||                       \
        cpu_bytes_per_pixel((task_)->stImgOut.stVFrame.enPixelFormat) == 0) {                      \
        if (!generic_vgs_pair(*(task_)))                                                           \
            return RK_ERR_VGS_NOT_SUPPORT;                                                        \
        VIDEO_FRAME_INFO_S generated{};                                                           \
        RK_S32 result = create_vgs_scaled_frame(*(task_), &generated);                            \
        if (result == RK_SUCCESS)                                                                  \
            result = operation_;                                                                  \
        return commit_vgs_frame(handle, (task_), &generated, result);                             \
    }

#define ROCKIT_RETURN_VGS_FULL_FRAME_TASK(task_, operation_, failure_)                             \
    CpuSurface source{}, destination{};                                                            \
    RK_S32 result = prepare_vgs_surface(task_, source, destination);                               \
    if (result == RK_SUCCESS) {                                                                    \
        result = cpu_copy_scale(source,                                                            \
                                0,                                                                 \
                                0,                                                                 \
                                source.width,                                                      \
                                source.height,                                                     \
                                destination,                                                       \
                                0,                                                                 \
                                0,                                                                 \
                                destination.width,                                                 \
                                destination.height);                                               \
        if (result == RK_SUCCESS)                                                                  \
            result = operation_;                                                                   \
        result = result == RK_SUCCESS ? RK_SUCCESS : failure_;                                     \
    }                                                                                              \
    return record_vgs_operation(handle, task_, result)

extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_BeginJob(VGS_HANDLE *handle) {
    return begin_job(vgs_jobs(), handle, RK_ERR_VGS_NULL_PTR);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_EndJob(VGS_HANDLE handle) {
    if (handle == static_cast<VGS_HANDLE>(-1))
        return RK_ERR_VGS_NULL_PTR;
    return finish_job(vgs_jobs(), handle, RK_ERR_VGS_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_CancelJob(VGS_HANDLE handle) {
    if (handle == static_cast<VGS_HANDLE>(-1))
        return RK_ERR_VGS_NULL_PTR;
    return finish_job(vgs_jobs(), handle, RK_ERR_VGS_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddScaleTask(VGS_HANDLE handle,
                                                     const VGS_TASK_ATTR_S *task,
                                                     VGS_SCLCOEF_MODE_E mode) {
    if (!task)
        return RK_ERR_VGS_NULL_PTR;
    if (mode < VGS_SCLCOEF_NORMAL || mode >= VGS_SCLCOEF_BUTT)
        return RK_ERR_VGS_ILLEGAL_PARAM;
    if (cpu_bytes_per_pixel(task->stImgIn.stVFrame.enPixelFormat) == 0 ||
        cpu_bytes_per_pixel(task->stImgOut.stVFrame.enPixelFormat) == 0) {
        if (!generic_vgs_pair(*task)) {
            return add_task(
                vgs_jobs(), handle, task, RK_ERR_VGS_NULL_PTR, RK_ERR_VGS_ILLEGAL_PARAM);
        }
        VIDEO_FRAME_INFO_S generated{};
        return commit_vgs_frame(
            handle, task, &generated, create_vgs_scaled_frame(*task, &generated));
    }
    ROCKIT_RETURN_VGS_FULL_FRAME_TASK(task, RK_SUCCESS, RK_ERR_VGS_NOT_SUPPORT);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddRotationTask(VGS_HANDLE handle,
                                                        const VGS_TASK_ATTR_S *task,
                                                        ROTATION_E rotation) {
    if (!task)
        return RK_ERR_VGS_NULL_PTR;
    if (rotation < ROTATION_0 || rotation >= ROTATION_BUTT)
        return RK_ERR_VGS_ILLEGAL_PARAM;
    if (cpu_bytes_per_pixel(task->stImgIn.stVFrame.enPixelFormat) == 0 ||
        cpu_bytes_per_pixel(task->stImgOut.stVFrame.enPixelFormat) == 0) {
        if (!generic_vgs_pair(*task))
            return RK_ERR_VGS_NOT_SUPPORT;
        rockit_local::FrameTransform transform{};
        transform.rotation = rotation;
        ROCKIT_RETURN_VGS_TRANSFORM(task, transform);
    }
    CpuSurface source{}, destination{};
    RK_S32 result = prepare_vgs_surface(task, source, destination);
    if (result == RK_SUCCESS) {
        result = cpu_rotate(source, destination, rotation) == RK_SUCCESS ? RK_SUCCESS
                                                                         : RK_ERR_VGS_NOT_SUPPORT;
    }
    return record_vgs_operation(handle, task, result);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddCropTask(VGS_HANDLE handle,
                                                    const VGS_TASK_ATTR_S *task,
                                                    const VGS_CROP_INFO_S *crop) {
    if (!task || !crop)
        return RK_ERR_VGS_NULL_PTR;
    if (crop->enCropCoordinate != VGS_CROP_ABS_COOR)
        return RK_ERR_VGS_NOT_SUPPORT;
    if (cpu_bytes_per_pixel(task->stImgIn.stVFrame.enPixelFormat) == 0 ||
        cpu_bytes_per_pixel(task->stImgOut.stVFrame.enPixelFormat) == 0) {
        if (!generic_vgs_pair(*task))
            return RK_ERR_VGS_NOT_SUPPORT;
        rockit_local::FrameTransform transform{};
        transform.cropEnabled = RK_TRUE;
        transform.crop = crop->stCropRect;
        transform.outputWidth = task->stImgOut.stVFrame.u32Width;
        transform.outputHeight = task->stImgOut.stVFrame.u32Height;
        ROCKIT_RETURN_VGS_TRANSFORM(task, transform);
    }
    CpuSurface source{}, destination{};
    RK_S32 result = prepare_vgs_surface(task, source, destination);
    if (result == RK_SUCCESS) {
        const RECT_S &rect = crop->stCropRect;
        result = cpu_copy_scale(source,
                                rect.s32X,
                                rect.s32Y,
                                rect.u32Width,
                                rect.u32Height,
                                destination,
                                0,
                                0,
                                destination.width,
                                destination.height) == RK_SUCCESS
                     ? RK_SUCCESS
                     : RK_ERR_VGS_ILLEGAL_PARAM;
    }
    return record_vgs_operation(handle, task, result);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddDrawLineTask(VGS_HANDLE handle,
                                                        const VGS_TASK_ATTR_S *task,
                                                        const VGS_DRAW_LINE_S *line) {
    if (!task || !line)
        return RK_ERR_VGS_NULL_PTR;
    ROCKIT_RETURN_VGS_GENERIC_FRAME_TASK(
        task,
        (rockit_local::cpu_draw_line_frame(&generated,
                                           line->stStartPoint,
                                           line->stEndPoint,
                                           std::max<RK_U32>(line->u32Thick, 1U),
                                           line->u32Color)));
    ROCKIT_RETURN_VGS_FULL_FRAME_TASK(
        task, (cpu_draw_line(destination, *line)), RK_ERR_VGS_NOT_SUPPORT);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddCoverTask(VGS_HANDLE handle,
                                                     const VGS_TASK_ATTR_S *task,
                                                     const VGS_ADD_COVER_S *cover) {
    if (!task || !cover)
        return RK_ERR_VGS_NULL_PTR;
    if (cover->enCoverType != COVER_RECT)
        return RK_ERR_VGS_NOT_SUPPORT;
    ROCKIT_RETURN_VGS_GENERIC_FRAME_TASK(
        task, (rockit_local::cpu_fill_frame(&generated, cover->stDstRect, cover->u32Color)));
    const RECT_S &rect = cover->stDstRect;
    ROCKIT_RETURN_VGS_FULL_FRAME_TASK(
        task,
        (cpu_fill(destination,
                  rect.s32X,
                  rect.s32Y,
                  rect.u32Width,
                  rect.u32Height,
                  cover->u32Color)),
        RK_ERR_VGS_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddMosaicTask(VGS_HANDLE handle,
                                                      const VGS_TASK_ATTR_S *task,
                                                      const VGS_MOSAIC_S *mosaic) {
    if (!task || !mosaic)
        return RK_ERR_VGS_NULL_PTR;
    ROCKIT_RETURN_VGS_GENERIC_FRAME_TASK(
        task,
        (rockit_local::cpu_mosaic_frame(
            &generated, mosaic->stDstRect, static_cast<RK_U32>(mosaic->enBlkSize))));
    ROCKIT_RETURN_VGS_FULL_FRAME_TASK(
        task, (cpu_mosaic(destination, *mosaic)), RK_ERR_VGS_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddOsdTask(VGS_HANDLE handle,
                                                   const VGS_TASK_ATTR_S *task,
                                                   const VGS_ADD_OSD_S *osd) {
    if (!task || !osd)
        return RK_ERR_VGS_NULL_PTR;
    if (!generic_vgs_pair(*task) || osd->pMbBlk == MB_INVALID_HANDLE || osd->stRect.u32Width == 0 ||
        osd->stRect.u32Height == 0) {
        return add_task(vgs_jobs(), handle, task, RK_ERR_VGS_NULL_PTR, RK_ERR_VGS_ILLEGAL_PARAM);
    }
    VIDEO_FRAME_INFO_S generated{};
    RK_S32 result = create_vgs_scaled_frame(*task, &generated);
    if (result == RK_SUCCESS) {
        const auto *source = static_cast<const RK_U8 *>(RK_MPI_MB_Handle2VirAddr(osd->pMbBlk));
        const RK_U64 sourceBytes = RK_MPI_MB_GetSize(osd->pMbBlk);
        const POINT_S point{osd->stRect.s32X, osd->stRect.s32Y};
        result = rockit_local::cpu_overlay_frame(&generated,
                                                 source,
                                                 sourceBytes,
                                                 osd->stRect.u32Width,
                                                 osd->stRect.u32Height,
                                                 osd->enPixelFmt,
                                                 point,
                                                 osd->u32FgAlpha);
    }
    return commit_vgs_frame(handle, task, &generated, result);
}
#undef ROCKIT_RETURN_VGS_FULL_FRAME_TASK
#undef ROCKIT_RETURN_VGS_GENERIC_FRAME_TASK
#undef ROCKIT_RETURN_VGS_TRANSFORM

namespace {
RK_S32 prepare_vgs_array_frame(const VGS_TASK_ATTR_S *task, VIDEO_FRAME_INFO_S *generated) {
    if (!task || !generated)
        return RK_ERR_VGS_NULL_PTR;
    if (!generic_vgs_pair(*task))
        return RK_ERR_VGS_NOT_SUPPORT;
    return create_vgs_scaled_frame(*task, generated);
}

RK_S32 finish_vgs_array(VGS_HANDLE handle,
                        const VGS_TASK_ATTR_S *task,
                        VIDEO_FRAME_INFO_S *generated,
                        RK_S32 result) {
    return commit_vgs_frame(handle, task, generated, result);
}
} // namespace

#define ROCKIT_VGS_ARRAY_PREAMBLE(handle_, task_, items_, count_)                                  \
    if (handle_ == static_cast<VGS_HANDLE>(-1))                                                    \
        return RK_ERR_VGS_ILLEGAL_PARAM;                                                           \
    if (!task_ || !items_)                                                                         \
        return RK_ERR_VGS_ILLEGAL_PARAM;                                                           \
    if (count_ == 0 || count_ > VGS_MAX_ARRAY_SIZE)                                                \
        return RK_ERR_VGS_ILLEGAL_PARAM;                                                           \
    VIDEO_FRAME_INFO_S generated{};                                                                \
    RK_S32 result = prepare_vgs_array_frame(task_, &generated)

extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddDrawLineTaskArray(VGS_HANDLE handle,
                                                             const VGS_TASK_ATTR_S *task,
                                                             const VGS_DRAW_LINE_S *items,
                                                             RK_U32 count) {
    ROCKIT_VGS_ARRAY_PREAMBLE(handle, task, items, count);
    for (RK_U32 index = 0; result == RK_SUCCESS && index < count; ++index) {
        result = rockit_local::cpu_draw_line_frame(&generated,
                                                   items[index].stStartPoint,
                                                   items[index].stEndPoint,
                                                   std::max<RK_U32>(items[index].u32Thick, 1U),
                                                   items[index].u32Color);
    }
    return finish_vgs_array(handle, task, &generated, result);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddCoverTaskArray(VGS_HANDLE handle,
                                                          const VGS_TASK_ATTR_S *task,
                                                          const VGS_ADD_COVER_S *items,
                                                          RK_U32 count) {
    ROCKIT_VGS_ARRAY_PREAMBLE(handle, task, items, count);
    for (RK_U32 index = 0; result == RK_SUCCESS && index < count; ++index) {
        if (items[index].enCoverType != COVER_RECT) {
            result = RK_ERR_VGS_NOT_SUPPORT;
            break;
        }
        result =
            rockit_local::cpu_fill_frame(&generated, items[index].stDstRect, items[index].u32Color);
    }
    return finish_vgs_array(handle, task, &generated, result);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddOsdTaskArray(VGS_HANDLE handle,
                                                        const VGS_TASK_ATTR_S *task,
                                                        const VGS_ADD_OSD_S *items,
                                                        RK_U32 count) {
    ROCKIT_VGS_ARRAY_PREAMBLE(handle, task, items, count);
    for (RK_U32 index = 0; result == RK_SUCCESS && index < count; ++index) {
        if (items[index].pMbBlk == MB_INVALID_HANDLE || items[index].stRect.u32Width == 0 ||
            items[index].stRect.u32Height == 0) {
            result = RK_ERR_VGS_ILLEGAL_PARAM;
            break;
        }
        const auto *source =
            static_cast<const RK_U8 *>(RK_MPI_MB_Handle2VirAddr(items[index].pMbBlk));
        const RK_U64 sourceBytes = RK_MPI_MB_GetSize(items[index].pMbBlk);
        const POINT_S point{items[index].stRect.s32X, items[index].stRect.s32Y};
        result = rockit_local::cpu_overlay_frame(&generated,
                                                 source,
                                                 sourceBytes,
                                                 items[index].stRect.u32Width,
                                                 items[index].stRect.u32Height,
                                                 items[index].enPixelFmt,
                                                 point,
                                                 items[index].u32FgAlpha);
    }
    return finish_vgs_array(handle, task, &generated, result);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_VGS_AddMosaicTaskArray(VGS_HANDLE handle,
                                                           const VGS_TASK_ATTR_S *task,
                                                           const VGS_MOSAIC_S *items,
                                                           RK_U32 count) {
    ROCKIT_VGS_ARRAY_PREAMBLE(handle, task, items, count);
    for (RK_U32 index = 0; result == RK_SUCCESS && index < count; ++index) {
        result = rockit_local::cpu_mosaic_frame(
            &generated, items[index].stDstRect, static_cast<RK_U32>(items[index].enBlkSize));
    }
    return finish_vgs_array(handle, task, &generated, result);
}

#undef ROCKIT_VGS_ARRAY_PREAMBLE

extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_BeginJob(GDC_HANDLE *h) {
    return begin_job(gdc_jobs(), h, RK_ERR_GDC_NULL_PTR);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_EndJob(GDC_HANDLE h) {
    if (h == static_cast<GDC_HANDLE>(-1))
        return RK_ERR_GDC_NULL_PTR;
    return finish_job(gdc_jobs(), h, RK_ERR_GDC_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_CancelJob(GDC_HANDLE h) {
    if (h == static_cast<GDC_HANDLE>(-1))
        return RK_ERR_GDC_NULL_PTR;
    return finish_job(gdc_jobs(), h, RK_ERR_GDC_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_StopJob(GDC_HANDLE h) {
    if (h == static_cast<GDC_HANDLE>(-1))
        return RK_ERR_GDC_NULL_PTR;
    std::lock_guard<std::mutex> l(gdc_jobs().mutex);
    return gdc_jobs().jobs.count(h) ? RK_SUCCESS : RK_ERR_GDC_ILLEGAL_PARAM;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_SetConfig(GDC_HANDLE, const FISHEYE_JOB_CONFIG_S *) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_AddCorrectionTask(GDC_HANDLE h,
                                                          const GDC_TASK_ATTR_S *t,
                                                          const FISHEYE_ATTR_S *p) {
    if (!p)
        return RK_ERR_GDC_NULL_PTR;
    return add_task(gdc_jobs(), h, t, RK_ERR_GDC_NULL_PTR, RK_ERR_GDC_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_AddCorrectionExTask(GDC_HANDLE h,
                                                            const GDC_TASK_ATTR_S *t,
                                                            const FISHEYE_ATTR_EX_S *p,
                                                            RK_BOOL) {
    if (!p)
        return RK_ERR_GDC_NULL_PTR;
    return add_task(gdc_jobs(), h, t, RK_ERR_GDC_NULL_PTR, RK_ERR_GDC_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_GDC_AddPMFTask(GDC_HANDLE h,
                                                   const GDC_TASK_ATTR_S *t,
                                                   const GDC_PMF_ATTR_S *p) {
    if (!p)
        return RK_ERR_GDC_NULL_PTR;
    return add_task(gdc_jobs(), h, t, RK_ERR_GDC_NULL_PTR, RK_ERR_GDC_ILLEGAL_PARAM);
}
extern "C" ROCKIT_API RK_S32
RK_MPI_GDC_FisheyePosQueryDst2Src(const GDC_FISHEYE_POINT_QUERY_ATTR_S *a,
                                  const VIDEO_FRAME_INFO_S *v,
                                  const POINT_S *d,
                                  POINT_S *s) {
    if (!a || !v || !d || !s)
        return RK_ERR_GDC_NULL_PTR;
    *s = *d;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32
RK_MPI_GDC_FisheyePosQueryDst2SrcArray(const GDC_FISHEYE_POINT_QUERY_ATTR_S *a,
                                       const VIDEO_FRAME_INFO_S *v,
                                       RK_U32 n,
                                       const POINT_S *d,
                                       POINT_S *s) {
    if (!a || !v || !d || !s)
        return RK_ERR_GDC_NULL_PTR;
    std::copy_n(d, n, s);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32
RK_MPI_GDC_FisheyePosQueryDst2Pano(const GDC_FISHEYE_POINT_QUERY_ATTR_S *a,
                                   const VIDEO_FRAME_INFO_S *v,
                                   RK_U32,
                                   const POINT_S *d,
                                   POINT_S *s) {
    if (!a || !v || !d || !s)
        return RK_ERR_GDC_NULL_PTR;
    *s = *d;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32
RK_MPI_GDC_FisheyePosQueryDst2PanoArray(const GDC_FISHEYE_POINT_QUERY_ATTR_S *a,
                                        const VIDEO_FRAME_INFO_S *v,
                                        RK_U32,
                                        RK_U32 n,
                                        const POINT_S *d,
                                        POINT_S *s) {
    if (!a || !v || !d || !s)
        return RK_ERR_GDC_NULL_PTR;
    std::copy_n(d, n, s);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_TDE_Open() {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API void RK_TDE_Close() {
}
extern "C" ROCKIT_API TDE_HANDLE RK_TDE_BeginJob() {
    RK_S32 handle = 0;
    return begin_job(tde_jobs(), &handle, RK_ERR_TDE_NULL_PTR) == RK_SUCCESS
               ? handle
               : RK_ERR_TDE_INVALID_HANDLE;
}
extern "C" ROCKIT_API RK_S32 RK_TDE_EndJob(TDE_HANDLE handle, RK_BOOL, RK_BOOL, RK_U32) {
    if (handle == static_cast<TDE_HANDLE>(-1))
        return RK_ERR_TDE_NULL_PTR;
    return finish_job(tde_jobs(), handle, RK_ERR_TDE_INVALID_HANDLE);
}
extern "C" ROCKIT_API RK_S32 RK_TDE_CancelJob(TDE_HANDLE handle) {
    if (handle == static_cast<TDE_HANDLE>(-1))
        return RK_ERR_TDE_NULL_PTR;
    return finish_job(tde_jobs(), handle, RK_ERR_TDE_INVALID_HANDLE);
}
extern "C" ROCKIT_API RK_S32 RK_TDE_WaitForDone(TDE_HANDLE) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_TDE_WaitAllDone() {
    return RK_SUCCESS;
}
static RK_S32 tde_task(TDE_HANDLE handle, const void *source, const void *destination) {
    if (!source || !destination)
        return RK_ERR_TDE_NULL_PTR;
    return add_task(tde_jobs(), handle, source, RK_ERR_TDE_NULL_PTR, RK_ERR_TDE_INVALID_HANDLE);
}
static RK_S32 record_tde_operation(TDE_HANDLE handle,
                                   const void *source,
                                   const void *destination,
                                   RK_S32 operationResult) {
    if (operationResult != RK_SUCCESS)
        return operationResult;
    return tde_task(handle, source, destination);
}
#define ROCKIT_TDE_REQUIRE_EQUAL_SIZE(sourceRect_, destinationRect_)                               \
    if ((sourceRect_)->u32Width != (destinationRect_)->u32Width ||                                 \
        (sourceRect_)->u32Height != (destinationRect_)->u32Height)                                 \
    return RK_ERR_TDE_INVALID_PARA

#define ROCKIT_TDE_ALLOW_RESIZE(sourceRect_, destinationRect_)

#define ROCKIT_TDE_QUICK_OPERATION(size_policy_)                                                   \
    if (!sourceSurface || !sourceRect || !destinationSurface || !destinationRect)                  \
        return RK_ERR_TDE_NULL_PTR;                                                                \
    if (cpu_bytes_per_pixel(sourceSurface->enColorFmt) == 0 ||                                     \
        cpu_bytes_per_pixel(destinationSurface->enColorFmt) == 0) {                                \
        VIDEO_FRAME_INFO_S sourceFrame = tde_frame(*sourceSurface);                                \
        VIDEO_FRAME_INFO_S destinationFrame = tde_frame(*destinationSurface);                      \
        if (!rockit_local::cpu_frame_is_supported(sourceFrame) ||                                  \
            !rockit_local::cpu_frame_is_supported(destinationFrame) ||                             \
            sourceSurface->enColorFmt != destinationSurface->enColorFmt) {                         \
            return tde_task(handle, sourceSurface, destinationSurface);                            \
        }                                                                                          \
        size_policy_(sourceRect, destinationRect);                                                 \
        const RK_S32 result = rockit_local::cpu_blit_frame(                                        \
            sourceFrame, tde_rect(*sourceRect), &destinationFrame, tde_rect(*destinationRect));    \
        return record_tde_operation(handle,                                                        \
                                    sourceSurface,                                                 \
                                    destinationSurface,                                            \
                                    result == RK_SUCCESS ? RK_SUCCESS : RK_ERR_TDE_INVALID_PARA);  \
    }                                                                                              \
    CpuSurface source{}, destination{};                                                            \
    if (!cpu_surface_from_tde(*sourceSurface, source) ||                                           \
        !cpu_surface_from_tde(*destinationSurface, destination))                                   \
        return RK_ERR_TDE_UNSUPPORTED_OPERATION;                                                   \
    size_policy_(sourceRect, destinationRect);                                                     \
    const RK_S32 result = cpu_copy_scale(source,                                                   \
                                         sourceRect->s32Xpos,                                      \
                                         sourceRect->s32Ypos,                                      \
                                         sourceRect->u32Width,                                     \
                                         sourceRect->u32Height,                                    \
                                         destination,                                              \
                                         destinationRect->s32Xpos,                                 \
                                         destinationRect->s32Ypos,                                 \
                                         destinationRect->u32Width,                                \
                                         destinationRect->u32Height);                              \
    return record_tde_operation(handle,                                                            \
                                sourceSurface,                                                     \
                                destinationSurface,                                                \
                                result == RK_SUCCESS ? RK_SUCCESS : RK_ERR_TDE_INVALID_PARA)

extern "C" ROCKIT_API RK_S32 RK_TDE_QuickCopy(TDE_HANDLE handle,
                                              const TDE_SURFACE_S *sourceSurface,
                                              const TDE_RECT_S *sourceRect,
                                              const TDE_SURFACE_S *destinationSurface,
                                              const TDE_RECT_S *destinationRect) {
    ROCKIT_TDE_QUICK_OPERATION(ROCKIT_TDE_REQUIRE_EQUAL_SIZE);
}
extern "C" ROCKIT_API RK_S32 RK_TDE_QuickResize(TDE_HANDLE handle,
                                                const TDE_SURFACE_S *sourceSurface,
                                                const TDE_RECT_S *sourceRect,
                                                const TDE_SURFACE_S *destinationSurface,
                                                const TDE_RECT_S *destinationRect) {
    ROCKIT_TDE_QUICK_OPERATION(ROCKIT_TDE_ALLOW_RESIZE);
}

#undef ROCKIT_TDE_QUICK_OPERATION
#undef ROCKIT_TDE_ALLOW_RESIZE
#undef ROCKIT_TDE_REQUIRE_EQUAL_SIZE

extern "C" ROCKIT_API RK_S32 RK_TDE_Bitblit(TDE_HANDLE handle,
                                            const TDE_SURFACE_S *background,
                                            const TDE_RECT_S *,
                                            const TDE_SURFACE_S *foreground,
                                            const TDE_RECT_S *foregroundRect,
                                            const TDE_SURFACE_S *destination,
                                            const TDE_RECT_S *destinationRect,
                                            const TDE_OPT_S *) {
    if (!foreground) {
        if (!background)
            return RK_ERR_TDE_NULL_PTR;
        return RK_ERR_TDE_UNSUPPORTED_OPERATION;
    }
    return RK_TDE_QuickResize(handle, foreground, foregroundRect, destination, destinationRect);
}
extern "C" ROCKIT_API RK_S32 RK_TDE_QuickFill(TDE_HANDLE handle,
                                              TDE_SURFACE_S *destinationSurface,
                                              TDE_RECT_S *destinationRect,
                                              RK_U32 color) {
    if (!destinationSurface || !destinationRect)
        return RK_ERR_TDE_NULL_PTR;
    if (cpu_bytes_per_pixel(destinationSurface->enColorFmt) == 0) {
        VIDEO_FRAME_INFO_S destinationFrame = tde_frame(*destinationSurface);
        if (!rockit_local::cpu_frame_is_supported(destinationFrame)) {
            return tde_task(handle, destinationSurface, destinationSurface);
        }
        const RK_S32 result =
            rockit_local::cpu_fill_frame(&destinationFrame, tde_rect(*destinationRect), color);
        return record_tde_operation(handle,
                                    destinationSurface,
                                    destinationSurface,
                                    result == RK_SUCCESS ? RK_SUCCESS : RK_ERR_TDE_INVALID_PARA);
    }
    CpuSurface destination{};
    if (!cpu_surface_from_tde(*destinationSurface, destination))
        return RK_ERR_TDE_UNSUPPORTED_OPERATION;
    const RK_S32 result = cpu_fill(destination,
                                   destinationRect->s32Xpos,
                                   destinationRect->s32Ypos,
                                   destinationRect->u32Width,
                                   destinationRect->u32Height,
                                   color);
    return record_tde_operation(handle,
                                destinationSurface,
                                destinationSurface,
                                result == RK_SUCCESS ? RK_SUCCESS : RK_ERR_TDE_INVALID_PARA);
}
extern "C" ROCKIT_API RK_S32 RK_TDE_Rotate(TDE_HANDLE handle,
                                           TDE_SURFACE_S *sourceSurface,
                                           TDE_RECT_S *sourceRect,
                                           TDE_SURFACE_S *destinationSurface,
                                           TDE_RECT_S *destinationRect,
                                           ROTATION_E rotation) {
    if (!sourceSurface || !sourceRect || !destinationSurface || !destinationRect)
        return RK_ERR_TDE_NULL_PTR;
    if (cpu_bytes_per_pixel(sourceSurface->enColorFmt) == 0 ||
        cpu_bytes_per_pixel(destinationSurface->enColorFmt) == 0) {
        VIDEO_FRAME_INFO_S sourceFrame = tde_frame(*sourceSurface);
        VIDEO_FRAME_INFO_S destinationFrame = tde_frame(*destinationSurface);
        if (!rockit_local::cpu_frame_is_supported(sourceFrame) ||
            !rockit_local::cpu_frame_is_supported(destinationFrame) ||
            sourceSurface->enColorFmt != destinationSurface->enColorFmt) {
            return tde_task(handle, sourceSurface, destinationSurface);
        }
        if (sourceRect->s32Xpos != 0 || sourceRect->s32Ypos != 0 || destinationRect->s32Xpos != 0 ||
            destinationRect->s32Ypos != 0 || sourceRect->u32Width != sourceSurface->u32Width ||
            sourceRect->u32Height != sourceSurface->u32Height ||
            destinationRect->u32Width != destinationSurface->u32Width ||
            destinationRect->u32Height != destinationSurface->u32Height) {
            return RK_ERR_TDE_UNSUPPORTED_OPERATION;
        }
        rockit_local::FrameTransform transform{};
        transform.rotation = rotation;
        VIDEO_FRAME_INFO_S generated{};
        RK_S32 result =
            rockit_local::cpu_transform_frame(sourceFrame, transform, nullptr, &generated);
        if (result == RK_SUCCESS) {
            result = rockit_local::cpu_copy_frame_to_existing(generated, &destinationFrame);
        }
        if (generated.stVFrame.pMbBlk != MB_INVALID_HANDLE) {
            (void)RK_MPI_MB_ReleaseMB(generated.stVFrame.pMbBlk);
        }
        return record_tde_operation(handle,
                                    sourceSurface,
                                    destinationSurface,
                                    result == RK_SUCCESS ? RK_SUCCESS : RK_ERR_TDE_INVALID_PARA);
    }
    if (sourceRect->s32Xpos != 0 || sourceRect->s32Ypos != 0 || destinationRect->s32Xpos != 0 ||
        destinationRect->s32Ypos != 0 || sourceRect->u32Width != sourceSurface->u32Width ||
        sourceRect->u32Height != sourceSurface->u32Height ||
        destinationRect->u32Width != destinationSurface->u32Width ||
        destinationRect->u32Height != destinationSurface->u32Height)
        return RK_ERR_TDE_UNSUPPORTED_OPERATION;
    CpuSurface source{}, destination{};
    if (!cpu_surface_from_tde(*sourceSurface, source) ||
        !cpu_surface_from_tde(*destinationSurface, destination))
        return RK_ERR_TDE_UNSUPPORTED_OPERATION;
    const RK_S32 result = cpu_rotate(source, destination, rotation);
    return record_tde_operation(handle,
                                sourceSurface,
                                destinationSurface,
                                result == RK_SUCCESS ? RK_SUCCESS : RK_ERR_TDE_INVALID_PARA);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_Create(RGN_HANDLE h, const RGN_ATTR_S *a) {
    if (h >= RGN_MAX_HANDLE_NUM)
        return RK_ERR_RGN_INVALID_DEVID;
    if (!a)
        return RK_ERR_RGN_NULL_PTR;
    if (h >= RGN_MAX_HANDLE_NUM)
        return RK_ERR_RGN_INVALID_CHNID;
    auto &s = rgn_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.entries.count(h))
        return RK_ERR_RGN_EXIST;
    RgnEntry e{};
    e.attr = *a;
    if (a->enType == OVERLAY_RGN || a->enType == OVERLAY_EX_RGN) {
        if (RK_MPI_SYS_MmzAlloc(&e.canvas, "rgn", nullptr, rgn_canvas_size(*a)) != RK_SUCCESS)
            return RK_ERR_RGN_NOMEM;
    }
    s.entries.emplace(h, std::move(e));
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_Destroy(RGN_HANDLE h) {
    if (h >= RGN_MAX_HANDLE_NUM)
        return RK_ERR_RGN_INVALID_DEVID;
    auto &s = rgn_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.entries.find(h);
    if (i == s.entries.end())
        return RK_ERR_RGN_UNEXIST;
    release_mb(i->second.canvas);
    s.entries.erase(i);
    return RK_SUCCESS;
}
#define ROCKIT_RGN_LOCK_ENTRY_PTR(handle_, pointer_)                                               \
    if ((handle_) >= RGN_MAX_HANDLE_NUM)                                                           \
        return RK_ERR_RGN_INVALID_DEVID;                                                           \
    if (!(pointer_))                                                                               \
        return RK_ERR_RGN_NULL_PTR;                                                                \
    auto &s = rgn_state();                                                                         \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    auto i = s.entries.find(handle_);                                                              \
    if (i == s.entries.end())                                                                      \
        return RK_ERR_RGN_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_GetAttr(RGN_HANDLE h, RGN_ATTR_S *a) {
    ROCKIT_RGN_LOCK_ENTRY_PTR(h, a);
    *a = i->second.attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_SetAttr(RGN_HANDLE h, const RGN_ATTR_S *a) {
    ROCKIT_RGN_LOCK_ENTRY_PTR(h, a);
    i->second.attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_SetBitMap(RGN_HANDLE h, const BITMAP_S *b) {
    if (h >= RGN_MAX_HANDLE_NUM)
        return RK_ERR_RGN_INVALID_DEVID;
    if (!b || !b->pData)
        return RK_ERR_RGN_NULL_PTR;
    auto &s = rgn_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.entries.find(h);
    if (i == s.entries.end())
        return RK_ERR_RGN_UNEXIST;
    if (i->second.canvas == MB_INVALID_HANDLE)
        return RK_ERR_RGN_NOT_SUPPORT;
    void *p = RK_MPI_MB_Handle2VirAddr(i->second.canvas);
    if (!p)
        return RK_ERR_RGN_BADADDR;
    const RK_U64 bytes = std::min<RK_U64>(
        RK_MPI_MB_GetSize(i->second.canvas),
        (static_cast<RK_U64>(b->u32Width) * b->u32Height * bits_per_pixel(b->enPixelFormat) + 7) /
            8);
    std::memcpy(p, b->pData, static_cast<std::size_t>(bytes));
    return RK_SUCCESS;
}
#define ROCKIT_RGN_REQUIRE_ATTACHMENT_ENTRY(handle_, channel_, attribute_)                         \
    if ((handle_) >= RGN_MAX_HANDLE_NUM)                                                           \
        return RK_ERR_RGN_INVALID_DEVID;                                                           \
    if (!(channel_) || !(attribute_))                                                              \
        return RK_ERR_RGN_NULL_PTR;                                                                \
    auto &s = rgn_state();                                                                         \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    auto i = s.entries.find(handle_);                                                              \
    if (i == s.entries.end())                                                                      \
    return RK_ERR_RGN_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_AttachToChn(RGN_HANDLE h,
                                                    const MPP_CHN_S *c,
                                                    const RGN_CHN_ATTR_S *a) {
    ROCKIT_RGN_REQUIRE_ATTACHMENT_ENTRY(h, c, a);
    auto k = key_of(*c);
    if (i->second.attachments.count(k))
        return RK_ERR_RGN_EXIST;
    i->second.attachments[k] = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_DetachFromChn(RGN_HANDLE h, const MPP_CHN_S *c) {
    ROCKIT_RGN_LOCK_ENTRY_PTR(h, c);
    return i->second.attachments.erase(key_of(*c)) ? RK_SUCCESS : RK_ERR_RGN_UNEXIST;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_SetDisplayAttr(RGN_HANDLE h,
                                                       const MPP_CHN_S *c,
                                                       const RGN_CHN_ATTR_S *a) {
    ROCKIT_RGN_REQUIRE_ATTACHMENT_ENTRY(h, c, a);
    auto k = key_of(*c);
    if (!i->second.attachments.count(k))
        return RK_ERR_RGN_UNEXIST;
    i->second.attachments[k] = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_GetDisplayAttr(RGN_HANDLE h,
                                                       const MPP_CHN_S *c,
                                                       RGN_CHN_ATTR_S *a) {
    ROCKIT_RGN_REQUIRE_ATTACHMENT_ENTRY(h, c, a);
    auto j = i->second.attachments.find(key_of(*c));
    if (j == i->second.attachments.end())
        return RK_ERR_RGN_UNEXIST;
    *a = j->second;
    return RK_SUCCESS;
}
#undef ROCKIT_RGN_REQUIRE_ATTACHMENT_ENTRY

extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_GetCanvasInfo(RGN_HANDLE h, RGN_CANVAS_INFO_S *c) {
    ROCKIT_RGN_LOCK_ENTRY_PTR(h, c);
    zero_value(c);
    c->canvasBlk = i->second.canvas;
    c->u64VirAddr = reinterpret_cast<RK_U64>(RK_MPI_MB_Handle2VirAddr(i->second.canvas));
    if (i->second.attr.enType == OVERLAY_RGN || i->second.attr.enType == OVERLAY_EX_RGN) {
        c->stSize = i->second.attr.unAttr.stOverlay.stSize;
        c->u32VirWidth = vir_width(c->stSize.u32Width, i->second.attr.unAttr.stOverlay.enPixelFmt);
        c->u32VirHeight = align_up(c->stSize.u32Height, 2);
        c->enPixelFmt = i->second.attr.unAttr.stOverlay.enPixelFmt;
    }
    return RK_SUCCESS;
}
#undef ROCKIT_RGN_LOCK_ENTRY_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_RGN_UpdateCanvas(RGN_HANDLE h) {
    if (h >= RGN_MAX_HANDLE_NUM)
        return RK_ERR_RGN_INVALID_DEVID;
    auto &s = rgn_state();
    std::lock_guard<std::mutex> l(s.mutex);
    return s.entries.count(h) ? RK_SUCCESS : RK_ERR_RGN_UNEXIST;
}

namespace rockit_local {
namespace {

struct RgbaPixel {
    RK_U8 red{0};
    RK_U8 green{0};
    RK_U8 blue{0};
    RK_U8 alpha{255};
};

RgbaPixel load_destination_pixel(const RK_U8 *pixel, PIXEL_FORMAT_E format) {
    switch (format) {
    case RK_FMT_YUV400SP:
        return {pixel[0], pixel[0], pixel[0], 255};
        ROCKIT_PACKED_RGBA_PIXEL_CASES;
    default:
        return {};
    }
}

void store_destination_pixel(RK_U8 *pixel, PIXEL_FORMAT_E format, const RgbaPixel &value) {
    const RK_U32 color = (static_cast<RK_U32>(value.red) << 16U) |
                         (static_cast<RK_U32>(value.green) << 8U) | value.blue;
    cpu_store_color(pixel, format, color);
    if (format == RK_FMT_RGBA8888 || format == RK_FMT_BGRA8888) {
        pixel[3] = value.alpha;
    } else if (format == RK_FMT_ARGB8888 || format == RK_FMT_ABGR8888 ||
               format == RK_FMT_XBGR8888) {
        pixel[0] = value.alpha;
    }
}

RgbaPixel load_overlay_pixel(const RK_U8 *pixel, PIXEL_FORMAT_E format) {
    if (format == RK_FMT_ARGB8888) {
        return {pixel[1], pixel[2], pixel[3], pixel[0]};
    }
    if (format == RK_FMT_BGRA8888) {
        return {pixel[2], pixel[1], pixel[0], pixel[3]};
    }
    if (format == RK_FMT_RGBA8888) {
        return {pixel[0], pixel[1], pixel[2], pixel[3]};
    }
    return {};
}

RK_U8 blend_channel(RK_U8 foreground, RK_U8 background, RK_U32 alpha) {
    return static_cast<RK_U8>((static_cast<RK_U32>(foreground) * alpha +
                               static_cast<RK_U32>(background) * (255U - alpha) + 127U) /
                              255U);
}

RK_U32 region_layer(const RGN_CHN_ATTR_S &attr) {
    switch (attr.enType) {
    case OVERLAY_RGN:
    case OVERLAY_EX_RGN:
        return attr.unChnAttr.stOverlayChn.u32Layer;
    case COVER_RGN:
        return attr.unChnAttr.stCoverChn.u32Layer;
    case MOSAIC_RGN:
        return attr.unChnAttr.stMosaicChn.u32Layer;
    case LINE_RGN:
        return 0;
    default:
        return 0;
    }
}

bool apply_overlay(const RgnEntry &entry,
                   const RGN_CHN_ATTR_S &attachment,
                   CpuSurface &destination) {
    if (entry.canvas == MB_INVALID_HANDLE || entry.attr.enType != attachment.enType ||
        entry.attr.unAttr.stOverlay.stSize.u32Width == 0 ||
        entry.attr.unAttr.stOverlay.stSize.u32Height == 0) {
        return false;
    }
    const PIXEL_FORMAT_E sourceFormat = entry.attr.unAttr.stOverlay.enPixelFmt;
    if (sourceFormat != RK_FMT_ARGB8888 && sourceFormat != RK_FMT_BGRA8888 &&
        sourceFormat != RK_FMT_RGBA8888) {
        return false;
    }
    const RK_U32 sourceWidth = entry.attr.unAttr.stOverlay.stSize.u32Width;
    const RK_U32 sourceHeight = entry.attr.unAttr.stOverlay.stSize.u32Height;
    const RK_U64 required = static_cast<RK_U64>(sourceWidth) * sourceHeight * 4U;
    if (required > RK_MPI_MB_GetSize(entry.canvas)) {
        return false;
    }
    const auto *source = static_cast<const RK_U8 *>(RK_MPI_MB_Handle2VirAddr(entry.canvas));
    if (!source) {
        return false;
    }
    const POINT_S &point = attachment.unChnAttr.stOverlayChn.stPoint;
    const RK_U32 globalAlpha = std::min<RK_U32>(attachment.unChnAttr.stOverlayChn.u32FgAlpha, 255U);
    for (RK_U32 sourceY = 0; sourceY < sourceHeight; ++sourceY) {
        const RK_S32 destinationY = point.s32Y + static_cast<RK_S32>(sourceY);
        if (destinationY < 0 || static_cast<RK_U32>(destinationY) >= destination.height) {
            continue;
        }
        for (RK_U32 sourceX = 0; sourceX < sourceWidth; ++sourceX) {
            const RK_S32 destinationX = point.s32X + static_cast<RK_S32>(sourceX);
            if (destinationX < 0 || static_cast<RK_U32>(destinationX) >= destination.width) {
                continue;
            }
            const RgbaPixel foreground = load_overlay_pixel(
                source + (static_cast<std::size_t>(sourceY) * sourceWidth + sourceX) * 4U,
                sourceFormat);
            const RK_U32 alpha =
                (static_cast<RK_U32>(foreground.alpha) * globalAlpha + 127U) / 255U;
            if (alpha == 0) {
                continue;
            }
            RK_U8 *destinationPixel = destination.data +
                                      static_cast<RK_U32>(destinationY) * destination.stride +
                                      static_cast<RK_U32>(destinationX) * destination.bytesPerPixel;
            const RgbaPixel background =
                load_destination_pixel(destinationPixel, destination.format);
            RgbaPixel result{};
            result.red = blend_channel(foreground.red, background.red, alpha);
            result.green = blend_channel(foreground.green, background.green, alpha);
            result.blue = blend_channel(foreground.blue, background.blue, alpha);
            result.alpha = 255;
            store_destination_pixel(destinationPixel, destination.format, result);
        }
    }
    return true;
}

bool apply_cover(const RGN_CHN_ATTR_S &attachment, CpuSurface &destination) {
    const COVER_CHN_ATTR_S &cover = attachment.unChnAttr.stCoverChn;
    if (cover.enCoverType != AREA_RECT || cover.enCoordinate != RGN_ABS_COOR) {
        return false;
    }
    return cpu_fill(destination,
                    cover.stRect.s32X,
                    cover.stRect.s32Y,
                    cover.stRect.u32Width,
                    cover.stRect.u32Height,
                    cover.u32Color) == RK_SUCCESS;
}

bool apply_mosaic_region(const RGN_CHN_ATTR_S &attachment, CpuSurface &destination) {
    const MOSAIC_CHN_ATTR_S &region = attachment.unChnAttr.stMosaicChn;
    if (region.enMosaicType != AREA_RECT || region.enBlkSize >= MOSAIC_BLK_SIZE_BUTT) {
        return false;
    }
    static constexpr RK_U32 blockSizes[] = {8U, 16U, 32U, 64U};
    VGS_MOSAIC_S mosaic{};
    mosaic.enBlkSize = static_cast<VGS_MOSAIC_BLK_SIZE_E>(blockSizes[region.enBlkSize]);
    mosaic.stDstRect = region.stRect;
    return cpu_mosaic(destination, mosaic) == RK_SUCCESS;
}

bool apply_line_region(const RGN_CHN_ATTR_S &attachment, CpuSurface &destination) {
    const LINE_CHN_ATTR_S &region = attachment.unChnAttr.stLineChn;
    VGS_DRAW_LINE_S line{};
    line.u32Thick = region.u32Thick;
    line.u32Color = region.u32Color;
    line.stStartPoint = region.stStartPoint;
    line.stEndPoint = region.stEndPoint;
    return cpu_draw_line(destination, line) == RK_SUCCESS;
}

} // namespace

RK_S32
apply_regions_to_frame(const MPP_CHN_S &channel, VIDEO_FRAME_INFO_S *frame, RK_U32 *appliedCount) {
    if (!frame) {
        return RK_FAILURE;
    }
    CpuSurface destination{};
    const bool packedDestination = cpu_surface_from_frame(*frame, destination);
    if (!packedDestination && !cpu_frame_is_supported(*frame)) {
        return RK_FAILURE;
    }
    auto &state = rgn_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const RgnAttachKey key = key_of(channel);
    std::vector<std::pair<const RgnEntry *, const RGN_CHN_ATTR_S *>> regions;
    for (const auto &[handle, entry] : state.entries) {
        (void)handle;
        const auto attachment = entry.attachments.find(key);
        if (attachment != entry.attachments.end() && attachment->second.bShow == RK_TRUE) {
            regions.emplace_back(&entry, &attachment->second);
        }
    }
    std::stable_sort(regions.begin(), regions.end(), [](const auto &left, const auto &right) {
        return region_layer(*left.second) < region_layer(*right.second);
    });
    RK_U32 count = 0;
    for (const auto &[entry, attachment] : regions) {
        bool applied = false;
        if (packedDestination) {
            switch (attachment->enType) {
            case OVERLAY_RGN:
            case OVERLAY_EX_RGN:
                applied = apply_overlay(*entry, *attachment, destination);
                break;
            case COVER_RGN:
                applied = apply_cover(*attachment, destination);
                break;
            case MOSAIC_RGN:
                applied = apply_mosaic_region(*attachment, destination);
                break;
            case LINE_RGN:
                applied = apply_line_region(*attachment, destination);
                break;
            default:
                break;
            }
        } else {
            switch (attachment->enType) {
            case OVERLAY_RGN:
            case OVERLAY_EX_RGN: {
                if (entry->canvas == MB_INVALID_HANDLE) {
                    break;
                }
                const RK_U8 *source =
                    static_cast<const RK_U8 *>(RK_MPI_MB_Handle2VirAddr(entry->canvas));
                if (!source) {
                    break;
                }
                const auto &overlay = entry->attr.unAttr.stOverlay;
                const auto &display = attachment->unChnAttr.stOverlayChn;
                applied = cpu_overlay_frame(frame,
                                            source,
                                            RK_MPI_MB_GetSize(entry->canvas),
                                            overlay.stSize.u32Width,
                                            overlay.stSize.u32Height,
                                            overlay.enPixelFmt,
                                            display.stPoint,
                                            display.u32FgAlpha) == RK_SUCCESS;
                break;
            }
            case COVER_RGN: {
                const auto &cover = attachment->unChnAttr.stCoverChn;
                if (cover.enCoverType == AREA_RECT && cover.enCoordinate == RGN_ABS_COOR) {
                    applied = cpu_fill_frame(frame, cover.stRect, cover.u32Color) == RK_SUCCESS;
                }
                break;
            }
            case MOSAIC_RGN: {
                const auto &mosaic = attachment->unChnAttr.stMosaicChn;
                static constexpr RK_U32 blockSizes[] = {8U, 16U, 32U, 64U};
                if (mosaic.enMosaicType == AREA_RECT && mosaic.enBlkSize >= MOSAIC_BLK_SIZE_8 &&
                    mosaic.enBlkSize < MOSAIC_BLK_SIZE_BUTT) {
                    applied =
                        cpu_mosaic_frame(frame,
                                         mosaic.stRect,
                                         blockSizes[static_cast<std::size_t>(mosaic.enBlkSize)]) ==
                        RK_SUCCESS;
                }
                break;
            }
            case LINE_RGN: {
                const auto &line = attachment->unChnAttr.stLineChn;
                applied =
                    cpu_draw_line_frame(
                        frame, line.stStartPoint, line.stEndPoint, line.u32Thick, line.u32Color) ==
                    RK_SUCCESS;
                break;
            }
            default:
                break;
            }
        }
        if (applied) {
            ++count;
        }
    }
    if (appliedCount) {
        *appliedCount = count;
    }
    return RK_SUCCESS;
}

} // namespace rockit_local

static RK_S32 ivs_channel_or_pointer_error(IVS_CHN c) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    auto &state = ivs_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.channels.count(c) ? RK_ERR_IVS_NULL_PTR : RK_ERR_IVS_UNEXIST;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_Init() {
    auto &s = ivs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.initialized = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_Exit() {
    auto &s = ivs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.channels.clear();
    s.initialized = false;
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_CreateChn(IVS_CHN c, IVS_CHN_ATTR_S *a) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    if (!a)
        return RK_ERR_IVS_NULL_PTR;
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    auto &s = ivs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.channels.count(c))
        return RK_ERR_IVS_EXIST;
    s.channels[c].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_DestroyChn(IVS_CHN c) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    auto &s = ivs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    return s.channels.erase(c) ? RK_SUCCESS : RK_ERR_IVS_UNEXIST;
}
#define ROCKIT_IVS_LOCK_EXISTING_CHANNEL_PTR(channel_, pointer_)                                   \
    if ((channel_) < 0 || (channel_) >= IVS_MAX_CHN_NUM)                                           \
        return RK_ERR_IVS_INVALID_CHNID;                                                           \
    if (!(pointer_))                                                                               \
        return ivs_channel_or_pointer_error(channel_);                                             \
    auto &s = ivs_state();                                                                         \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    auto i = s.channels.find(channel_);                                                            \
    if (i == s.channels.end())                                                                     \
    return RK_ERR_IVS_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_SetChnAttr(IVS_CHN c, IVS_CHN_ATTR_S *a) {
    ROCKIT_IVS_LOCK_EXISTING_CHANNEL_PTR(c, a);
    i->second.attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_GetChnAttr(IVS_CHN c, IVS_CHN_ATTR_S *a) {
    ROCKIT_IVS_LOCK_EXISTING_CHANNEL_PTR(c, a);
    *a = i->second.attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_SetMdAttr(IVS_CHN c, IVS_MD_ATTR_S *a) {
    ROCKIT_IVS_LOCK_EXISTING_CHANNEL_PTR(c, a);
    i->second.md = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_GetMdAttr(IVS_CHN c, IVS_MD_ATTR_S *a) {
    ROCKIT_IVS_LOCK_EXISTING_CHANNEL_PTR(c, a);
    *a = i->second.md;
    return RK_SUCCESS;
}
#undef ROCKIT_IVS_LOCK_EXISTING_CHANNEL_PTR

extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_SendFrame(IVS_CHN c,
                                                  const VIDEO_FRAME_INFO_S *f,
                                                  RK_S32 ms) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    {
        auto &state = ivs_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.channels.count(c))
            return RK_ERR_IVS_UNEXIST;
    }
    if (!f)
        return RK_ERR_IVS_NULL_PTR;
    if (ms < -1)
        return RK_ERR_IVS_ILLEGAL_PARAM;
    std::vector<RK_U8> luma;
    RK_U32 width = 0, height = 0;
    if (rockit_local::cpu_extract_luma(*f, &luma, &width, &height) != RK_SUCCESS)
        return RK_ERR_IVS_NOT_SUPPORT;
    auto &s = ivs_state();
    std::unique_lock<std::mutex> l(s.mutex);
    auto exists = [&] {
        return s.channels.find(c) != s.channels.end();
    };
    if (!exists())
        return RK_ERR_IVS_UNEXIST;
    auto space = [&] {
        auto i = s.channels.find(c);
        return i == s.channels.end() || i->second.results.size() < 8;
    };
    if (!wait_for_item(s.cv, l, ms, space))
        return RK_ERR_IVS_BUF_FULL;
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_IVS_UNEXIST;
    auto &channel = i->second;
    IvsResultData result{};
    result.frameId = ++channel.seq;
    const RK_S32 configured = channel.md.s32ThreshSad;
    const RK_U32 threshold =
        static_cast<RK_U32>(std::clamp(configured > 0 ? configured : 24, 1, 255));
    result.map.assign(luma.size(), 0);
    RK_U32 minX = width, minY = height, maxX = 0, maxY = 0, changed = 0;
    const bool comparable = channel.previousWidth == width && channel.previousHeight == height &&
                            channel.previousLuma.size() == luma.size();
    if (comparable) {
        for (RK_U32 y = 0; y < height; ++y) {
            for (RK_U32 x = 0; x < width; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * width + x;
                const RK_U32 left = luma[index], right = channel.previousLuma[index];
                const RK_U32 difference = left > right ? left - right : right - left;
                if (difference >= threshold) {
                    result.map[index] = 1;
                    ++changed;
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                }
            }
        }
    }
    result.square = changed;
    const RK_U32 minimum = std::max<RK_U32>(
        1, (width * height) / (128U * std::max<RK_U32>(1, channel.attr.u32MDSensibility)));
    if (changed >= minimum && minX <= maxX && minY <= maxY) {
        RECT_S rectangle{};
        rectangle.s32X = static_cast<RK_S32>(minX);
        rectangle.s32Y = static_cast<RK_S32>(minY);
        rectangle.u32Width = maxX - minX + 1;
        rectangle.u32Height = maxY - minY + 1;
        result.rectangles.push_back(rectangle);
    }
    RK_U32 dark = 0;
    for (const RK_U8 value : luma)
        if (value < 24)
            ++dark;
    result.pixelSum = dark;
    const RK_U32 percent = width && height
                               ? static_cast<RK_U32>((static_cast<RK_U64>(dark) * 100U) /
                                                     (static_cast<RK_U64>(width) * height))
                               : 0;
    const RK_U32 odThreshold =
        static_cast<RK_U32>(std::clamp(channel.attr.s32ODPercent, 5, 8)) * 10U;
    result.odFlag = percent >= odThreshold ? 1U : 0U;
    channel.previousLuma = std::move(luma);
    channel.previousWidth = width;
    channel.previousHeight = height;
    channel.results.push_back(std::move(result));
    s.cv.notify_all();
    return RK_SUCCESS;
}
static RK_S32 ivs_get(IVS_CHN c, IVS_RESULT_INFO_S *r, RK_S32 ms) {
    if (!r)
        return ivs_channel_or_pointer_error(c);
    if (ms < -1)
        return RK_ERR_IVS_ILLEGAL_PARAM;
    auto &s = ivs_state();
    std::unique_lock<std::mutex> l(s.mutex);
    auto exists = [&] {
        return s.channels.find(c) != s.channels.end();
    };
    if (!exists())
        return RK_ERR_IVS_UNEXIST;
    auto ready = [&] {
        auto i = s.channels.find(c);
        return i == s.channels.end() || !i->second.results.empty();
    };
    if (!wait_for_item(s.cv, l, ms, ready))
        return RK_ERR_IVS_BUF_EMPTY;
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_IVS_UNEXIST;
    if (i->second.results.empty())
        return RK_ERR_IVS_BUF_EMPTY;
    IvsResultData data = std::move(i->second.results.front());
    i->second.results.pop_front();
    s.cv.notify_all();
    auto *one = new (std::nothrow) IVS_RESULT_S[1]{};
    if (!one)
        return RK_ERR_IVS_NOMEM;
    one->stMdInfo.frameId = data.frameId;
    one->stMdInfo.u32RectNum =
        static_cast<RK_U32>(std::min<std::size_t>(data.rectangles.size(), 4096));
    for (RK_U32 index = 0; index < one->stMdInfo.u32RectNum; ++index)
        one->stMdInfo.stRect[index] = data.rectangles[index];
    one->stMdInfo.u32Square = data.square;
    one->stMdInfo.u32Size = static_cast<RK_U32>(data.map.size());
    one->stMdInfo.s32Fd = -1;
    if (!data.map.empty()) {
        one->stMdInfo.pData = new (std::nothrow) RK_CHAR[data.map.size()];
        if (!one->stMdInfo.pData) {
            delete[] one;
            return RK_ERR_IVS_NOMEM;
        }
        std::copy(data.map.begin(), data.map.end(), one->stMdInfo.pData);
    }
    one->stOdInfo.frameId = data.frameId;
    one->stOdInfo.u32Flag = data.odFlag;
    one->stOdInfo.u32PixSum = data.pixelSum;
    r->s32ResultNum = 1;
    r->pstResults = one;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_GetResults(IVS_CHN c, IVS_RESULT_INFO_S *r, RK_S32 ms) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    return ivs_get(c, r, ms);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_GetResultsRaw(IVS_CHN c, IVS_RESULT_INFO_S *r, RK_S32 ms) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    return ivs_get(c, r, ms);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_IVS_ReleaseResults(IVS_CHN c, IVS_RESULT_INFO_S *r) {
    if (c < 0 || c >= IVS_MAX_CHN_NUM)
        return RK_ERR_IVS_INVALID_CHNID;
    {
        auto &state = ivs_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.channels.count(c))
            return RK_ERR_IVS_UNEXIST;
    }
    if (!r)
        return RK_ERR_IVS_NULL_PTR;
    if (r->pstResults) {
        for (RK_S32 index = 0; index < r->s32ResultNum; ++index)
            delete[] r->pstResults[index].stMdInfo.pData;
    }
    delete[] r->pstResults;
    r->pstResults = nullptr;
    r->s32ResultNum = 0;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_CreateGrp(AVS_GRP g, const AVS_GRP_ATTR_S *a) {
    if (g < 0 || g >= AVS_MAX_GRP_NUM)
        return RK_ERR_AVS_INVALID_DEVID;
    if (!a)
        return RK_ERR_AVS_NULL_PTR;
    if (g < 0 || g >= AVS_MAX_GRP_NUM)
        return RK_ERR_AVS_INVALID_DEVID;
    auto &s = avs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.groups.count(g))
        return RK_ERR_AVS_EXIST;
    s.groups[g].attr = *a;
    return RK_SUCCESS;
}
#define ROCKIT_AVS_LOCK_GROUP(group_)                                                              \
    if ((group_) < 0 || (group_) >= AVS_MAX_GRP_NUM)                                              \
        return RK_ERR_AVS_INVALID_DEVID;                                                          \
    auto &s = avs_state();                                                                        \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    auto i = s.groups.find(group_);                                                               \
    if (i == s.groups.end())                                                                      \
        return RK_ERR_AVS_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_DestroyGrp(AVS_GRP g) {
    ROCKIT_AVS_LOCK_GROUP(g);
    clear_avs(i->second);
    s.groups.erase(i);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_StartGrp(AVS_GRP g) {
    ROCKIT_AVS_LOCK_GROUP(g);
    i->second.started = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_StopGrp(AVS_GRP g) {
    ROCKIT_AVS_LOCK_GROUP(g);
    i->second.started = false;
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_ResetGrp(AVS_GRP g) {
    ROCKIT_AVS_LOCK_GROUP(g);
    clear_avs(i->second);
    return RK_SUCCESS;
}
#define ROCKIT_AVS_LOCK_GROUP_PTR(group_, pointer_)                                                \
    if ((group_) < 0 || (group_) >= AVS_MAX_GRP_NUM)                                              \
        return RK_ERR_AVS_INVALID_DEVID;                                                          \
    if (!(pointer_))                                                                               \
        return RK_ERR_AVS_NULL_PTR;                                                               \
    auto &s = avs_state();                                                                        \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    auto i = s.groups.find(group_);                                                               \
    if (i == s.groups.end())                                                                      \
        return RK_ERR_AVS_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_GetGrpAttr(AVS_GRP g, AVS_GRP_ATTR_S *a) {
    ROCKIT_AVS_LOCK_GROUP_PTR(g, a);
    *a = i->second.attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_SetGrpAttr(AVS_GRP g, const AVS_GRP_ATTR_S *a) {
    ROCKIT_AVS_LOCK_GROUP_PTR(g, a);
    i->second.attr = *a;
    return RK_SUCCESS;
}
#undef ROCKIT_AVS_LOCK_GROUP_PTR
#undef ROCKIT_AVS_LOCK_GROUP
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_SendPipeFrame(AVS_GRP g,
                                                      AVS_PIPE p,
                                                      const VIDEO_FRAME_INFO_S *f,
                                                      RK_S32 ms) {
    if (g < 0 || g >= AVS_MAX_GRP_NUM)
        return RK_ERR_AVS_INVALID_DEVID;
    if (!f)
        return RK_ERR_AVS_NULL_PTR;
    if (p < 0 || p >= AVS_PIPE_NUM)
        return RK_ERR_AVS_INVALID_PIPEID;
    if (ms < -1)
        return RK_ERR_AVS_ILLEGAL_PARAM;
    auto &s = avs_state();
    std::unique_lock<std::mutex> l(s.mutex);
    auto exists = [&] {
        return s.groups.find(g) != s.groups.end();
    };
    if (!exists())
        return RK_ERR_AVS_UNEXIST;
    auto room = [&] {
        auto i = s.groups.find(g);
        if (i == s.groups.end())
            return true;
        if (i->second.pipes[p].size() >= 8)
            return false;
        for (const auto &channel : i->second.channels)
            if (channel.configured && channel.enabled &&
                channel.ready.size() >= avs_channel_capacity(channel))
                return false;
        return true;
    };
    if (!wait_for_item(s.cv, l, ms, room))
        return RK_ERR_AVS_BUF_FULL;
    auto i = s.groups.find(g);
    if (i == s.groups.end())
        return RK_ERR_AVS_UNEXIST;
    if (!i->second.started)
        return RK_ERR_AVS_NOTREADY;
    MB_BLK b = f->stVFrame.pMbBlk;
    if (!retain_mb(b))
        return RK_ERR_AVS_ILLEGAL_PARAM;
    if (!retain_mb(b)) {
        release_mb(b);
        return RK_ERR_AVS_ILLEGAL_PARAM;
    }
    i->second.pipes[p].push_back({*f});
    if (i->second.latestValid[p])
        release_mb(i->second.latest[p].frame.stVFrame.pMbBlk);
    i->second.latest[p] = {*f};
    i->second.latestValid[p] = true;
    const RK_U32 pipeCount = std::clamp<RK_U32>(i->second.attr.u32PipeNum, 1, AVS_PIPE_NUM);
    bool allPipesReady = true;
    for (RK_U32 pipe = 0; pipe < pipeCount; ++pipe) {
        allPipesReady = allPipesReady && i->second.latestValid[pipe];
    }
    std::vector<std::pair<MPP_CHN_S, VIDEO_FRAME_INFO_S>> routeFrames;
    if (allPipesReady) {
        for (std::size_t channelId = 0; channelId < i->second.channels.size(); ++channelId) {
            auto &channel = i->second.channels[channelId];
            if (!channel.configured || !channel.enabled)
                continue;
            VIDEO_FRAME_INFO_S output{};
            if (build_avs_output(i->second, channel, &output)) {
                channel.ready.push_back({output});
            } else {
                if (!retain_mb(b))
                    continue;
                output = *f;
                if (channel.attr.u32Width) {
                    output.stVFrame.u32Width = channel.attr.u32Width;
                    output.stVFrame.u32VirWidth = channel.attr.u32Width;
                }
                if (channel.attr.u32Height) {
                    output.stVFrame.u32Height = channel.attr.u32Height;
                    output.stVFrame.u32VirHeight = channel.attr.u32Height;
                }
                channel.ready.push_back({output});
            }
            routeFrames.push_back(
                {MPP_CHN_S{RK_ID_AVS, g, static_cast<RK_S32>(channelId)}, output});
        }
    }
    s.cv.notify_all();
    l.unlock();
    for (const auto &route : routeFrames) {
        const RK_S32 routeResult = rockit_local::route_video_frame(route.first, route.second, ms);
        if (routeResult != RK_SUCCESS)
            return routeResult;
    }
    return RK_SUCCESS;
}
#define ROCKIT_AVS_LOCK_PIPE_FRAME(group_, pipe_, frame_)                                          \
    if ((group_) < 0 || (group_) >= AVS_MAX_GRP_NUM)                                               \
        return RK_ERR_AVS_INVALID_DEVID;                                                           \
    if (!(frame_))                                                                                 \
        return RK_ERR_AVS_NULL_PTR;                                                                \
    if ((pipe_) < 0 || (pipe_) >= AVS_PIPE_NUM)                                                    \
        return RK_ERR_AVS_INVALID_PIPEID;                                                          \
    auto &s = avs_state();                                                                         \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    auto i = s.groups.find(group_);                                                                \
    if (i == s.groups.end())                                                                       \
    return RK_ERR_AVS_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_GetPipeFrame(AVS_GRP g, AVS_PIPE p, VIDEO_FRAME_INFO_S *f) {
    ROCKIT_AVS_LOCK_PIPE_FRAME(g, p, f);
    if (i->second.pipes[p].empty())
        return RK_ERR_AVS_BUF_EMPTY;
    *f = i->second.pipes[p].front().frame;
    i->second.pipes[p].pop_front();
    ++i->second.pipeOutstanding[p][f->stVFrame.pMbBlk];
    s.cv.notify_all();
    return RK_SUCCESS;
}
#define ROCKIT_RELEASE_AVS_OUTSTANDING_FRAME(map_, frame_)                                       \
    auto j = (map_).find((frame_)->stVFrame.pMbBlk);                                              \
    if (j == (map_).end())                                                                        \
        return RK_ERR_AVS_ILLEGAL_PARAM;                                                          \
    if (--j->second == 0)                                                                         \
        (map_).erase(j);                                                                          \
    release_mb((frame_)->stVFrame.pMbBlk);                                                        \
    s.cv.notify_all();                                                                             \
    return RK_SUCCESS

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_ReleasePipeFrame(AVS_GRP g,
                                                         AVS_PIPE p,
                                                         const VIDEO_FRAME_INFO_S *f) {
    ROCKIT_AVS_LOCK_PIPE_FRAME(g, p, f);
    auto &map = i->second.pipeOutstanding[p];
    ROCKIT_RELEASE_AVS_OUTSTANDING_FRAME(map, f);
}
#undef ROCKIT_AVS_LOCK_PIPE_FRAME

#define ROCKIT_AVS_LOCK_CHANNEL_PTR(group_, channel_, pointer_)                                    \
    if ((group_) < 0 || (group_) >= AVS_MAX_GRP_NUM)                                              \
        return RK_ERR_AVS_INVALID_DEVID;                                                          \
    if (!(pointer_))                                                                               \
        return RK_ERR_AVS_NULL_PTR;                                                               \
    if ((channel_) < 0 || (channel_) >= AVS_MAX_CHN_NUM)                                          \
        return RK_ERR_AVS_INVALID_CHNID;                                                          \
    auto &s = avs_state();                                                                        \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    auto i = s.groups.find(group_);                                                               \
    if (i == s.groups.end())                                                                      \
        return RK_ERR_AVS_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_SetChnAttr(AVS_GRP g, AVS_CHN c, const AVS_CHN_ATTR_S *a) {
    ROCKIT_AVS_LOCK_CHANNEL_PTR(g, c, a);
    i->second.channels[c].configured = true;
    i->second.channels[c].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_GetChnAttr(AVS_GRP g, AVS_CHN c, AVS_CHN_ATTR_S *a) {
    ROCKIT_AVS_LOCK_CHANNEL_PTR(g, c, a);
    if (!i->second.channels[c].configured)
        return RK_ERR_AVS_UNEXIST;
    *a = i->second.channels[c].attr;
    return RK_SUCCESS;
}
#define ROCKIT_AVS_LOCK_CONFIGURED_CHANNEL(group_, channel_)                                      \
    if ((group_) < 0 || (group_) >= AVS_MAX_GRP_NUM)                                              \
        return RK_ERR_AVS_INVALID_DEVID;                                                          \
    if ((channel_) < 0 || (channel_) >= AVS_MAX_CHN_NUM)                                          \
        return RK_ERR_AVS_INVALID_CHNID;                                                          \
    auto &s = avs_state();                                                                        \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    auto i = s.groups.find(group_);                                                               \
    if (i == s.groups.end() || !i->second.channels[channel_].configured)                           \
        return RK_ERR_AVS_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_EnableChn(AVS_GRP g, AVS_CHN c) {
    ROCKIT_AVS_LOCK_CONFIGURED_CHANNEL(g, c);
    i->second.channels[c].enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_DisableChn(AVS_GRP g, AVS_CHN c) {
    ROCKIT_AVS_LOCK_CONFIGURED_CHANNEL(g, c);
    i->second.channels[c].enabled = false;
    clear_q(i->second.channels[c].ready);
    return RK_SUCCESS;
}
#undef ROCKIT_AVS_LOCK_CONFIGURED_CHANNEL
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_GetChnFrame(AVS_GRP g,
                                                    AVS_CHN c,
                                                    VIDEO_FRAME_INFO_S *f,
                                                    RK_S32 ms) {
    if (g < 0 || g >= AVS_MAX_GRP_NUM)
        return RK_ERR_AVS_INVALID_DEVID;
    if (!f)
        return RK_ERR_AVS_NULL_PTR;
    if (c < 0 || c >= AVS_MAX_CHN_NUM)
        return RK_ERR_AVS_INVALID_CHNID;
    auto &s = avs_state();
    std::unique_lock<std::mutex> l(s.mutex);
    auto ready = [&] {
        auto i = s.groups.find(g);
        return i == s.groups.end() || !i->second.channels[c].ready.empty();
    };
    if (!wait_for_item(s.cv, l, ms, ready))
        return RK_ERR_AVS_BUF_EMPTY;
    auto i = s.groups.find(g);
    if (i == s.groups.end() || !i->second.channels[c].configured)
        return RK_ERR_AVS_UNEXIST;
    auto &ch = i->second.channels[c];
    if (ch.ready.empty())
        return RK_ERR_AVS_BUF_EMPTY;
    *f = ch.ready.front().frame;
    ch.ready.pop_front();
    ++ch.outstanding[f->stVFrame.pMbBlk];
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_ReleaseChnFrame(AVS_GRP g,
                                                        AVS_CHN c,
                                                        const VIDEO_FRAME_INFO_S *f) {
    ROCKIT_AVS_LOCK_CHANNEL_PTR(g, c, f);
    auto &map = i->second.channels[c].outstanding;
    ROCKIT_RELEASE_AVS_OUTSTANDING_FRAME(map, f);
}
#undef ROCKIT_RELEASE_AVS_OUTSTANDING_FRAME
#undef ROCKIT_AVS_LOCK_CHANNEL_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_SetModParam(const AVS_MOD_PARAM_S *p) {
    if (!p)
        return RK_ERR_AVS_NULL_PTR;
    auto &s = avs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.mod = *p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_GetModParam(AVS_MOD_PARAM_S *p) {
    if (!p)
        return RK_ERR_AVS_NULL_PTR;
    auto &s = avs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *p = s.mod;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AVS_GetFinalLut(AVS_GRP g, AVS_FINAL_LUT_S *p) {
    if (!p)
        return RK_ERR_AVS_NULL_PTR;
    auto &s = avs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.groups.count(g))
        return RK_ERR_AVS_UNEXIST;
    zero_value(p);
    return RK_SUCCESS;
}

#define ROCKIT_PVS_LOCK_DEVICE_PTR(device_, pointer_)                                              \
    if ((device_) < 0 || (device_) >= PVS_MAX_DEV_NUM)                                             \
        return RK_ERR_PVS_INVALID_DEVID;                                                           \
    if (!(pointer_))                                                                               \
        return RK_ERR_PVS_NULL_PTR;                                                                \
    if ((device_) < 0 || (device_) >= PVS_MAX_DEV_NUM)                                             \
        return RK_ERR_PVS_INVALID_DEVID;                                                           \
    auto &s = pvs_state();                                                                         \
    std::lock_guard<std::mutex> l(s.mutex)

#define ROCKIT_PVS_LOCK_CHANNEL_PTR(device_, channel_, pointer_)                                   \
    if ((device_) < 0 || (device_) >= PVS_MAX_DEV_NUM)                                             \
        return RK_ERR_PVS_INVALID_DEVID;                                                           \
    if (!(pointer_))                                                                               \
        return RK_ERR_PVS_NULL_PTR;                                                                \
    if ((device_) < 0 || (device_) >= PVS_MAX_DEV_NUM)                                             \
        return RK_ERR_PVS_INVALID_DEVID;                                                           \
    if ((channel_) < 0 || (channel_) >= PVS_MAX_CHN_NUM)                                           \
        return RK_ERR_PVS_INVALID_CHNID;                                                           \
    auto &s = pvs_state();                                                                         \
    std::lock_guard<std::mutex> l(s.mutex)

extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_SetDevAttr(PVS_DEV d, const PVS_DEV_ATTR_S *a) {
    ROCKIT_PVS_LOCK_DEVICE_PTR(d, a);
    s.devs[d].configured = true;
    s.devs[d].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_GetDevAttr(PVS_DEV d, PVS_DEV_ATTR_S *a) {
    ROCKIT_PVS_LOCK_DEVICE_PTR(d, a);
    if (!s.devs[d].configured)
        return RK_ERR_PVS_NOT_CONFIG;
    *a = s.devs[d].attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_EnableDev(PVS_DEV d) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].configured)
        return RK_ERR_PVS_NOT_CONFIG;
    s.devs[d].enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_DisableDev(PVS_DEV d) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d];
    x.enabled = false;
    clear_q(x.ready);
    for (auto &[b, n] : x.outstanding)
        for (RK_U32 i = 0; i < n; ++i)
            release_mb(b);
    x.outstanding.clear();
    for (std::size_t i = 0; i < x.latest.size(); ++i) {
        if (x.latestValid[i])
            release_mb(x.latest[i].frame.stVFrame.pMbBlk);
        x.latestValid[i] = false;
        x.latest[i] = VideoItem{};
    }
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_SetChnAttr(PVS_DEV d, PVS_CHN c, PVS_CHN_ATTR_S *a) {
    ROCKIT_PVS_LOCK_CHANNEL_PTR(d, c, a);
    s.devs[d].channels[c].configured = true;
    s.devs[d].channels[c].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_GetChnAttr(PVS_DEV d, PVS_CHN c, PVS_CHN_ATTR_S *a) {
    ROCKIT_PVS_LOCK_CHANNEL_PTR(d, c, a);
    auto &x = s.devs[d].channels[c];
    if (!x.configured)
        return RK_ERR_PVS_NOT_CONFIG;
    *a = x.attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_SetChnParam(PVS_DEV d, PVS_CHN c, PVS_CHN_PARAM_S *a) {
    ROCKIT_PVS_LOCK_CHANNEL_PTR(d, c, a);
    s.devs[d].channels[c].param = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_GetChnParam(PVS_DEV d, PVS_CHN c, PVS_CHN_PARAM_S *a) {
    ROCKIT_PVS_LOCK_CHANNEL_PTR(d, c, a);
    *a = s.devs[d].channels[c].param;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_EnableChn(PVS_DEV d, PVS_CHN c) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    if (c < 0 || c >= PVS_MAX_CHN_NUM)
        return RK_ERR_PVS_INVALID_CHNID;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d].channels[c];
    if (!x.configured)
        return RK_ERR_PVS_NOT_CONFIG;
    x.enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_DisableChn(PVS_DEV d, PVS_CHN c) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    if (c < 0 || c >= PVS_MAX_CHN_NUM)
        return RK_ERR_PVS_INVALID_CHNID;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d];
    x.channels[c].enabled = false;
    if (x.latestValid[c]) {
        release_mb(x.latest[c].frame.stVFrame.pMbBlk);
        x.latestValid[c] = false;
        x.latest[c] = VideoItem{};
    }
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_SendFrame(PVS_DEV d,
                                                  PVS_CHN c,
                                                  const VIDEO_FRAME_INFO_S *f) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    if (!f)
        return RK_ERR_PVS_NULL_PTR;
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    if (c < 0 || c >= PVS_MAX_CHN_NUM)
        return RK_ERR_PVS_INVALID_CHNID;
    auto &s = pvs_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto &device = s.devs[d];
    if (!device.enabled || !device.channels[c].enabled)
        return RK_ERR_PVS_UNEXIST;
    if (device.ready.size() >= 8)
        return RK_ERR_PVS_BUF_FULL;
    MB_BLK block = f->stVFrame.pMbBlk;
    if (!retain_mb(block))
        return RK_ERR_PVS_ILLEGAL_PARAM;
    if (device.latestValid[c]) {
        release_mb(device.latest[c].frame.stVFrame.pMbBlk);
    }
    device.latest[c] = {*f};
    device.latestValid[c] = true;
    VIDEO_FRAME_INFO_S output{};
    if (build_pvs_output(device, &output)) {
        device.ready.push_back({output});
    } else {
        if (!retain_mb(block))
            return RK_ERR_PVS_ILLEGAL_PARAM;
        device.ready.push_back({*f});
    }
    s.cv.notify_all();
    const MPP_CHN_S sourceChannel{RK_ID_PVS, d, 0};
    lock.unlock();
    return rockit_local::route_video_frame(sourceChannel, output, 0);
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_GetFrame(PVS_DEV d, VIDEO_FRAME_INFO_S *f, RK_S32 ms) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    if (!f)
        return RK_ERR_PVS_NULL_PTR;
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    auto &s = pvs_state();
    std::unique_lock<std::mutex> l(s.mutex);
    auto ready = [&] {
        return !s.devs[d].enabled || !s.devs[d].ready.empty();
    };
    if (!wait_for_item(s.cv, l, ms, ready))
        return RK_ERR_PVS_BUF_EMPTY;
    auto &x = s.devs[d];
    if (x.ready.empty())
        return RK_ERR_PVS_BUF_EMPTY;
    *f = x.ready.front().frame;
    x.ready.pop_front();
    ++x.outstanding[f->stVFrame.pMbBlk];
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_ReleaseFrame(const VIDEO_FRAME_INFO_S *f) {
    if (!f)
        return RK_ERR_PVS_NULL_PTR;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    for (auto &x : s.devs) {
        auto i = x.outstanding.find(f->stVFrame.pMbBlk);
        if (i != x.outstanding.end()) {
            if (--i->second == 0)
                x.outstanding.erase(i);
            release_mb(f->stVFrame.pMbBlk);
            s.cv.notify_all();
            return RK_SUCCESS;
        }
    }
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_SetVProcDev(PVS_DEV d, VIDEO_PROC_DEV_TYPE_E p) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].proc = p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_PVS_GetVProcDev(PVS_DEV d, VIDEO_PROC_DEV_TYPE_E *p) {
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    if (!p)
        return RK_ERR_PVS_NULL_PTR;
    if (d < 0 || d >= PVS_MAX_DEV_NUM)
        return RK_ERR_PVS_INVALID_DEVID;
    auto &s = pvs_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *p = s.devs[d].proc;
    return RK_SUCCESS;
}

#undef ROCKIT_PVS_LOCK_CHANNEL_PTR
#undef ROCKIT_PVS_LOCK_DEVICE_PTR
