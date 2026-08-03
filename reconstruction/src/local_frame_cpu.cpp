#include "local_frame_cpu.hpp"
#include "rgba_pixel_cases.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"

namespace rockit_local {
namespace {

enum class StorageKind {
    Packed,
    SemiPlanar,
    Planar,
    Packed422,
};

struct FormatLayout {
    StorageKind storage{StorageKind::Packed};
    RK_U32 packedBytes{0};
    RK_U32 chromaSubX{1};
    RK_U32 chromaSubY{1};
    bool vuOrder{false};
    bool yuv{false};
};

struct FrameView {
    RK_U8 *data{nullptr};
    RK_U64 bytes{0};
    RK_U32 width{0};
    RK_U32 height{0};
    RK_U32 virtualWidth{0};
    RK_U32 virtualHeight{0};
    PIXEL_FORMAT_E format{RK_FMT_BUTT};
    FormatLayout layout{};
};

struct YuvPixel {
    RK_U8 y{0};
    RK_U8 u{128};
    RK_U8 v{128};
};

struct RgbaPixel {
    RK_U8 red{0};
    RK_U8 green{0};
    RK_U8 blue{0};
    RK_U8 alpha{255};
};

bool checked_add(RK_U64 left, RK_U64 right, RK_U64 *result) {
    if (!result || right > std::numeric_limits<RK_U64>::max() - left) {
        return false;
    }
    *result = left + right;
    return true;
}

bool checked_mul(RK_U64 left, RK_U64 right, RK_U64 *result) {
    if (!result || (left != 0 && right > std::numeric_limits<RK_U64>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

RK_U32 ceil_div(RK_U32 value, RK_U32 divisor) {
    return divisor ? (value + divisor - 1U) / divisor : 0U;
}

bool format_layout(PIXEL_FORMAT_E format, FormatLayout *layout) {
    if (!layout) {
        return false;
    }
    switch (format) {
    case RK_FMT_YUV400SP:
        *layout = {StorageKind::Packed, 1, 1, 1, false, true};
        return true;
    case RK_FMT_RGB888:
    case RK_FMT_BGR888:
        *layout = {StorageKind::Packed, 3, 1, 1, false, false};
        return true;
    case RK_FMT_ARGB8888:
    case RK_FMT_ABGR8888:
    case RK_FMT_BGRA8888:
    case RK_FMT_RGBA8888:
    case RK_FMT_XBGR8888:
        *layout = {StorageKind::Packed, 4, 1, 1, false, false};
        return true;
    case RK_FMT_YUV420SP:
        *layout = {StorageKind::SemiPlanar, 0, 2, 2, false, true};
        return true;
    case RK_FMT_YUV420SP_VU:
        *layout = {StorageKind::SemiPlanar, 0, 2, 2, true, true};
        return true;
    case RK_FMT_YUV422SP:
        *layout = {StorageKind::SemiPlanar, 0, 2, 1, false, true};
        return true;
    case RK_FMT_YUV422SP_VU:
        *layout = {StorageKind::SemiPlanar, 0, 2, 1, true, true};
        return true;
    case RK_FMT_YUV420P:
        *layout = {StorageKind::Planar, 0, 2, 2, false, true};
        return true;
    case RK_FMT_YUV420P_VU:
        *layout = {StorageKind::Planar, 0, 2, 2, true, true};
        return true;
    case RK_FMT_YUV422P:
        *layout = {StorageKind::Planar, 0, 2, 1, false, true};
        return true;
    case RK_FMT_YUV444SP:
        *layout = {StorageKind::SemiPlanar, 0, 1, 1, false, true};
        return true;
    case RK_FMT_YUV444P:
        *layout = {StorageKind::Planar, 0, 1, 1, false, true};
        return true;
    case RK_FMT_YUV444:
        *layout = {StorageKind::Packed, 3, 1, 1, false, true};
        return true;
    case RK_FMT_YUV422_YUYV:
    case RK_FMT_YUV422_UYVY:
    case RK_FMT_YUV422_YVYU:
    case RK_FMT_YUV422_VYUY:
        *layout = {StorageKind::Packed422, 0, 2, 1, false, true};
        return true;
    default:
        return false;
    }
}

bool storage_size(const FormatLayout &layout,
                  RK_U32 virtualWidth,
                  RK_U32 virtualHeight,
                  RK_U64 *required) {
    if (!required || virtualWidth == 0 || virtualHeight == 0) {
        return false;
    }
    RK_U64 yBytes = 0;
    switch (layout.storage) {
    case StorageKind::Packed:
        return checked_mul(
            static_cast<RK_U64>(virtualWidth) * layout.packedBytes, virtualHeight, required);
    case StorageKind::Packed422:
        return checked_mul(
            static_cast<RK_U64>(ceil_div(virtualWidth, 2U)) * 4U, virtualHeight, required);
    case StorageKind::SemiPlanar: {
        if (!checked_mul(virtualWidth, virtualHeight, &yBytes)) {
            return false;
        }
        RK_U64 chromaBytes = 0;
        const RK_U32 chromaWidth = ceil_div(virtualWidth, layout.chromaSubX);
        const RK_U32 chromaHeight = ceil_div(virtualHeight, layout.chromaSubY);
        if (!checked_mul(static_cast<RK_U64>(chromaWidth) * 2U, chromaHeight, &chromaBytes)) {
            return false;
        }
        return checked_add(yBytes, chromaBytes, required);
    }
    case StorageKind::Planar: {
        if (!checked_mul(virtualWidth, virtualHeight, &yBytes)) {
            return false;
        }
        RK_U64 onePlane = 0;
        const RK_U32 chromaWidth = ceil_div(virtualWidth, layout.chromaSubX);
        const RK_U32 chromaHeight = ceil_div(virtualHeight, layout.chromaSubY);
        if (!checked_mul(chromaWidth, chromaHeight, &onePlane)) {
            return false;
        }
        RK_U64 chromaBytes = 0;
        if (!checked_mul(onePlane, 2U, &chromaBytes)) {
            return false;
        }
        return checked_add(yBytes, chromaBytes, required);
    }
    }
    return false;
}

bool view_from_frame(const VIDEO_FRAME_INFO_S &frame, FrameView *view) {
    if (!view) {
        return false;
    }
    const VIDEO_FRAME_S &video = frame.stVFrame;
    FormatLayout layout{};
    if (video.pMbBlk == MB_INVALID_HANDLE || video.u32Width == 0 || video.u32Height == 0 ||
        video.enCompressMode != COMPRESS_MODE_NONE ||
        !format_layout(video.enPixelFormat, &layout)) {
        return false;
    }
    const RK_U32 virtualWidth = video.u32VirWidth ? video.u32VirWidth : video.u32Width;
    const RK_U32 virtualHeight = video.u32VirHeight ? video.u32VirHeight : video.u32Height;
    if (virtualWidth < video.u32Width || virtualHeight < video.u32Height) {
        return false;
    }
    RK_U64 required = 0;
    if (!storage_size(layout, virtualWidth, virtualHeight, &required) ||
        required > RK_MPI_MB_GetSize(video.pMbBlk)) {
        return false;
    }
    auto *data = static_cast<RK_U8 *>(RK_MPI_MB_Handle2VirAddr(video.pMbBlk));
    if (!data) {
        return false;
    }
    *view = {data,
             required,
             video.u32Width,
             video.u32Height,
             virtualWidth,
             virtualHeight,
             video.enPixelFormat,
             layout};
    return true;
}

RK_U8 clamp_byte(RK_S32 value) {
    return static_cast<RK_U8>(std::clamp(value, 0, 255));
}

YuvPixel rgb_to_yuv(const RgbaPixel &pixel) {
    const RK_S32 red = pixel.red;
    const RK_S32 green = pixel.green;
    const RK_S32 blue = pixel.blue;
    YuvPixel result{};
    result.y = clamp_byte((77 * red + 150 * green + 29 * blue + 128) >> 8);
    result.u = clamp_byte(128 + ((-43 * red - 85 * green + 128 * blue + 128) >> 8));
    result.v = clamp_byte(128 + ((128 * red - 107 * green - 21 * blue + 128) >> 8));
    return result;
}

RgbaPixel yuv_to_rgb(const YuvPixel &pixel) {
    const RK_S32 y = pixel.y;
    const RK_S32 u = static_cast<RK_S32>(pixel.u) - 128;
    const RK_S32 v = static_cast<RK_S32>(pixel.v) - 128;
    RgbaPixel result{};
    result.red = clamp_byte(y + ((359 * v + 128) >> 8));
    result.green = clamp_byte(y - ((88 * u + 183 * v + 128) >> 8));
    result.blue = clamp_byte(y + ((454 * u + 128) >> 8));
    result.alpha = 255;
    return result;
}

RgbaPixel color_to_rgba(RK_U32 color) {
    return {static_cast<RK_U8>((color >> 16U) & 0xffU),
            static_cast<RK_U8>((color >> 8U) & 0xffU),
            static_cast<RK_U8>(color & 0xffU),
            255};
}

YuvPixel load_yuv(const FrameView &view, RK_U32 x, RK_U32 y) {
    const FormatLayout &layout = view.layout;
    if (layout.storage == StorageKind::Packed) {
        const RK_U8 *pixel =
            view.data + (static_cast<RK_U64>(y) * view.virtualWidth + x) * layout.packedBytes;
        if (view.format == RK_FMT_YUV400SP) {
            return {pixel[0], 128, 128};
        }
        if (view.format == RK_FMT_YUV444) {
            return {pixel[0], pixel[1], pixel[2]};
        }
        return rgb_to_yuv({});
    }
    if (layout.storage == StorageKind::Packed422) {
        const RK_U32 pair = x / 2U;
        const RK_U8 *bytes =
            view.data + static_cast<RK_U64>(y) * ceil_div(view.virtualWidth, 2U) * 4U + pair * 4U;
        RK_U8 y0 = 0;
        RK_U8 y1 = 0;
        RK_U8 u = 128;
        RK_U8 v = 128;
        switch (view.format) {
        case RK_FMT_YUV422_YUYV:
            y0 = bytes[0];
            u = bytes[1];
            y1 = bytes[2];
            v = bytes[3];
            break;
        case RK_FMT_YUV422_UYVY:
            u = bytes[0];
            y0 = bytes[1];
            v = bytes[2];
            y1 = bytes[3];
            break;
        case RK_FMT_YUV422_YVYU:
            y0 = bytes[0];
            v = bytes[1];
            y1 = bytes[2];
            u = bytes[3];
            break;
        case RK_FMT_YUV422_VYUY:
            v = bytes[0];
            y0 = bytes[1];
            u = bytes[2];
            y1 = bytes[3];
            break;
        default:
            break;
        }
        return {(x & 1U) ? y1 : y0, u, v};
    }
    const RK_U64 yPlaneBytes = static_cast<RK_U64>(view.virtualWidth) * view.virtualHeight;
    const RK_U8 yValue = view.data[static_cast<RK_U64>(y) * view.virtualWidth + x];
    const RK_U32 chromaX = x / layout.chromaSubX;
    const RK_U32 chromaY = y / layout.chromaSubY;
    const RK_U32 chromaWidth = ceil_div(view.virtualWidth, layout.chromaSubX);
    if (layout.storage == StorageKind::SemiPlanar) {
        const RK_U64 offset =
            yPlaneBytes + (static_cast<RK_U64>(chromaY) * chromaWidth + chromaX) * 2U;
        const RK_U8 first = view.data[offset];
        const RK_U8 second = view.data[offset + 1U];
        return {yValue, layout.vuOrder ? second : first, layout.vuOrder ? first : second};
    }
    const RK_U32 chromaHeight = ceil_div(view.virtualHeight, layout.chromaSubY);
    const RK_U64 onePlane = static_cast<RK_U64>(chromaWidth) * chromaHeight;
    const RK_U64 chromaOffset = static_cast<RK_U64>(chromaY) * chromaWidth + chromaX;
    const RK_U8 first = view.data[yPlaneBytes + chromaOffset];
    const RK_U8 second = view.data[yPlaneBytes + onePlane + chromaOffset];
    return {yValue, layout.vuOrder ? second : first, layout.vuOrder ? first : second};
}

RgbaPixel load_rgba(const FrameView &view, RK_U32 x, RK_U32 y) {
    if (view.layout.yuv) {
        return yuv_to_rgb(load_yuv(view, x, y));
    }
    const RK_U8 *pixel =
        view.data + (static_cast<RK_U64>(y) * view.virtualWidth + x) * view.layout.packedBytes;
    switch (view.format) {
        ROCKIT_PACKED_RGBA_PIXEL_CASES;
    default:
        return {};
    }
}

#define ROCKIT_DEFINE_PIXEL_DECODER(name_, pixel_type_, reject_, load_)                            \
    bool name_(const FrameView &view, std::vector<pixel_type_> *pixels) {                          \
        if (reject_) {                                                                             \
            return false;                                                                          \
        }                                                                                          \
        RK_U64 count = 0;                                                                          \
        if (!checked_mul(view.width, view.height, &count) ||                                       \
            count > std::numeric_limits<std::size_t>::max()) {                                     \
            return false;                                                                          \
        }                                                                                          \
        try {                                                                                      \
            pixels->resize(static_cast<std::size_t>(count));                                       \
        } catch (...) {                                                                            \
            return false;                                                                          \
        }                                                                                          \
        for (RK_U32 y = 0; y < view.height; ++y) {                                                 \
            for (RK_U32 x = 0; x < view.width; ++x) {                                              \
                (*pixels)[static_cast<std::size_t>(y) * view.width + x] = load_(view, x, y);       \
            }                                                                                      \
        }                                                                                          \
        return true;                                                                               \
    }

ROCKIT_DEFINE_PIXEL_DECODER(decode_yuv, YuvPixel, !pixels || !view.layout.yuv, load_yuv)
ROCKIT_DEFINE_PIXEL_DECODER(decode_rgba, RgbaPixel, !pixels, load_rgba)

#undef ROCKIT_DEFINE_PIXEL_DECODER

bool encode_yuv(const std::vector<YuvPixel> &pixels,
                RK_U32 width,
                RK_U32 height,
                RK_U32 virtualWidth,
                RK_U32 virtualHeight,
                PIXEL_FORMAT_E format,
                std::vector<RK_U8> *bytes) {
    if (!bytes || width == 0 || height == 0 || virtualWidth < width || virtualHeight < height ||
        pixels.size() != static_cast<std::size_t>(width) * height) {
        return false;
    }
    FormatLayout layout{};
    RK_U64 required = 0;
    if (!format_layout(format, &layout) || !layout.yuv ||
        !storage_size(layout, virtualWidth, virtualHeight, &required) ||
        required > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    try {
        bytes->assign(static_cast<std::size_t>(required), 0);
    } catch (...) {
        return false;
    }
    if (layout.storage == StorageKind::Packed && format == RK_FMT_YUV400SP) {
        for (RK_U32 y = 0; y < height; ++y) {
            for (RK_U32 x = 0; x < width; ++x) {
                (*bytes)[static_cast<std::size_t>(y) * virtualWidth + x] =
                    pixels[static_cast<std::size_t>(y) * width + x].y;
            }
        }
        return true;
    }
    if (layout.storage == StorageKind::Packed && format == RK_FMT_YUV444) {
        for (RK_U32 y = 0; y < height; ++y) {
            for (RK_U32 x = 0; x < width; ++x) {
                const YuvPixel value = pixels[static_cast<std::size_t>(y) * width + x];
                RK_U8 *destination =
                    bytes->data() + (static_cast<std::size_t>(y) * virtualWidth + x) * 3U;
                destination[0] = value.y;
                destination[1] = value.u;
                destination[2] = value.v;
            }
        }
        return true;
    }
    if (layout.storage == StorageKind::Packed422) {
        const RK_U32 pairStride = ceil_div(virtualWidth, 2U) * 4U;
        for (RK_U32 y = 0; y < height; ++y) {
            for (RK_U32 x = 0; x < width; x += 2U) {
                const YuvPixel first = pixels[static_cast<std::size_t>(y) * width + x];
                const YuvPixel second = pixels[static_cast<std::size_t>(y) * width +
                                               std::min<RK_U32>(x + 1U, width - 1U)];
                const RK_U8 u =
                    static_cast<RK_U8>((static_cast<RK_U32>(first.u) + second.u + 1U) / 2U);
                const RK_U8 v =
                    static_cast<RK_U8>((static_cast<RK_U32>(first.v) + second.v + 1U) / 2U);
                RK_U8 *destination =
                    bytes->data() + static_cast<std::size_t>(y) * pairStride + (x / 2U) * 4U;
                switch (format) {
                case RK_FMT_YUV422_YUYV:
                    destination[0] = first.y;
                    destination[1] = u;
                    destination[2] = second.y;
                    destination[3] = v;
                    break;
                case RK_FMT_YUV422_UYVY:
                    destination[0] = u;
                    destination[1] = first.y;
                    destination[2] = v;
                    destination[3] = second.y;
                    break;
                case RK_FMT_YUV422_YVYU:
                    destination[0] = first.y;
                    destination[1] = v;
                    destination[2] = second.y;
                    destination[3] = u;
                    break;
                case RK_FMT_YUV422_VYUY:
                    destination[0] = v;
                    destination[1] = first.y;
                    destination[2] = u;
                    destination[3] = second.y;
                    break;
                default:
                    return false;
                }
            }
        }
        return true;
    }

    const RK_U64 yPlaneBytes = static_cast<RK_U64>(virtualWidth) * virtualHeight;
    for (RK_U32 y = 0; y < height; ++y) {
        for (RK_U32 x = 0; x < width; ++x) {
            (*bytes)[static_cast<std::size_t>(y) * virtualWidth + x] =
                pixels[static_cast<std::size_t>(y) * width + x].y;
        }
    }
    const RK_U32 chromaWidth = ceil_div(virtualWidth, layout.chromaSubX);
    const RK_U32 chromaHeight = ceil_div(virtualHeight, layout.chromaSubY);
    const RK_U64 onePlane = static_cast<RK_U64>(chromaWidth) * chromaHeight;
    for (RK_U32 chromaY = 0; chromaY < ceil_div(height, layout.chromaSubY); ++chromaY) {
        for (RK_U32 chromaX = 0; chromaX < ceil_div(width, layout.chromaSubX); ++chromaX) {
            RK_U32 sumU = 0;
            RK_U32 sumV = 0;
            RK_U32 count = 0;
            const RK_U32 startX = chromaX * layout.chromaSubX;
            const RK_U32 startY = chromaY * layout.chromaSubY;
            for (RK_U32 dy = 0; dy < layout.chromaSubY && startY + dy < height; ++dy) {
                for (RK_U32 dx = 0; dx < layout.chromaSubX && startX + dx < width; ++dx) {
                    const YuvPixel value =
                        pixels[static_cast<std::size_t>(startY + dy) * width + startX + dx];
                    sumU += value.u;
                    sumV += value.v;
                    ++count;
                }
            }
            const RK_U8 u = static_cast<RK_U8>((sumU + count / 2U) / count);
            const RK_U8 v = static_cast<RK_U8>((sumV + count / 2U) / count);
            const RK_U64 chromaOffset = static_cast<RK_U64>(chromaY) * chromaWidth + chromaX;
            if (layout.storage == StorageKind::SemiPlanar) {
                RK_U8 *destination = bytes->data() + yPlaneBytes + chromaOffset * 2U;
                destination[0] = layout.vuOrder ? v : u;
                destination[1] = layout.vuOrder ? u : v;
            } else {
                (*bytes)[static_cast<std::size_t>(yPlaneBytes + chromaOffset)] =
                    layout.vuOrder ? v : u;
                (*bytes)[static_cast<std::size_t>(yPlaneBytes + onePlane + chromaOffset)] =
                    layout.vuOrder ? u : v;
            }
        }
    }
    return true;
}

bool encode_rgba(const std::vector<RgbaPixel> &pixels,
                 RK_U32 width,
                 RK_U32 height,
                 RK_U32 virtualWidth,
                 RK_U32 virtualHeight,
                 PIXEL_FORMAT_E format,
                 std::vector<RK_U8> *bytes) {
    if (!bytes || pixels.size() != static_cast<std::size_t>(width) * height) {
        return false;
    }
    FormatLayout layout{};
    if (!format_layout(format, &layout)) {
        return false;
    }
    if (layout.yuv) {
        std::vector<YuvPixel> converted(pixels.size());
        std::transform(pixels.begin(), pixels.end(), converted.begin(), rgb_to_yuv);
        return encode_yuv(converted, width, height, virtualWidth, virtualHeight, format, bytes);
    }
    RK_U64 required = 0;
    if (!storage_size(layout, virtualWidth, virtualHeight, &required) ||
        required > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    try {
        bytes->assign(static_cast<std::size_t>(required), 0);
    } catch (...) {
        return false;
    }
    for (RK_U32 y = 0; y < height; ++y) {
        for (RK_U32 x = 0; x < width; ++x) {
            const RgbaPixel value = pixels[static_cast<std::size_t>(y) * width + x];
            RK_U8 *destination = bytes->data() + (static_cast<std::size_t>(y) * virtualWidth + x) *
                                                     layout.packedBytes;
            switch (format) {
            case RK_FMT_RGB888:
                destination[0] = value.red;
                destination[1] = value.green;
                destination[2] = value.blue;
                break;
            case RK_FMT_BGR888:
                destination[0] = value.blue;
                destination[1] = value.green;
                destination[2] = value.red;
                break;
            case RK_FMT_RGBA8888:
                destination[0] = value.red;
                destination[1] = value.green;
                destination[2] = value.blue;
                destination[3] = value.alpha;
                break;
            case RK_FMT_BGRA8888:
                destination[0] = value.blue;
                destination[1] = value.green;
                destination[2] = value.red;
                destination[3] = value.alpha;
                break;
            case RK_FMT_ARGB8888:
                destination[0] = value.alpha;
                destination[1] = value.red;
                destination[2] = value.green;
                destination[3] = value.blue;
                break;
            case RK_FMT_ABGR8888:
            case RK_FMT_XBGR8888:
                destination[0] = value.alpha;
                destination[1] = value.blue;
                destination[2] = value.green;
                destination[3] = value.red;
                break;
            default:
                return false;
            }
        }
    }
    return true;
}

template <typename Pixel>
bool normalize_crop(const FrameTransform &transform,
                    RK_U32 inputWidth,
                    RK_U32 inputHeight,
                    RECT_S *crop) {
    (void)sizeof(Pixel);
    if (!crop) {
        return false;
    }
    if (!transform.cropEnabled) {
        *crop = {0, 0, inputWidth, inputHeight};
        return true;
    }
    const RECT_S &candidate = transform.crop;
    if (candidate.s32X < 0 || candidate.s32Y < 0 || candidate.u32Width == 0 ||
        candidate.u32Height == 0) {
        return false;
    }
    const RK_U64 right =
        static_cast<RK_U64>(static_cast<RK_U32>(candidate.s32X)) + candidate.u32Width;
    const RK_U64 bottom =
        static_cast<RK_U64>(static_cast<RK_U32>(candidate.s32Y)) + candidate.u32Height;
    if (right > inputWidth || bottom > inputHeight) {
        return false;
    }
    *crop = candidate;
    return true;
}

template <typename Pixel>
bool scale_crop(const std::vector<Pixel> &source,
                RK_U32 sourceWidth,
                RK_U32 sourceHeight,
                const RECT_S &crop,
                RK_U32 outputWidth,
                RK_U32 outputHeight,
                std::vector<Pixel> *destination) {
    if (!destination || source.size() != static_cast<std::size_t>(sourceWidth) * sourceHeight ||
        outputWidth == 0 || outputHeight == 0) {
        return false;
    }
    RK_U64 count = 0;
    if (!checked_mul(outputWidth, outputHeight, &count) ||
        count > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    try {
        destination->resize(static_cast<std::size_t>(count));
    } catch (...) {
        return false;
    }
    for (RK_U32 y = 0; y < outputHeight; ++y) {
        const RK_U32 sourceY =
            static_cast<RK_U32>(crop.s32Y) +
            static_cast<RK_U32>((static_cast<RK_U64>(y) * crop.u32Height) / outputHeight);
        for (RK_U32 x = 0; x < outputWidth; ++x) {
            const RK_U32 sourceX =
                static_cast<RK_U32>(crop.s32X) +
                static_cast<RK_U32>((static_cast<RK_U64>(x) * crop.u32Width) / outputWidth);
            (*destination)[static_cast<std::size_t>(y) * outputWidth + x] =
                source[static_cast<std::size_t>(sourceY) * sourceWidth + sourceX];
        }
    }
    return true;
}

template <typename Pixel>
void mirror_flip(std::vector<Pixel> *pixels, RK_U32 width, RK_U32 height, bool mirror, bool flip) {
    if (!pixels || (!mirror && !flip)) {
        return;
    }
    const std::vector<Pixel> source = *pixels;
    for (RK_U32 y = 0; y < height; ++y) {
        const RK_U32 sourceY = flip ? height - 1U - y : y;
        for (RK_U32 x = 0; x < width; ++x) {
            const RK_U32 sourceX = mirror ? width - 1U - x : x;
            (*pixels)[static_cast<std::size_t>(y) * width + x] =
                source[static_cast<std::size_t>(sourceY) * width + sourceX];
        }
    }
}

template <typename Pixel>
bool rotate_pixels(std::vector<Pixel> *pixels, RK_U32 *width, RK_U32 *height, ROTATION_E rotation) {
    if (!pixels || !width || !height || rotation < ROTATION_0 || rotation >= ROTATION_BUTT) {
        return false;
    }
    if (rotation == ROTATION_0) {
        return true;
    }
    const RK_U32 sourceWidth = *width;
    const RK_U32 sourceHeight = *height;
    const RK_U32 destinationWidth =
        (rotation == ROTATION_90 || rotation == ROTATION_270) ? sourceHeight : sourceWidth;
    const RK_U32 destinationHeight =
        (rotation == ROTATION_90 || rotation == ROTATION_270) ? sourceWidth : sourceHeight;
    const std::vector<Pixel> source = *pixels;
    std::vector<Pixel> destination(source.size());
    for (RK_U32 y = 0; y < destinationHeight; ++y) {
        for (RK_U32 x = 0; x < destinationWidth; ++x) {
            RK_U32 sourceX = x;
            RK_U32 sourceY = y;
            switch (rotation) {
            case ROTATION_90:
                sourceX = y;
                sourceY = sourceHeight - 1U - x;
                break;
            case ROTATION_180:
                sourceX = sourceWidth - 1U - x;
                sourceY = sourceHeight - 1U - y;
                break;
            case ROTATION_270:
                sourceX = sourceWidth - 1U - y;
                sourceY = x;
                break;
            default:
                break;
            }
            destination[static_cast<std::size_t>(y) * destinationWidth + x] =
                source[static_cast<std::size_t>(sourceY) * sourceWidth + sourceX];
        }
    }
    *pixels = std::move(destination);
    *width = destinationWidth;
    *height = destinationHeight;
    return true;
}

template <typename Pixel>
bool apply_transform(std::vector<Pixel> *pixels,
                     RK_U32 *width,
                     RK_U32 *height,
                     const FrameTransform &transform,
                     bool allowCropAndScale) {
    if (!pixels || !width || !height) {
        return false;
    }
    if (allowCropAndScale &&
        (transform.cropEnabled || transform.outputWidth != 0 || transform.outputHeight != 0)) {
        RECT_S crop{};
        if (!normalize_crop<Pixel>(transform, *width, *height, &crop)) {
            return false;
        }
        const RK_U32 targetWidth = transform.outputWidth ? transform.outputWidth : crop.u32Width;
        const RK_U32 targetHeight =
            transform.outputHeight ? transform.outputHeight : crop.u32Height;
        std::vector<Pixel> scaled;
        if (!scale_crop(*pixels, *width, *height, crop, targetWidth, targetHeight, &scaled)) {
            return false;
        }
        *pixels = std::move(scaled);
        *width = targetWidth;
        *height = targetHeight;
    }
    mirror_flip(pixels, *width, *height, transform.mirror == RK_TRUE, transform.flip == RK_TRUE);
    return rotate_pixels(pixels, width, height, transform.rotation);
}

RK_S32 allocate_output(const VIDEO_FRAME_INFO_S &metadata,
                       RK_U32 width,
                       RK_U32 height,
                       std::vector<RK_U8> &&pixels,
                       VIDEO_FRAME_INFO_S *output) {
    if (!output || width == 0 || height == 0 || pixels.empty() ||
        pixels.size() > std::numeric_limits<RK_U32>::max()) {
        return RK_FAILURE;
    }
    MB_BLK block = MB_INVALID_HANDLE;
    if (RK_MPI_SYS_MmzAlloc(&block, "frame-cpu", nullptr, static_cast<RK_U32>(pixels.size())) !=
        RK_SUCCESS) {
        return RK_FAILURE;
    }
    void *address = RK_MPI_MB_Handle2VirAddr(block);
    if (!address) {
        (void)RK_MPI_MB_ReleaseMB(block);
        return RK_FAILURE;
    }
    std::memcpy(address, pixels.data(), pixels.size());
    *output = metadata;
    output->stVFrame.pMbBlk = block;
    output->stVFrame.u32Width = width;
    output->stVFrame.u32Height = height;
    output->stVFrame.u32VirWidth = width;
    output->stVFrame.u32VirHeight = height;
    output->stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    return RK_SUCCESS;
}

bool write_rgba_back(const FrameView &view, const std::vector<RgbaPixel> &pixels) {
    std::vector<RK_U8> encoded;
    if (!encode_rgba(pixels,
                     view.width,
                     view.height,
                     view.virtualWidth,
                     view.virtualHeight,
                     view.format,
                     &encoded) ||
        encoded.size() > view.bytes) {
        return false;
    }
    std::memcpy(view.data, encoded.data(), encoded.size());
    return true;
}

bool valid_rect(const RECT_S &rect, RK_U32 width, RK_U32 height) {
    if (rect.s32X < 0 || rect.s32Y < 0 || rect.u32Width == 0 || rect.u32Height == 0) {
        return false;
    }
    return static_cast<RK_U64>(static_cast<RK_U32>(rect.s32X)) + rect.u32Width <= width &&
           static_cast<RK_U64>(static_cast<RK_U32>(rect.s32Y)) + rect.u32Height <= height;
}

RgbaPixel load_overlay_source(const RK_U8 *pixel, PIXEL_FORMAT_E format) {
    switch (format) {
    case RK_FMT_RGBA8888:
        return {pixel[0], pixel[1], pixel[2], pixel[3]};
    case RK_FMT_BGRA8888:
        return {pixel[2], pixel[1], pixel[0], pixel[3]};
    case RK_FMT_ARGB8888:
        return {pixel[1], pixel[2], pixel[3], pixel[0]};
    case RK_FMT_ABGR8888:
    case RK_FMT_XBGR8888:
        return {pixel[3], pixel[2], pixel[1], pixel[0]};
    case RK_FMT_RGB888:
        return {pixel[0], pixel[1], pixel[2], 255};
    case RK_FMT_BGR888:
        return {pixel[2], pixel[1], pixel[0], 255};
    default:
        return {};
    }
}

RK_U32 overlay_source_bpp(PIXEL_FORMAT_E format) {
    switch (format) {
    case RK_FMT_RGB888:
    case RK_FMT_BGR888:
        return 3;
    case RK_FMT_RGBA8888:
    case RK_FMT_BGRA8888:
    case RK_FMT_ARGB8888:
    case RK_FMT_ABGR8888:
    case RK_FMT_XBGR8888:
        return 4;
    default:
        return 0;
    }
}

RK_U8 blend_channel(RK_U8 foreground, RK_U8 background, RK_U32 alpha) {
    return static_cast<RK_U8>((static_cast<RK_U32>(foreground) * alpha +
                               static_cast<RK_U32>(background) * (255U - alpha) + 127U) /
                              255U);
}

} // namespace

RK_U32 cpu_frame_bytes_per_pixel(PIXEL_FORMAT_E format) {
    FormatLayout layout{};
    if (!format_layout(format, &layout) || layout.storage != StorageKind::Packed || layout.yuv) {
        return format == RK_FMT_YUV400SP ? 1U : 0U;
    }
    return layout.packedBytes;
}

RK_U32 cpu_frame_supported_format_count() {
    return 22;
}

bool cpu_frame_is_supported(const VIDEO_FRAME_INFO_S &frame) {
    FrameView view{};
    return view_from_frame(frame, &view);
}

RK_S32 cpu_extract_luma(const VIDEO_FRAME_INFO_S &frame,
                        std::vector<RK_U8> *luma,
                        RK_U32 *width,
                        RK_U32 *height) {
    if (!luma || !width || !height) {
        return RK_FAILURE;
    }
    FrameView source{};
    if (!view_from_frame(frame, &source)) {
        return RK_FAILURE;
    }
    RK_U64 count64 = 0;
    if (!checked_mul(source.width, source.height, &count64) ||
        count64 > std::numeric_limits<std::size_t>::max()) {
        return RK_FAILURE;
    }
    luma->clear();
    luma->reserve(static_cast<std::size_t>(count64));
    if (source.layout.yuv) {
        std::vector<YuvPixel> pixels;
        if (!decode_yuv(source, &pixels)) {
            return RK_FAILURE;
        }
        for (const auto &pixel : pixels) {
            luma->push_back(pixel.y);
        }
    } else {
        std::vector<RgbaPixel> pixels;
        if (!decode_rgba(source, &pixels)) {
            return RK_FAILURE;
        }
        for (const auto &pixel : pixels) {
            luma->push_back(rgb_to_yuv(pixel).y);
        }
    }
    *width = source.width;
    *height = source.height;
    return RK_SUCCESS;
}

RK_S32 cpu_transform_frame(const VIDEO_FRAME_INFO_S &input,
                           const FrameTransform &first,
                           const FrameTransform *second,
                           VIDEO_FRAME_INFO_S *output) {
    if (!output) {
        return RK_FAILURE;
    }
    FrameView source{};
    if (!view_from_frame(input, &source)) {
        return RK_FAILURE;
    }
    RECT_S crop{};
    if (!normalize_crop<RgbaPixel>(first, source.width, source.height, &crop)) {
        return RK_FAILURE;
    }
    const RK_U32 initialWidth = first.outputWidth ? first.outputWidth : crop.u32Width;
    const RK_U32 initialHeight = first.outputHeight ? first.outputHeight : crop.u32Height;
    RK_U32 width = initialWidth;
    RK_U32 height = initialHeight;
    FrameTransform firstPost = first;
    firstPost.cropEnabled = RK_FALSE;
    firstPost.outputWidth = 0;
    firstPost.outputHeight = 0;

#define ROCKIT_TRANSFORM_PIXELS(pixel_type_, decode_, encode_)                                     \
    std::vector<pixel_type_> original;                                                            \
    std::vector<pixel_type_> pixels;                                                              \
    if (!decode_(source, &original) ||                                                            \
        !scale_crop(original,                                                                      \
                    source.width,                                                                  \
                    source.height,                                                                 \
                    crop,                                                                          \
                    initialWidth,                                                                  \
                    initialHeight,                                                                 \
                    &pixels) ||                                                                    \
        !apply_transform(&pixels, &width, &height, firstPost, false) ||                            \
        (second && !apply_transform(&pixels, &width, &height, *second, true)) ||                   \
        !encode_(pixels, width, height, width, height, source.format, &encoded))                   \
        return RK_FAILURE

    std::vector<RK_U8> encoded;
    if (source.layout.yuv) {
        ROCKIT_TRANSFORM_PIXELS(YuvPixel, decode_yuv, encode_yuv);
    } else {
        ROCKIT_TRANSFORM_PIXELS(RgbaPixel, decode_rgba, encode_rgba);
    }
#undef ROCKIT_TRANSFORM_PIXELS
    return allocate_output(input, width, height, std::move(encoded), output);
}

RK_S32
cpu_solid_frame(const VIDEO_FRAME_INFO_S &metadata, RK_U32 color, VIDEO_FRAME_INFO_S *output) {
    if (!output) {
        return RK_FAILURE;
    }
    const RK_U32 width = metadata.stVFrame.u32Width;
    const RK_U32 height = metadata.stVFrame.u32Height;
    FormatLayout layout{};
    if (width == 0 || height == 0 || metadata.stVFrame.enCompressMode != COMPRESS_MODE_NONE ||
        !format_layout(metadata.stVFrame.enPixelFormat, &layout)) {
        return RK_FAILURE;
    }
    RK_U64 count = 0;
    if (!checked_mul(width, height, &count) || count > std::numeric_limits<std::size_t>::max()) {
        return RK_FAILURE;
    }
    std::vector<RgbaPixel> pixels(static_cast<std::size_t>(count), color_to_rgba(color));
    std::vector<RK_U8> encoded;
    if (!encode_rgba(
            pixels, width, height, width, height, metadata.stVFrame.enPixelFormat, &encoded)) {
        return RK_FAILURE;
    }
    return allocate_output(metadata, width, height, std::move(encoded), output);
}

RK_S32 cpu_compose_frames(const std::vector<VIDEO_FRAME_INFO_S> &inputs,
                          const std::vector<RECT_S> &destinations,
                          const VIDEO_FRAME_INFO_S &metadata,
                          bool averageOverlap,
                          VIDEO_FRAME_INFO_S *output) {
    if (!output || inputs.empty() || inputs.size() != destinations.size()) {
        return RK_FAILURE;
    }
    const RK_U32 width = metadata.stVFrame.u32Width;
    const RK_U32 height = metadata.stVFrame.u32Height;
    FormatLayout outputLayout{};
    RK_U64 pixelCount64 = 0;
    if (width == 0 || height == 0 || metadata.stVFrame.enCompressMode != COMPRESS_MODE_NONE ||
        !format_layout(metadata.stVFrame.enPixelFormat, &outputLayout) ||
        !checked_mul(width, height, &pixelCount64) ||
        pixelCount64 > std::numeric_limits<std::size_t>::max()) {
        return RK_FAILURE;
    }
    const std::size_t pixelCount = static_cast<std::size_t>(pixelCount64);
    std::vector<RgbaPixel> canvas(pixelCount, RgbaPixel{0, 0, 0, 255});
    std::vector<RK_U64> sumRed;
    std::vector<RK_U64> sumGreen;
    std::vector<RK_U64> sumBlue;
    std::vector<RK_U64> sumAlpha;
    std::vector<RK_U32> counts;
    if (averageOverlap) {
        try {
            sumRed.assign(pixelCount, 0);
            sumGreen.assign(pixelCount, 0);
            sumBlue.assign(pixelCount, 0);
            sumAlpha.assign(pixelCount, 0);
            counts.assign(pixelCount, 0);
        } catch (...) {
            return RK_FAILURE;
        }
    }
    for (std::size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
        const RECT_S &destination = destinations[inputIndex];
        if (!valid_rect(destination, width, height)) {
            return RK_FAILURE;
        }
        FrameView view{};
        std::vector<RgbaPixel> source;
        if (!view_from_frame(inputs[inputIndex], &view) || !decode_rgba(view, &source)) {
            return RK_FAILURE;
        }
        RECT_S full{};
        full.u32Width = view.width;
        full.u32Height = view.height;
        std::vector<RgbaPixel> scaled;
        if (!scale_crop(source,
                        view.width,
                        view.height,
                        full,
                        destination.u32Width,
                        destination.u32Height,
                        &scaled)) {
            return RK_FAILURE;
        }
        for (RK_U32 y = 0; y < destination.u32Height; ++y) {
            for (RK_U32 x = 0; x < destination.u32Width; ++x) {
                const std::size_t sourceOffset =
                    static_cast<std::size_t>(y) * destination.u32Width + x;
                const std::size_t destinationOffset =
                    static_cast<std::size_t>(static_cast<RK_U32>(destination.s32Y) + y) * width +
                    static_cast<RK_U32>(destination.s32X) + x;
                const RgbaPixel pixel = scaled[sourceOffset];
                if (averageOverlap) {
                    sumRed[destinationOffset] += pixel.red;
                    sumGreen[destinationOffset] += pixel.green;
                    sumBlue[destinationOffset] += pixel.blue;
                    sumAlpha[destinationOffset] += pixel.alpha;
                    ++counts[destinationOffset];
                } else {
                    canvas[destinationOffset] = pixel;
                }
            }
        }
    }
    if (averageOverlap) {
        for (std::size_t index = 0; index < pixelCount; ++index) {
            if (counts[index] == 0)
                continue;
            const RK_U32 count = counts[index];
            canvas[index] = {static_cast<RK_U8>((sumRed[index] + count / 2U) / count),
                             static_cast<RK_U8>((sumGreen[index] + count / 2U) / count),
                             static_cast<RK_U8>((sumBlue[index] + count / 2U) / count),
                             static_cast<RK_U8>((sumAlpha[index] + count / 2U) / count)};
        }
    }
    std::vector<RK_U8> encoded;
    if (!encode_rgba(
            canvas, width, height, width, height, metadata.stVFrame.enPixelFormat, &encoded)) {
        return RK_FAILURE;
    }
    return allocate_output(metadata, width, height, std::move(encoded), output);
}

RK_S32 cpu_copy_frame_to_existing(const VIDEO_FRAME_INFO_S &source,
                                  VIDEO_FRAME_INFO_S *destination) {
    if (!destination) {
        return RK_FAILURE;
    }
    FrameView sourceView{};
    FrameView destinationView{};
    if (!view_from_frame(source, &sourceView) || !view_from_frame(*destination, &destinationView) ||
        sourceView.width != destinationView.width || sourceView.height != destinationView.height ||
        sourceView.format != destinationView.format ||
        sourceView.layout.storage != destinationView.layout.storage) {
        return RK_FAILURE;
    }

    const FormatLayout &layout = sourceView.layout;
    auto copy_rows = [](const RK_U8 *sourceData,
                        RK_U64 sourceStride,
                        RK_U8 *destinationData,
                        RK_U64 destinationStride,
                        RK_U32 rows,
                        RK_U64 visibleBytes) {
        for (RK_U32 row = 0; row < rows; ++row) {
            std::memcpy(destinationData + static_cast<RK_U64>(row) * destinationStride,
                        sourceData + static_cast<RK_U64>(row) * sourceStride,
                        static_cast<std::size_t>(visibleBytes));
        }
    };

#define ROCKIT_COPY_LUMA_AND_DEFINE_CHROMA_WIDTHS()                                                \
    copy_rows(sourceView.data,                                                                     \
              sourceView.virtualWidth,                                                             \
              destinationView.data,                                                                \
              destinationView.virtualWidth,                                                        \
              sourceView.height,                                                                   \
              sourceView.width);                                                                   \
    const RK_U64 sourceYBytes =                                                                    \
        static_cast<RK_U64>(sourceView.virtualWidth) * sourceView.virtualHeight;                   \
    const RK_U64 destinationYBytes =                                                               \
        static_cast<RK_U64>(destinationView.virtualWidth) * destinationView.virtualHeight;         \
    const RK_U32 sourceChromaWidth = ceil_div(sourceView.virtualWidth, layout.chromaSubX);         \
    const RK_U32 destinationChromaWidth = ceil_div(destinationView.virtualWidth, layout.chromaSubX)

    switch (layout.storage) {
    case StorageKind::Packed: {
        const RK_U64 visibleBytes = static_cast<RK_U64>(sourceView.width) * layout.packedBytes;
        copy_rows(sourceView.data,
                  static_cast<RK_U64>(sourceView.virtualWidth) * layout.packedBytes,
                  destinationView.data,
                  static_cast<RK_U64>(destinationView.virtualWidth) * layout.packedBytes,
                  sourceView.height,
                  visibleBytes);
        return RK_SUCCESS;
    }
    case StorageKind::Packed422: {
        const RK_U64 visibleBytes = static_cast<RK_U64>(ceil_div(sourceView.width, 2U)) * 4U;
        copy_rows(sourceView.data,
                  static_cast<RK_U64>(ceil_div(sourceView.virtualWidth, 2U)) * 4U,
                  destinationView.data,
                  static_cast<RK_U64>(ceil_div(destinationView.virtualWidth, 2U)) * 4U,
                  sourceView.height,
                  visibleBytes);
        return RK_SUCCESS;
    }
    case StorageKind::SemiPlanar: {
        ROCKIT_COPY_LUMA_AND_DEFINE_CHROMA_WIDTHS();
        const RK_U32 visibleChromaWidth = ceil_div(sourceView.width, layout.chromaSubX);
        const RK_U32 visibleChromaHeight = ceil_div(sourceView.height, layout.chromaSubY);
        copy_rows(sourceView.data + sourceYBytes,
                  static_cast<RK_U64>(sourceChromaWidth) * 2U,
                  destinationView.data + destinationYBytes,
                  static_cast<RK_U64>(destinationChromaWidth) * 2U,
                  visibleChromaHeight,
                  static_cast<RK_U64>(visibleChromaWidth) * 2U);
        return RK_SUCCESS;
    }
    case StorageKind::Planar: {
        ROCKIT_COPY_LUMA_AND_DEFINE_CHROMA_WIDTHS();
        const RK_U32 sourceChromaHeight = ceil_div(sourceView.virtualHeight, layout.chromaSubY);
        const RK_U32 destinationChromaHeight =
            ceil_div(destinationView.virtualHeight, layout.chromaSubY);
        const RK_U32 visibleChromaWidth = ceil_div(sourceView.width, layout.chromaSubX);
        const RK_U32 visibleChromaHeight = ceil_div(sourceView.height, layout.chromaSubY);
        for (RK_U32 plane = 0; plane < 2; ++plane) {
            const RK_U8 *sourcePlane =
                sourceView.data + sourceYBytes +
                static_cast<RK_U64>(plane) * sourceChromaWidth * sourceChromaHeight;
            RK_U8 *destinationPlane =
                destinationView.data + destinationYBytes +
                static_cast<RK_U64>(plane) * destinationChromaWidth * destinationChromaHeight;
            copy_rows(sourcePlane,
                      sourceChromaWidth,
                      destinationPlane,
                      destinationChromaWidth,
                      visibleChromaHeight,
                      visibleChromaWidth);
        }
        return RK_SUCCESS;
    }
    }
#undef ROCKIT_COPY_LUMA_AND_DEFINE_CHROMA_WIDTHS
    return RK_FAILURE;
}

RK_S32 cpu_blit_frame(const VIDEO_FRAME_INFO_S &source,
                      const RECT_S &sourceRect,
                      VIDEO_FRAME_INFO_S *destination,
                      const RECT_S &destinationRect) {
    if (!destination) {
        return RK_FAILURE;
    }
    FrameView sourceView{};
    FrameView destinationView{};
    std::vector<RgbaPixel> sourcePixels;
    std::vector<RgbaPixel> destinationPixels;
    if (!view_from_frame(source, &sourceView) || !view_from_frame(*destination, &destinationView) ||
        sourceView.format != destinationView.format ||
        !valid_rect(sourceRect, sourceView.width, sourceView.height) ||
        !valid_rect(destinationRect, destinationView.width, destinationView.height) ||
        !decode_rgba(sourceView, &sourcePixels) ||
        !decode_rgba(destinationView, &destinationPixels)) {
        return RK_FAILURE;
    }
    std::vector<RgbaPixel> scaled;
    if (!scale_crop(sourcePixels,
                    sourceView.width,
                    sourceView.height,
                    sourceRect,
                    destinationRect.u32Width,
                    destinationRect.u32Height,
                    &scaled)) {
        return RK_FAILURE;
    }
    for (RK_U32 y = 0; y < destinationRect.u32Height; ++y) {
        for (RK_U32 x = 0; x < destinationRect.u32Width; ++x) {
            const std::size_t sourceOffset =
                static_cast<std::size_t>(y) * destinationRect.u32Width + x;
            const std::size_t destinationOffset =
                static_cast<std::size_t>(static_cast<RK_U32>(destinationRect.s32Y) + y) *
                    destinationView.width +
                static_cast<RK_U32>(destinationRect.s32X) + x;
            destinationPixels[destinationOffset] = scaled[sourceOffset];
        }
    }
    return write_rgba_back(destinationView, destinationPixels) ? RK_SUCCESS : RK_FAILURE;
}

RK_S32 cpu_fill_frame(VIDEO_FRAME_INFO_S *frame, const RECT_S &rect, RK_U32 color) {
    if (!frame) {
        return RK_FAILURE;
    }
    FrameView view{};
    std::vector<RgbaPixel> pixels;
    if (!view_from_frame(*frame, &view) || !valid_rect(rect, view.width, view.height) ||
        !decode_rgba(view, &pixels)) {
        return RK_FAILURE;
    }
    const RgbaPixel value = color_to_rgba(color);
    for (RK_U32 y = 0; y < rect.u32Height; ++y) {
        for (RK_U32 x = 0; x < rect.u32Width; ++x) {
            pixels[static_cast<std::size_t>(static_cast<RK_U32>(rect.s32Y) + y) * view.width +
                   static_cast<RK_U32>(rect.s32X) + x] = value;
        }
    }
    return write_rgba_back(view, pixels) ? RK_SUCCESS : RK_FAILURE;
}

RK_S32 cpu_draw_line_frame(VIDEO_FRAME_INFO_S *frame,
                           const POINT_S &start,
                           const POINT_S &end,
                           RK_U32 thickness,
                           RK_U32 color) {
    if (!frame || thickness == 0) {
        return RK_FAILURE;
    }
    FrameView view{};
    std::vector<RgbaPixel> pixels;
    if (!view_from_frame(*frame, &view) || !decode_rgba(view, &pixels)) {
        return RK_FAILURE;
    }
    const RgbaPixel value = color_to_rgba(color);
    RK_S32 x0 = start.s32X;
    RK_S32 y0 = start.s32Y;
    const RK_S32 x1 = end.s32X;
    const RK_S32 y1 = end.s32Y;
    const RK_S32 dx = std::abs(x1 - x0);
    const RK_S32 sx = x0 < x1 ? 1 : -1;
    const RK_S32 dy = -std::abs(y1 - y0);
    const RK_S32 sy = y0 < y1 ? 1 : -1;
    RK_S32 error = dx + dy;
    const RK_S32 radiusBefore = static_cast<RK_S32>((thickness - 1U) / 2U);
    const RK_S32 radiusAfter = static_cast<RK_S32>(thickness / 2U);
    for (;;) {
        for (RK_S32 oy = -radiusBefore; oy <= radiusAfter; ++oy) {
            for (RK_S32 ox = -radiusBefore; ox <= radiusAfter; ++ox) {
                const RK_S32 px = x0 + ox;
                const RK_S32 py = y0 + oy;
                if (px >= 0 && py >= 0 && static_cast<RK_U32>(px) < view.width &&
                    static_cast<RK_U32>(py) < view.height) {
                    pixels[static_cast<std::size_t>(py) * view.width + static_cast<RK_U32>(px)] =
                        value;
                }
            }
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const RK_S32 doubled = 2 * error;
        if (doubled >= dy) {
            error += dy;
            x0 += sx;
        }
        if (doubled <= dx) {
            error += dx;
            y0 += sy;
        }
    }
    return write_rgba_back(view, pixels) ? RK_SUCCESS : RK_FAILURE;
}

RK_S32 cpu_mosaic_frame(VIDEO_FRAME_INFO_S *frame, const RECT_S &rect, RK_U32 blockSize) {
    if (!frame || blockSize == 0) {
        return RK_FAILURE;
    }
    FrameView view{};
    std::vector<RgbaPixel> pixels;
    if (!view_from_frame(*frame, &view) || !valid_rect(rect, view.width, view.height) ||
        !decode_rgba(view, &pixels)) {
        return RK_FAILURE;
    }
    const RK_U32 baseX = static_cast<RK_U32>(rect.s32X);
    const RK_U32 baseY = static_cast<RK_U32>(rect.s32Y);
    for (RK_U32 blockY = 0; blockY < rect.u32Height; blockY += blockSize) {
        for (RK_U32 blockX = 0; blockX < rect.u32Width; blockX += blockSize) {
            const RK_U32 actualWidth = std::min(blockSize, rect.u32Width - blockX);
            const RK_U32 actualHeight = std::min(blockSize, rect.u32Height - blockY);
            RK_U64 sumRed = 0;
            RK_U64 sumGreen = 0;
            RK_U64 sumBlue = 0;
            RK_U64 sumAlpha = 0;
            const RK_U32 count = actualWidth * actualHeight;
            for (RK_U32 y = 0; y < actualHeight; ++y) {
                for (RK_U32 x = 0; x < actualWidth; ++x) {
                    const RgbaPixel value =
                        pixels[static_cast<std::size_t>(baseY + blockY + y) * view.width + baseX +
                               blockX + x];
                    sumRed += value.red;
                    sumGreen += value.green;
                    sumBlue += value.blue;
                    sumAlpha += value.alpha;
                }
            }
            const RgbaPixel average{static_cast<RK_U8>((sumRed + count / 2U) / count),
                                    static_cast<RK_U8>((sumGreen + count / 2U) / count),
                                    static_cast<RK_U8>((sumBlue + count / 2U) / count),
                                    static_cast<RK_U8>((sumAlpha + count / 2U) / count)};
            for (RK_U32 y = 0; y < actualHeight; ++y) {
                for (RK_U32 x = 0; x < actualWidth; ++x) {
                    pixels[static_cast<std::size_t>(baseY + blockY + y) * view.width + baseX +
                           blockX + x] = average;
                }
            }
        }
    }
    return write_rgba_back(view, pixels) ? RK_SUCCESS : RK_FAILURE;
}

RK_S32 cpu_overlay_frame(VIDEO_FRAME_INFO_S *frame,
                         const RK_U8 *source,
                         RK_U64 sourceBytes,
                         RK_U32 sourceWidth,
                         RK_U32 sourceHeight,
                         PIXEL_FORMAT_E sourceFormat,
                         const POINT_S &destinationPoint,
                         RK_U32 globalAlpha) {
    if (!frame || !source || sourceWidth == 0 || sourceHeight == 0) {
        return RK_FAILURE;
    }
    const RK_U32 sourceBpp = overlay_source_bpp(sourceFormat);
    RK_U64 required = 0;
    if (sourceBpp == 0 ||
        !checked_mul(static_cast<RK_U64>(sourceWidth) * sourceHeight, sourceBpp, &required) ||
        required > sourceBytes) {
        return RK_FAILURE;
    }
    FrameView view{};
    std::vector<RgbaPixel> pixels;
    if (!view_from_frame(*frame, &view) || !decode_rgba(view, &pixels)) {
        return RK_FAILURE;
    }
    const RK_U32 clampedGlobalAlpha = std::min<RK_U32>(globalAlpha, 255U);
    for (RK_U32 y = 0; y < sourceHeight; ++y) {
        const RK_S32 destinationY = destinationPoint.s32Y + static_cast<RK_S32>(y);
        if (destinationY < 0 || static_cast<RK_U32>(destinationY) >= view.height) {
            continue;
        }
        for (RK_U32 x = 0; x < sourceWidth; ++x) {
            const RK_S32 destinationX = destinationPoint.s32X + static_cast<RK_S32>(x);
            if (destinationX < 0 || static_cast<RK_U32>(destinationX) >= view.width) {
                continue;
            }
            const RgbaPixel foreground = load_overlay_source(
                source + (static_cast<std::size_t>(y) * sourceWidth + x) * sourceBpp, sourceFormat);
            const RK_U32 alpha =
                (static_cast<RK_U32>(foreground.alpha) * clampedGlobalAlpha + 127U) / 255U;
            if (alpha == 0) {
                continue;
            }
            RgbaPixel &background = pixels[static_cast<std::size_t>(destinationY) * view.width +
                                           static_cast<RK_U32>(destinationX)];
            background.red = blend_channel(foreground.red, background.red, alpha);
            background.green = blend_channel(foreground.green, background.green, alpha);
            background.blue = blend_channel(foreground.blue, background.blue, alpha);
            background.alpha = 255;
        }
    }
    return write_rgba_back(view, pixels) ? RK_SUCCESS : RK_FAILURE;
}

} // namespace rockit_local
