#pragma once

#include "rk_common.h"
#include "rk_comm_video.h"

#include <vector>

namespace rockit_local {

struct FrameTransform {
    RECT_S crop{};
    RK_BOOL cropEnabled{RK_FALSE};
    RK_U32 outputWidth{0};
    RK_U32 outputHeight{0};
    RK_BOOL mirror{RK_FALSE};
    RK_BOOL flip{RK_FALSE};
    ROTATION_E rotation{ROTATION_0};
};

// Returns bytes per pixel for byte-addressable packed formats. Planar and
// semi-planar formats return 0 and are described by cpu_frame_is_supported().
RK_U32 cpu_frame_bytes_per_pixel(PIXEL_FORMAT_E format);
RK_U32 cpu_frame_supported_format_count();
bool cpu_frame_is_supported(const VIDEO_FRAME_INFO_S &frame);
RK_S32 cpu_extract_luma(const VIDEO_FRAME_INFO_S &frame,
                        std::vector<RK_U8> *luma,
                        RK_U32 *width,
                        RK_U32 *height);
RK_S32 cpu_transform_frame(const VIDEO_FRAME_INFO_S &input,
                           const FrameTransform &first,
                           const FrameTransform *second,
                           VIDEO_FRAME_INFO_S *output);
RK_S32
cpu_solid_frame(const VIDEO_FRAME_INFO_S &metadata, RK_U32 color, VIDEO_FRAME_INFO_S *output);
RK_S32 cpu_compose_frames(const std::vector<VIDEO_FRAME_INFO_S> &inputs,
                          const std::vector<RECT_S> &destinations,
                          const VIDEO_FRAME_INFO_S &metadata,
                          bool averageOverlap,
                          VIDEO_FRAME_INFO_S *output);
// Copies visible pixels between equal-format/equal-geometry MB-backed frames while
// respecting independent virtual strides and plane offsets.
RK_S32 cpu_copy_frame_to_existing(const VIDEO_FRAME_INFO_S &source,
                                  VIDEO_FRAME_INFO_S *destination);
// Scales one source rectangle into an existing destination rectangle.
RK_S32 cpu_blit_frame(const VIDEO_FRAME_INFO_S &source,
                      const RECT_S &sourceRect,
                      VIDEO_FRAME_INFO_S *destination,
                      const RECT_S &destinationRect);

// Generic CPU drawing primitives. They operate on every format accepted by
// cpu_frame_is_supported(), including common planar/semi-planar YUV formats.
RK_S32 cpu_fill_frame(VIDEO_FRAME_INFO_S *frame, const RECT_S &rect, RK_U32 color);
RK_S32 cpu_draw_line_frame(VIDEO_FRAME_INFO_S *frame,
                           const POINT_S &start,
                           const POINT_S &end,
                           RK_U32 thickness,
                           RK_U32 color);
RK_S32 cpu_mosaic_frame(VIDEO_FRAME_INFO_S *frame, const RECT_S &rect, RK_U32 blockSize);
RK_S32 cpu_overlay_frame(VIDEO_FRAME_INFO_S *frame,
                         const RK_U8 *source,
                         RK_U64 sourceBytes,
                         RK_U32 sourceWidth,
                         RK_U32 sourceHeight,
                         PIXEL_FORMAT_E sourceFormat,
                         const POINT_S &destinationPoint,
                         RK_U32 globalAlpha);

// Implemented by local_image_modules.cpp. Applies all visible RGN attachments
// for the supplied MPP channel directly to an MB-backed CPU-supported frame.
RK_S32
apply_regions_to_frame(const MPP_CHN_S &channel, VIDEO_FRAME_INFO_S *frame, RK_U32 *appliedCount);

} // namespace rockit_local
