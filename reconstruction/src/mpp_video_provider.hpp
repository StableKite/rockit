#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rk_comm_vdec.h"
#include "rk_comm_venc.h"
#include "rk_comm_video.h"

namespace rockit_mpp {

enum class Status {
    Ok,
    Again,
    Timeout,
    Unavailable,
    Unsupported,
    Invalid,
    NoMemory,
    Corrupt,
    Failed,
};

enum class Policy {
    Disabled,
    Auto,
    Required,
};

struct Encoder;
struct Decoder;

struct EncodedPacket {
    MB_BLK block{MB_INVALID_HANDLE};
    RK_U32 length{0};
    RK_U32 offset{0};
    RK_U64 pts{0};
    bool frame_end{true};
    bool stream_end{false};
    bool idr{false};
};

struct DecodedFrame {
    VIDEO_FRAME_INFO_S frame{};
    bool end_of_stream{false};
    bool corrupt{false};
};

Policy policy() noexcept;
bool policy_is_required() noexcept;
const char *status_name(Status status) noexcept;

Status create_encoder(const VENC_CHN_ATTR_S &attribute, std::shared_ptr<Encoder> *encoder) noexcept;
Status encoder_send(const std::shared_ptr<Encoder> &encoder,
                    const VIDEO_FRAME_INFO_S &frame,
                    RK_S32 timeout_ms,
                    EncodedPacket *packet) noexcept;
Status encoder_request_idr(const std::shared_ptr<Encoder> &encoder) noexcept;
Status encoder_reset(const std::shared_ptr<Encoder> &encoder) noexcept;
Status encoder_send_eos(const std::shared_ptr<Encoder> &encoder,
                        RK_S32 timeout_ms,
                        EncodedPacket *packet) noexcept;

Status create_decoder(const VDEC_CHN_ATTR_S &attribute, std::shared_ptr<Decoder> *decoder) noexcept;
Status decoder_send(const std::shared_ptr<Decoder> &decoder,
                    const VDEC_STREAM_S &stream,
                    RK_S32 timeout_ms,
                    std::vector<DecodedFrame> *frames) noexcept;
Status decoder_drain(const std::shared_ptr<Decoder> &decoder,
                     RK_S32 timeout_ms,
                     std::vector<DecodedFrame> *frames) noexcept;
Status decoder_reset(const std::shared_ptr<Decoder> &decoder) noexcept;

} // namespace rockit_mpp
