#include "rockit_internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rk_defines.h"
#include "rk_mpi_adec.h"
#include "rk_mpi_aenc.h"
#include "rk_mpi_af.h"
#include "rk_mpi_ai.h"
#include "rk_mpi_amix.h"
#include "rk_mpi_ao.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"

namespace {
template <typename T> void clear_value(T *v) {
    if (v)
        std::memset(v, 0, sizeof(T));
}
bool retain_mb(MB_BLK b) {
    return b != MB_INVALID_HANDLE && RK_MPI_MB_AddUserCnt(b) > 0;
}
void release_mb(MB_BLK b) {
    if (b != MB_INVALID_HANDLE)
        (void)RK_MPI_MB_ReleaseMB(b);
}
template <class P>
bool wait_item(std::condition_variable &cv, std::unique_lock<std::mutex> &l, RK_S32 ms, P p) {
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

struct AudioFrameItem {
    AUDIO_FRAME_S frame{};
};
void clear_frames(std::deque<AudioFrameItem> &q) {
    for (auto &i : q)
        release_mb(i.frame.pMbBlk);
    q.clear();
}

constexpr RK_S32 kAudioDevMax = 16;
bool valid_dev(AUDIO_DEV d) {
    return d >= 0 && d < kAudioDevMax;
}
bool valid_ai(AI_CHN c) {
    return c >= 0 && c < AI_MAX_CHN_NUM;
}
bool valid_ao(AO_CHN c) {
    return c >= 0 && c < AO_MAX_CHN_NUM;
}
bool valid_aenc(AENC_CHN c) {
    return c >= 0 && c < AENC_MAX_CHN_NUM;
}
bool valid_adec(ADEC_CHN c) {
    return c >= 0 && c < ADEC_MAX_CHN_NUM;
}
bool valid_af(AF_CHN c) {
    return c >= 0 && c < AF_MAX_CHN_NUM;
}
RK_S32 validate_ai_endpoint(AUDIO_DEV d, AI_CHN c) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    if (!valid_ai(c))
        return RK_ERR_AI_INVALID_CHNID;
    return RK_SUCCESS;
}
RK_S32 validate_ao_endpoint(AUDIO_DEV d, AO_CHN c) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    if (!valid_ao(c))
        return RK_ERR_AO_INVALID_CHNID;
    return RK_SUCCESS;
}

struct AiChannel {
    bool enabled{false}, dataRead{false}, resample{false}, vqe{false}, aed{false}, bcd{false},
        buz{false}, gbs{false};
    AUDIO_SAMPLE_RATE_E outRate{AUDIO_SAMPLE_RATE_BUTT};
    AI_CHN_PARAM_S param{};
    AI_CHN_ATTR_S attr{};
    AI_VQE_CONFIG_S vqeAttr{};
    AI_AED_CONFIG_S aedAttr{};
    AI_BCD_CONFIG_S bcdAttr{};
    AI_BUZ_CONFIG_S buzAttr{};
    AI_GBS_CONFIG_S gbsAttr{};
    AUDIO_SAVE_FILE_INFO_S save{};
    std::deque<AudioFrameItem> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
    RK_U32 seq{0};
};
struct AiDevice {
    bool configured{false}, enabled{false};
    AIO_ATTR_S attr{};
    AUDIO_TRACK_MODE_E track{AUDIO_TRACK_NORMAL};
    RK_S32 volume{0};
    RK_BOOL mute{RK_FALSE};
    AUDIO_FADE_S fade{};
    std::array<AiChannel, AI_MAX_CHN_NUM> channels{};
};
struct AiState {
    std::mutex mutex;
    std::condition_variable cv;
    std::array<AiDevice, kAudioDevMax> devs;
};
AiState &ai_state() {
    static AiState s;
    return s;
}
void clear_ai_channel(AiChannel &c) {
    clear_frames(c.ready);
    for (auto &[b, n] : c.outstanding)
        for (RK_U32 i = 0; i < n; ++i)
            release_mb(b);
    c.outstanding.clear();
}

struct AoChannel {
    bool enabled{false}, paused{false}, resample{false}, vqe{false};
    AUDIO_SAMPLE_RATE_E inRate{AUDIO_SAMPLE_RATE_BUTT};
    AO_CHN_ATTR_S attr{};
    AO_CHN_PARAM_S param{};
    AO_VQE_CONFIG_S vqeAttr{};
    AUDIO_SAVE_FILE_INFO_S save{};
    std::deque<AudioFrameItem> queued;
};
struct AoDevice {
    bool configured{false}, enabled{false};
    AIO_ATTR_S attr{};
    AUDIO_TRACK_MODE_E track{AUDIO_TRACK_NORMAL};
    RK_S32 volume{0};
    RK_BOOL mute{RK_FALSE};
    AUDIO_FADE_S fade{};
    std::array<AoChannel, AO_MAX_CHN_NUM> channels{};
};
struct AoState {
    std::mutex mutex;
    std::condition_variable cv;
    std::array<AoDevice, kAudioDevMax> devs;
};
AoState &ao_state() {
    static AoState s;
    return s;
}

struct AencChannel {
    AENC_CHN_ATTR_S attr{};
    RK_BOOL mute{RK_FALSE};
    AUDIO_SAVE_FILE_INFO_S save{};
    std::deque<AUDIO_STREAM_S> ready;
    std::unordered_map<MB_BLK, RK_U32> outstanding;
    RK_U32 seq{0};
};
struct AencState {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<AENC_CHN, AencChannel> channels;
    RK_S32 nextReg{1};
    std::unordered_set<RK_S32> regs;
};
AencState &aenc_state() {
    static AencState s;
    return s;
}
void clear_streams(std::deque<AUDIO_STREAM_S> &q) {
    for (auto &i : q)
        release_mb(i.pMbBlk);
    q.clear();
}

struct AdecChannel {
    ADEC_CHN_ATTR_S attr{};
    bool eos{false};
    std::deque<AUDIO_FRAME_S> ready;
    std::unordered_map<AUDIO_FRAME_S *, MB_BLK> outstanding;
    RK_U32 seq{0};
};
struct AdecState {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<ADEC_CHN, AdecChannel> channels;
    RK_S32 nextReg{1};
    std::unordered_set<RK_S32> regs;
};
AdecState &adec_state() {
    static AdecState s;
    return s;
}
void clear_adec(AdecChannel &c) {
    for (auto &f : c.ready)
        release_mb(f.pMbBlk);
    c.ready.clear();
    for (auto &[p, b] : c.outstanding) {
        release_mb(b);
        delete p;
    }
    c.outstanding.clear();
}

struct AfChannel {
    AF_ATTR_S attr{};
    std::deque<AUDIO_FRAME_S> ready;
    std::unordered_map<AUDIO_FRAME_S *, MB_BLK> outstanding;
};
struct AfState {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<AF_CHN, AfChannel> channels;
};
AfState &af_state() {
    static AfState s;
    return s;
}
void clear_af(AfChannel &c) {
    for (auto &f : c.ready)
        release_mb(f.pMbBlk);
    c.ready.clear();
    for (auto &[p, b] : c.outstanding) {
        release_mb(b);
        delete p;
    }
    c.outstanding.clear();
}

AUDIO_SOUND_MODE_E channel_sound_mode(RK_U32 channels) {
    switch (channels) {
    case 1:
        return AUDIO_SOUND_MODE_MONO;
    case 2:
        return AUDIO_SOUND_MODE_STEREO;
    case 4:
        return AUDIO_SOUND_MODE_4_CHN;
    case 6:
        return AUDIO_SOUND_MODE_6_CHN;
    case 8:
        return AUDIO_SOUND_MODE_8_CHN;
    default:
        return AUDIO_SOUND_MODE_BUTT;
    }
}

bool af_resample_configured(const AF_ATTR_S &attr) {
    const auto &resample = attr.stResAttr;
    return attr.enType == AUDIO_FILTER_RESAMPLE && resample.u32InRate != 0 &&
           resample.u32OutRate != 0 && resample.u32InChn != 0 && resample.u32InChn <= 8 &&
           resample.u32OutChn != 0 && resample.u32OutChn <= 8 &&
           resample.enInBitWidth == AUDIO_BIT_WIDTH_16 &&
           resample.enOutBitWidth == AUDIO_BIT_WIDTH_16 &&
           channel_sound_mode(resample.u32OutChn) != AUDIO_SOUND_MODE_BUTT;
}

RK_S32 af_resample_frame(const AF_ATTR_S &attr, const AUDIO_FRAME_S &input, AUDIO_FRAME_S *output) {
    if (!output)
        return RK_ERR_AF_NULL_PTR;
    const auto &resample = attr.stResAttr;
    if (!af_resample_configured(attr) || input.enBitWidth != AUDIO_BIT_WIDTH_16 ||
        input.pMbBlk == MB_INVALID_HANDLE || input.u32Len == 0 ||
        (input.u32Len % sizeof(RK_S16)) != 0) {
        return RK_ERR_AF_ILLEGAL_PARAM;
    }
    const RK_U64 inputSamples = input.u32Len / sizeof(RK_S16);
    const RK_U64 inputBytes = inputSamples * resample.u32InChn * sizeof(RK_S16);
    if (inputSamples == 0 || inputBytes > RK_MPI_MB_GetSize(input.pMbBlk)) {
        return RK_ERR_AF_ILLEGAL_PARAM;
    }
    const RK_U64 outputSamples = std::max<RK_U64>(
        1, (inputSamples * resample.u32OutRate + resample.u32InRate / 2U) / resample.u32InRate);
    const RK_U64 outputBytes = outputSamples * resample.u32OutChn * sizeof(RK_S16);
    if (outputSamples > UINT32_MAX / sizeof(RK_S16) || outputBytes > UINT32_MAX) {
        return RK_ERR_AF_ILLEGAL_PARAM;
    }
    const auto *source = static_cast<const RK_S16 *>(RK_MPI_MB_Handle2VirAddr(input.pMbBlk));
    if (!source)
        return RK_ERR_AF_ILLEGAL_PARAM;
    MB_BLK block = MB_INVALID_HANDLE;
    if (RK_MPI_SYS_MmzAlloc(&block, "af-resample", nullptr, outputBytes) != RK_SUCCESS) {
        return RK_ERR_AF_NOBUF;
    }
    auto *destination = static_cast<RK_S16 *>(RK_MPI_MB_Handle2VirAddr(block));
    if (!destination) {
        release_mb(block);
        return RK_ERR_AF_NOBUF;
    }
    for (RK_U64 outputIndex = 0; outputIndex < outputSamples; ++outputIndex) {
        const RK_U64 sourceIndex = std::min<RK_U64>(
            inputSamples - 1, (outputIndex * resample.u32InRate) / resample.u32OutRate);
        const auto *sourceFrame = source + sourceIndex * resample.u32InChn;
        auto *destinationFrame = destination + outputIndex * resample.u32OutChn;
        for (RK_U32 channel = 0; channel < resample.u32OutChn; ++channel) {
            RK_S32 value = 0;
            if (resample.u32OutChn == 1 && resample.u32InChn > 1) {
                RK_S64 sum = 0;
                for (RK_U32 inputChannel = 0; inputChannel < resample.u32InChn; ++inputChannel) {
                    sum += sourceFrame[inputChannel];
                }
                value = static_cast<RK_S32>(sum / resample.u32InChn);
            } else if (resample.u32InChn == 1) {
                value = sourceFrame[0];
            } else {
                value = sourceFrame[channel % resample.u32InChn];
            }
            destinationFrame[channel] = static_cast<RK_S16>(value);
        }
    }
    *output = input;
    output->pMbBlk = block;
    output->enBitWidth = AUDIO_BIT_WIDTH_16;
    output->enSoundMode = channel_sound_mode(resample.u32OutChn);
    output->u32Len = static_cast<RK_U32>(outputSamples * sizeof(RK_S16));
    output->s32SampleRate = static_cast<RK_S32>(resample.u32OutRate);
    output->bBypassMbBlk = RK_TRUE;
    return RK_SUCCESS;
}

bool g711_codec(RK_CODEC_ID_E codec) {
    return codec == RK_AUDIO_ID_PCM_ALAW || codec == RK_AUDIO_ID_PCM_MULAW;
}

RK_U8 linear_to_alaw(RK_S16 sample) {
    constexpr RK_S32 kClip = 32635;
    RK_S32 pcm = sample;
    RK_U8 mask = 0xD5;
    if (pcm < 0) {
        mask = 0x55;
        pcm = -pcm - 1;
    }
    pcm = std::min(pcm, kClip);
    RK_U8 segment = 0;
    for (RK_S32 value = pcm >> 4; value > 15 && segment < 7; value >>= 1) {
        ++segment;
    }
    RK_U8 value = static_cast<RK_U8>(segment << 4);
    value |= segment < 2 ? static_cast<RK_U8>((pcm >> 4) & 0x0F)
                         : static_cast<RK_U8>((pcm >> (segment + 3)) & 0x0F);
    return static_cast<RK_U8>(value ^ mask);
}

RK_S16 alaw_to_linear(RK_U8 encoded) {
    encoded ^= 0x55;
    RK_S32 value = static_cast<RK_S32>(encoded & 0x0F) << 4;
    const RK_S32 segment = (encoded & 0x70) >> 4;
    value += 8;
    if (segment >= 1)
        value += 0x100;
    if (segment > 1)
        value <<= segment - 1;
    return static_cast<RK_S16>((encoded & 0x80) ? value : -value);
}

RK_U8 linear_to_mulaw(RK_S16 sample) {
    constexpr RK_S32 kBias = 0x84;
    constexpr RK_S32 kClip = 32635;
    RK_S32 pcm = sample;
    RK_U8 mask = 0xFF;
    if (pcm < 0) {
        mask = 0x7F;
        pcm = -pcm;
    }
    pcm = std::min(pcm, kClip) + kBias;
    RK_U8 segment = 0;
    for (RK_S32 value = pcm >> 7; value > 1 && segment < 7; value >>= 1) {
        ++segment;
    }
    const RK_U8 mantissa = static_cast<RK_U8>((pcm >> (segment + 3)) & 0x0F);
    return static_cast<RK_U8>((segment << 4 | mantissa) ^ mask);
}

RK_S16 mulaw_to_linear(RK_U8 encoded) {
    constexpr RK_S32 kBias = 0x84;
    encoded = static_cast<RK_U8>(~encoded);
    RK_S32 value = ((encoded & 0x0F) << 3) + kBias;
    value <<= (encoded & 0x70) >> 4;
    value -= kBias;
    return static_cast<RK_S16>((encoded & 0x80) ? -value : value);
}

RK_U8 g711_encode_sample(RK_CODEC_ID_E codec, RK_S16 sample) {
    return codec == RK_AUDIO_ID_PCM_ALAW ? linear_to_alaw(sample) : linear_to_mulaw(sample);
}

RK_S16 g711_decode_sample(RK_CODEC_ID_E codec, RK_U8 sample) {
    return codec == RK_AUDIO_ID_PCM_ALAW ? alaw_to_linear(sample) : mulaw_to_linear(sample);
}

RK_S32
aenc_encode_frame(const AencChannel &channel, const AUDIO_FRAME_S &input, AUDIO_STREAM_S *output) {
    if (!output || input.pMbBlk == MB_INVALID_HANDLE || input.enBitWidth != AUDIO_BIT_WIDTH_16 ||
        input.u32Len == 0 || (input.u32Len % sizeof(RK_S16)) != 0 ||
        input.u32Len > RK_MPI_MB_GetSize(input.pMbBlk)) {
        return RK_ERR_AENC_ILLEGAL_PARAM;
    }
    const auto codec = channel.attr.enType;
    if (!g711_codec(codec))
        return RK_ERR_AENC_NOT_SUPPORT;
    const auto *source = static_cast<const RK_S16 *>(RK_MPI_MB_Handle2VirAddr(input.pMbBlk));
    if (!source)
        return RK_ERR_AENC_ILLEGAL_PARAM;
    const RK_U32 samples = input.u32Len / sizeof(RK_S16);
    MB_BLK block = MB_INVALID_HANDLE;
    if (RK_MPI_SYS_MmzAlloc(&block, "aenc-g711", nullptr, samples) != RK_SUCCESS) {
        return RK_ERR_AENC_NOBUF;
    }
    auto *destination = static_cast<RK_U8 *>(RK_MPI_MB_Handle2VirAddr(block));
    if (!destination) {
        release_mb(block);
        return RK_ERR_AENC_NOBUF;
    }
    for (RK_U32 index = 0; index < samples; ++index) {
        destination[index] = g711_encode_sample(
            codec, channel.mute == RK_TRUE ? static_cast<RK_S16>(0) : source[index]);
    }
    output->pMbBlk = block;
    output->u32Len = samples;
    output->u64TimeStamp = input.u64TimeStamp;
    output->bBypassMbBlk = RK_TRUE;
    return RK_SUCCESS;
}

RK_S32
adec_decode_stream(const AdecChannel &channel, const AUDIO_STREAM_S &input, AUDIO_FRAME_S *output) {
    if (!output || input.pMbBlk == MB_INVALID_HANDLE || input.u32Len == 0 ||
        input.u32Len > RK_MPI_MB_GetSize(input.pMbBlk)) {
        return RK_ERR_ADEC_ILLEGAL_PARAM;
    }
    const auto codec = channel.attr.enType;
    if (!g711_codec(codec))
        return RK_ERR_ADEC_NOT_SUPPORT;
    const auto *source = static_cast<const RK_U8 *>(RK_MPI_MB_Handle2VirAddr(input.pMbBlk));
    if (!source)
        return RK_ERR_ADEC_ILLEGAL_PARAM;
    const RK_U64 outputBytes = static_cast<RK_U64>(input.u32Len) * sizeof(RK_S16);
    if (outputBytes > UINT32_MAX)
        return RK_ERR_ADEC_ILLEGAL_PARAM;
    MB_BLK block = MB_INVALID_HANDLE;
    if (RK_MPI_SYS_MmzAlloc(&block, "adec-g711", nullptr, outputBytes) != RK_SUCCESS) {
        return RK_ERR_ADEC_NOBUF;
    }
    auto *destination = static_cast<RK_S16 *>(RK_MPI_MB_Handle2VirAddr(block));
    if (!destination) {
        release_mb(block);
        return RK_ERR_ADEC_NOBUF;
    }
    for (RK_U32 index = 0; index < input.u32Len; ++index) {
        destination[index] = g711_decode_sample(codec, source[index]);
    }
    output->pMbBlk = block;
    output->u32Len = static_cast<RK_U32>(outputBytes);
    output->u64TimeStamp = input.u64TimeStamp;
    output->enBitWidth = AUDIO_BIT_WIDTH_16;
    output->enSoundMode = channel_sound_mode(
        channel.attr.stCodecAttr.u32Channels == 0 ? 1 : channel.attr.stCodecAttr.u32Channels);
    output->s32SampleRate = static_cast<RK_S32>(channel.attr.stCodecAttr.u32SampleRate);
    output->bBypassMbBlk = RK_TRUE;
    return RK_SUCCESS;
}

struct AmixState {
    std::mutex mutex;
    std::unordered_map<AUDIO_DEV, std::unordered_map<std::string, std::string>> controls;
};
AmixState &amix_state() {
    static AmixState s;
    return s;
}
} // namespace

extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SetPubAttr(AUDIO_DEV d, const AIO_ATTR_S *a) {
    if (!a)
        return RK_ERR_AI_NULL_PTR;
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].configured = true;
    s.devs[d].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetPubAttr(AUDIO_DEV d, AIO_ATTR_S *a) {
    if (!a)
        return RK_ERR_AI_NULL_PTR;
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].configured)
        return RK_ERR_AI_NOT_CONFIG;
    *a = s.devs[d].attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_ClrPubAttr(AUDIO_DEV d) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d];
    if (x.enabled)
        return RK_ERR_AI_NOT_PERM;
    x.configured = false;
    clear_value(&x.attr);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_Enable(AUDIO_DEV d) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].configured)
        return RK_ERR_AI_NOT_CONFIG;
    s.devs[d].enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_Disable(AUDIO_DEV d) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d];
    for (auto &c : x.channels) {
        c.enabled = false;
        clear_ai_channel(c);
    }
    x.enabled = false;
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_EnableChn(AUDIO_DEV d, AI_CHN c) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    if (!valid_ai(c))
        return RK_ERR_AI_INVALID_CHNID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].enabled)
        return RK_ERR_AI_NOT_ENABLED;
    s.devs[d].channels[c].enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_DisableChn(AUDIO_DEV d, AI_CHN c) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    if (!valid_ai(c))
        return RK_ERR_AI_INVALID_CHNID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d].channels[c];
    x.enabled = false;
    clear_ai_channel(x);
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_EnableDataRead(AUDIO_DEV d, AI_CHN c) {
    RK_S32 vr = validate_ai_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].dataRead = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_DisableDataRead(AUDIO_DEV d, AI_CHN c) {
    RK_S32 vr = validate_ai_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].dataRead = false;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SendFrame(AUDIO_DEV d,
                                                 AO_CHN c,
                                                 const AUDIO_FRAME_S *f,
                                                 RK_S32 timeout) {
    RK_S32 validation = validate_ai_endpoint(d, c);
    if (validation != RK_SUCCESS)
        return validation;
    if (!f)
        return RK_ERR_AI_NULL_PTR;
    auto &s = ai_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto &device = s.devs[d];
    auto &channel = device.channels[c];
    if (!device.enabled || !channel.enabled)
        return RK_ERR_AI_NOT_ENABLED;
    const std::size_t capacity = channel.param.s32UsrFrmDepth > 0
                                     ? static_cast<std::size_t>(channel.param.s32UsrFrmDepth)
                                     : 8;
    const auto room = [&] {
        return channel.ready.size() < capacity || !channel.enabled;
    };
    if (!wait_item(s.cv, lock, timeout, room) || channel.ready.size() >= capacity) {
        return RK_ERR_AI_BUF_FULL;
    }
    if (!retain_mb(f->pMbBlk))
        return RK_ERR_AI_ILLEGAL_PARAM;
    AUDIO_FRAME_S copy = *f;
    copy.u32Seq = ++channel.seq;
    channel.ready.push_back({copy});
    s.cv.notify_all();
    const MPP_CHN_S source{RK_ID_AI, d, c};
    lock.unlock();
    return rockit_local::route_audio_frame(source, copy, timeout);
}

extern "C" ROCKIT_API RK_S32
RK_MPI_AI_GetFrame(AUDIO_DEV d, AI_CHN c, AUDIO_FRAME_S *f, AEC_FRAME_S *a, RK_S32 ms) {
    RK_S32 validation = validate_ai_endpoint(d, c);
    if (validation != RK_SUCCESS)
        return validation;
    if (!f)
        return RK_ERR_AI_NULL_PTR;
    auto &s = ai_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    const auto ready = [&] {
        return !s.devs[d].channels[c].enabled || !s.devs[d].channels[c].ready.empty();
    };
    if (!wait_item(s.cv, lock, ms, ready))
        return RK_ERR_AI_BUF_EMPTY;
    auto &channel = s.devs[d].channels[c];
    if (channel.ready.empty())
        return RK_ERR_AI_BUF_EMPTY;
    *f = channel.ready.front().frame;
    channel.ready.pop_front();
    ++channel.outstanding[f->pMbBlk];
    if (a)
        clear_value(a);
    s.cv.notify_all();
    return RK_SUCCESS;
}
#define ROCKIT_AI_LOCK_ENDPOINT_PTR(device_, channel_, pointer_)                                  \
    RK_S32 vr = validate_ai_endpoint((device_), (channel_));                                      \
    if (vr != RK_SUCCESS)                                                                          \
        return vr;                                                                                 \
    if (!(pointer_))                                                                               \
        return RK_ERR_AI_NULL_PTR;                                                                \
    auto &s = ai_state();                                                                          \
    std::lock_guard<std::mutex> l(s.mutex)

extern "C" ROCKIT_API RK_S32 RK_MPI_AI_ReleaseFrame(AUDIO_DEV d,
                                                    AI_CHN c,
                                                    const AUDIO_FRAME_S *f,
                                                    const AEC_FRAME_S *) {
    ROCKIT_AI_LOCK_ENDPOINT_PTR(d, c, f);
    auto &x = s.devs[d].channels[c];
    auto i = x.outstanding.find(f->pMbBlk);
    if (i == x.outstanding.end())
        return RK_ERR_AI_ILLEGAL_PARAM;
    if (--i->second == 0)
        x.outstanding.erase(i);
    release_mb(f->pMbBlk);
    s.cv.notify_all();
    return RK_SUCCESS;
}
#define AI_CHN_SETGET(setn, getn, type, field)                                                     \
    extern "C" ROCKIT_API RK_S32 setn(AUDIO_DEV d, AI_CHN c, const type *p) {                      \
        if (!p)                                                                                    \
            return RK_ERR_AI_NULL_PTR;                                                             \
        if (!valid_dev(d) || !valid_ai(c))                                                         \
            return RK_ERR_AI_ILLEGAL_PARAM;                                                        \
        auto &s = ai_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        s.devs[d].channels[c].field = *p;                                                          \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 getn(AUDIO_DEV d, AI_CHN c, type *p) {                            \
        if (!p)                                                                                    \
            return RK_ERR_AI_NULL_PTR;                                                             \
        if (!valid_dev(d) || !valid_ai(c))                                                         \
            return RK_ERR_AI_ILLEGAL_PARAM;                                                        \
        auto &s = ai_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        *p = s.devs[d].channels[c].field;                                                          \
        return RK_SUCCESS;                                                                         \
    }
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SetChnParam(AUDIO_DEV d, AI_CHN c, const AI_CHN_PARAM_S *p) {
    ROCKIT_AI_LOCK_ENDPOINT_PTR(d, c, p);
    s.devs[d].channels[c].param = *p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetChnParam(AUDIO_DEV d, AI_CHN c, AI_CHN_PARAM_S *p) {
    ROCKIT_AI_LOCK_ENDPOINT_PTR(d, c, p);
    *p = s.devs[d].channels[c].param;
    return RK_SUCCESS;
}
AI_CHN_SETGET(RK_MPI_AI_SetChnAttr, RK_MPI_AI_GetChnAttr, AI_CHN_ATTR_S, attr)
AI_CHN_SETGET(RK_MPI_AI_SetAedAttr, RK_MPI_AI_GetAedAttr, AI_AED_CONFIG_S, aedAttr)
AI_CHN_SETGET(RK_MPI_AI_SetBcdAttr, RK_MPI_AI_GetBcdAttr, AI_BCD_CONFIG_S, bcdAttr)
AI_CHN_SETGET(RK_MPI_AI_SetBuzAttr, RK_MPI_AI_GetBuzAttr, AI_BUZ_CONFIG_S, buzAttr)
AI_CHN_SETGET(RK_MPI_AI_SetGbsAttr, RK_MPI_AI_GetGbsAttr, AI_GBS_CONFIG_S, gbsAttr)
#undef AI_CHN_SETGET
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_EnableReSmp(AUDIO_DEV d, AI_CHN c, AUDIO_SAMPLE_RATE_E r) {
    RK_S32 vr = validate_ai_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d].channels[c];
    x.resample = true;
    x.outRate = r;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_DisableReSmp(AUDIO_DEV d, AI_CHN c) {
    RK_S32 vr = validate_ai_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].resample = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SetTrackMode(AUDIO_DEV d, AUDIO_TRACK_MODE_E m) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].track = m;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetTrackMode(AUDIO_DEV d, AUDIO_TRACK_MODE_E *m) {
    if (!m)
        return RK_ERR_AI_NULL_PTR;
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *m = s.devs[d].track;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SetVolumeCurve(AUDIO_DEV d, AUDIO_VOLUME_CURVE_S *p) {
    if (!p)
        return RK_ERR_AI_NULL_PTR;
    return valid_dev(d) ? RK_SUCCESS : RK_ERR_AI_INVALID_DEVID;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SetVolume(AUDIO_DEV d, RK_S32 v) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].volume = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetVolume(AUDIO_DEV d, RK_S32 *v) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    if (!v)
        return RK_ERR_AI_NULL_PTR;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *v = s.devs[d].volume;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SetMute(AUDIO_DEV d, RK_BOOL b, const AUDIO_FADE_S *f) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].mute = b;
    if (f)
        s.devs[d].fade = *f;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetMute(AUDIO_DEV d, RK_BOOL *b, AUDIO_FADE_S *f) {
    if (!valid_dev(d))
        return RK_ERR_AI_INVALID_DEVID;
    if (!b)
        return RK_ERR_AI_NULL_PTR;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *b = s.devs[d].mute;
    if (f)
        *f = s.devs[d].fade;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_SaveFile(AUDIO_DEV d,
                                                AI_CHN c,
                                                const AUDIO_SAVE_FILE_INFO_S *f) {
    ROCKIT_AI_LOCK_ENDPOINT_PTR(d, c, f);
    s.devs[d].channels[c].save = *f;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_QueryFileStatus(AUDIO_DEV d,
                                                       AI_CHN c,
                                                       AUDIO_FILE_STATUS_S *f) {
    ROCKIT_AI_LOCK_ENDPOINT_PTR(d, c, f);
    f->bSaving = s.devs[d].channels[c].save.bCfg;
    return RK_SUCCESS;
}
#undef ROCKIT_AI_LOCK_ENDPOINT_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetFd(AUDIO_DEV d, AI_CHN c) {
    RK_S32 vr = validate_ai_endpoint(d, c);
    return vr == RK_SUCCESS ? 1000 + d * AI_MAX_CHN_NUM + c : vr;
}
extern "C" ROCKIT_API RK_S32
RK_MPI_AI_SetVqeAttr(AUDIO_DEV d, AI_CHN c, AUDIO_DEV, AO_CHN, const AI_VQE_CONFIG_S *p) {
    if (!p)
        return RK_ERR_AI_NULL_PTR;
    if (!valid_dev(d) || !valid_ai(c))
        return RK_ERR_AI_ILLEGAL_PARAM;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].vqeAttr = *p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AI_GetVqeAttr(AUDIO_DEV d, AI_CHN c, AI_VQE_CONFIG_S *p) {
    if (!p)
        return RK_ERR_AI_NULL_PTR;
    if (!valid_dev(d) || !valid_ai(c))
        return RK_ERR_AI_ILLEGAL_PARAM;
    auto &s = ai_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *p = s.devs[d].channels[c].vqeAttr;
    return RK_SUCCESS;
}
#define AI_FLAG(en, dis, field)                                                                    \
    extern "C" ROCKIT_API RK_S32 en(AUDIO_DEV d, AI_CHN c) {                                       \
        RK_S32 vr = validate_ai_endpoint(d, c);                                                    \
        if (vr != RK_SUCCESS)                                                                      \
            return vr;                                                                             \
        auto &s = ai_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        s.devs[d].channels[c].field = true;                                                        \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 dis(AUDIO_DEV d, AI_CHN c) {                                      \
        RK_S32 vr = validate_ai_endpoint(d, c);                                                    \
        if (vr != RK_SUCCESS)                                                                      \
            return vr;                                                                             \
        auto &s = ai_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        s.devs[d].channels[c].field = false;                                                       \
        return RK_SUCCESS;                                                                         \
    }
AI_FLAG(RK_MPI_AI_EnableVqe, RK_MPI_AI_DisableVqe, vqe)
AI_FLAG(RK_MPI_AI_EnableAed, RK_MPI_AI_DisableAed, aed)
    AI_FLAG(RK_MPI_AI_EnableBcd, RK_MPI_AI_DisableBcd, bcd)
        AI_FLAG(RK_MPI_AI_EnableBuz, RK_MPI_AI_DisableBuz, buz)
            AI_FLAG(RK_MPI_AI_EnableGbs, RK_MPI_AI_DisableGbs, gbs)
#undef AI_FLAG
#define AI_RESULT(name, type, field)                                                               \
    extern "C" ROCKIT_API RK_S32 name(AUDIO_DEV d, AI_CHN c, type *r) {                            \
        if (!r)                                                                                    \
            return RK_ERR_AI_NULL_PTR;                                                             \
        if (!valid_dev(d) || !valid_ai(c))                                                         \
            return RK_ERR_AI_ILLEGAL_PARAM;                                                        \
        auto &s = ai_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        if (!s.devs[d].channels[c].field)                                                          \
            return RK_ERR_AI_NOT_ENABLED;                                                          \
        clear_value(r);                                                                            \
        return RK_SUCCESS;                                                                         \
    }
                AI_RESULT(RK_MPI_AI_GetAedResult, AI_AED_RESULT_S, aed)
                    AI_RESULT(RK_MPI_AI_GetBcdResult, AI_BCD_RESULT_S, bcd)
                        AI_RESULT(RK_MPI_AI_GetBuzResult, AI_BUZ_RESULT_S, buz)
                            AI_RESULT(RK_MPI_AI_GetGbsResult, AI_GBS_RESULT_S, gbs)
#undef AI_RESULT

                                extern "C" ROCKIT_API RK_S32
    RK_MPI_AO_SetPubAttr(AUDIO_DEV d, const AIO_ATTR_S *a) {
    if (!a)
        return RK_ERR_AO_NULL_PTR;
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].configured = true;
    s.devs[d].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_GetPubAttr(AUDIO_DEV d, AIO_ATTR_S *a) {
    if (!a)
        return RK_ERR_AO_NULL_PTR;
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].configured)
        return RK_ERR_AO_NOT_CONFIG;
    *a = s.devs[d].attr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_ClrPubAttr(AUDIO_DEV d) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.devs[d].enabled)
        return RK_ERR_AO_NOT_PERM;
    s.devs[d].configured = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_Enable(AUDIO_DEV d) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].configured)
        return RK_ERR_AO_NOT_CONFIG;
    s.devs[d].enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_Disable(AUDIO_DEV d) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    for (auto &c : s.devs[d].channels) {
        clear_frames(c.queued);
        c.enabled = false;
    }
    s.devs[d].enabled = false;
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_EnableChn(AUDIO_DEV d, AO_CHN c) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    if (!valid_ao(c))
        return RK_ERR_AO_INVALID_CHNID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (!s.devs[d].enabled)
        return RK_ERR_AO_NOT_ENABLED;
    s.devs[d].channels[c].enabled = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_DisableChn(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto &x = s.devs[d].channels[c];
    clear_frames(x.queued);
    x.enabled = false;
    s.cv.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SendFrame(AUDIO_DEV d,
                                                 AO_CHN c,
                                                 const AUDIO_FRAME_S *f,
                                                 RK_S32 timeout) {
    RK_S32 validation = validate_ao_endpoint(d, c);
    if (validation != RK_SUCCESS)
        return validation;
    if (!f)
        return RK_ERR_AO_NULL_PTR;
    auto &s = ao_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto &channel = s.devs[d].channels[c];
    if (!s.devs[d].enabled || !channel.enabled)
        return RK_ERR_AO_NOT_ENABLED;
    const auto room = [&] {
        return channel.queued.size() < 8 || !channel.enabled;
    };
    if (!wait_item(s.cv, lock, timeout, room) || channel.queued.size() >= 8) {
        return RK_ERR_AO_BUF_FULL;
    }
    if (!retain_mb(f->pMbBlk))
        return RK_ERR_AO_ILLEGAL_PARAM;
    channel.queued.push_back({*f});
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_ClearChnBuf(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    clear_frames(s.devs[d].channels[c].queued);
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_QueryChnStat(AUDIO_DEV d, AO_CHN c, AO_CHN_STATE_S *p) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    if (!p)
        return RK_ERR_AO_NULL_PTR;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    const auto n = static_cast<RK_U32>(s.devs[d].channels[c].queued.size());
    p->u32ChnTotalNum = 8;
    p->u32ChnBusyNum = n;
    p->u32ChnFreeNum = 8 - n;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_PauseChn(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].paused = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_ResumeChn(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].paused = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_WaitEos(AUDIO_DEV d, AO_CHN c, RK_S32) {
    return RK_MPI_AO_ClearChnBuf(d, c);
}
#define AO_CHN_SETGET(setn, getn, type, field)                                                     \
    extern "C" ROCKIT_API RK_S32 setn(AUDIO_DEV d, AO_CHN c, const type *p) {                      \
        if (!p)                                                                                    \
            return RK_ERR_AO_NULL_PTR;                                                             \
        if (!valid_dev(d) || !valid_ao(c))                                                         \
            return RK_ERR_AO_ILLEGAL_PARAM;                                                        \
        auto &s = ao_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        s.devs[d].channels[c].field = *p;                                                          \
        return RK_SUCCESS;                                                                         \
    }                                                                                              \
    extern "C" ROCKIT_API RK_S32 getn(AUDIO_DEV d, AO_CHN c, type *p) {                            \
        if (!p)                                                                                    \
            return RK_ERR_AO_NULL_PTR;                                                             \
        if (!valid_dev(d) || !valid_ao(c))                                                         \
            return RK_ERR_AO_ILLEGAL_PARAM;                                                        \
        auto &s = ao_state();                                                                      \
        std::lock_guard<std::mutex> l(s.mutex);                                                    \
        *p = s.devs[d].channels[c].field;                                                          \
        return RK_SUCCESS;                                                                         \
    }
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SetChnAttr(AUDIO_DEV d, AO_CHN c, const AO_CHN_ATTR_S *p) {
    if (!p)
        return RK_ERR_AI_NULL_PTR;
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].attr = *p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_GetChnAttr(AUDIO_DEV d, AO_CHN c, AO_CHN_ATTR_S *p) {
    if (!p)
        return RK_ERR_AI_NULL_PTR;
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *p = s.devs[d].channels[c].attr;
    return RK_SUCCESS;
}
AO_CHN_SETGET(RK_MPI_AO_SetChnParams, RK_MPI_AO_GetChnParams, AO_CHN_PARAM_S, param)
#undef AO_CHN_SETGET
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_EnableReSmp(AUDIO_DEV d, AO_CHN c, AUDIO_SAMPLE_RATE_E r) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].resample = true;
    s.devs[d].channels[c].inRate = r;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_DisableReSmp(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].resample = false;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SetTrackMode(AUDIO_DEV d, AUDIO_TRACK_MODE_E m) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].track = m;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_GetTrackMode(AUDIO_DEV d, AUDIO_TRACK_MODE_E *m) {
    if (!m)
        return RK_ERR_AO_NULL_PTR;
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *m = s.devs[d].track;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SetVolumeCurve(AUDIO_DEV d, AUDIO_VOLUME_CURVE_S *p) {
    if (!p)
        return RK_ERR_AO_NULL_PTR;
    return valid_dev(d) ? RK_SUCCESS : RK_ERR_AO_INVALID_DEVID;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SetVolume(AUDIO_DEV d, RK_S32 v) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].volume = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_GetVolume(AUDIO_DEV d, RK_S32 *v) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    if (!v)
        return RK_ERR_AO_NULL_PTR;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *v = s.devs[d].volume;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SetMute(AUDIO_DEV d, RK_BOOL b, const AUDIO_FADE_S *f) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].mute = b;
    if (f)
        s.devs[d].fade = *f;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_GetMute(AUDIO_DEV d, RK_BOOL *b, AUDIO_FADE_S *f) {
    if (!valid_dev(d))
        return RK_ERR_AO_INVALID_DEVID;
    if (!b)
        return RK_ERR_AO_NULL_PTR;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *b = s.devs[d].mute;
    if (f)
        *f = s.devs[d].fade;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SaveFile(AUDIO_DEV d, AO_CHN c, AUDIO_SAVE_FILE_INFO_S *f) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    if (!f)
        return RK_ERR_AO_NULL_PTR;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].save = *f;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_QueryFileStatus(AUDIO_DEV d,
                                                       AO_CHN c,
                                                       AUDIO_FILE_STATUS_S *f) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    if (!f)
        return RK_ERR_AO_NULL_PTR;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    f->bSaving = s.devs[d].channels[c].save.bCfg;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_SetVqeAttr(AUDIO_DEV d, AO_CHN c, AO_VQE_CONFIG_S *p) {
    if (!p)
        return RK_ERR_AO_NULL_PTR;
    if (!valid_dev(d) || !valid_ao(c))
        return RK_ERR_AO_ILLEGAL_PARAM;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].vqeAttr = *p;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_GetVqeAttr(AUDIO_DEV d, AO_CHN c, AO_VQE_CONFIG_S *p) {
    if (!p)
        return RK_ERR_AO_NULL_PTR;
    if (!valid_dev(d) || !valid_ao(c))
        return RK_ERR_AO_ILLEGAL_PARAM;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *p = s.devs[d].channels[c].vqeAttr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_EnableVqe(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].vqe = true;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AO_DisableVqe(AUDIO_DEV d, AO_CHN c) {
    RK_S32 vr = validate_ao_endpoint(d, c);
    if (vr != RK_SUCCESS)
        return vr;
    auto &s = ao_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.devs[d].channels[c].vqe = false;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_CreateChn(AENC_CHN c, const AENC_CHN_ATTR_S *a) {
    if (!a)
        return RK_ERR_AENC_NULL_PTR;
    if (!valid_aenc(c))
        return RK_ERR_AENC_INVALID_CHNID;
    auto &s = aenc_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.channels.count(c))
        return RK_ERR_AENC_EXIST;
    s.channels[c].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_DestroyChn(AENC_CHN c) {
    if (!valid_aenc(c))
        return RK_ERR_AENC_INVALID_CHNID;
    auto &s = aenc_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_AENC_UNEXIST;
    clear_streams(i->second.ready);
    for (auto &[b, n] : i->second.outstanding)
        for (RK_U32 j = 0; j < n; ++j)
            release_mb(b);
    s.channels.erase(i);
    s.cv.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_SendFrame(AENC_CHN c,
                                                   const AUDIO_FRAME_S *f,
                                                   const AEC_FRAME_S *,
                                                   RK_S32 timeout) {
    if (!valid_aenc(c))
        return RK_ERR_AENC_INVALID_CHNID;
    if (!f)
        return RK_ERR_AENC_NULL_PTR;
    auto &s = aenc_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_AENC_UNEXIST;
    const std::size_t capacity =
        iterator->second.attr.u32BufCount ? iterator->second.attr.u32BufCount : 8;
    const auto room = [&] {
        const auto current = s.channels.find(c);
        return current == s.channels.end() || current->second.ready.size() < capacity;
    };
    if (!wait_item(s.cv, lock, timeout, room))
        return RK_ERR_AENC_BUF_FULL;
    iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_AENC_UNEXIST;
    if (iterator->second.ready.size() >= capacity)
        return RK_ERR_AENC_BUF_FULL;
    AUDIO_STREAM_S output{};
    if (g711_codec(iterator->second.attr.enType)) {
        const RK_S32 result = aenc_encode_frame(iterator->second, *f, &output);
        if (result != RK_SUCCESS)
            return result;
    } else {
        if (!retain_mb(f->pMbBlk))
            return RK_ERR_AENC_ILLEGAL_PARAM;
        output.pMbBlk = f->pMbBlk;
        output.u32Len = f->u32Len;
        output.u64TimeStamp = f->u64TimeStamp;
        output.bBypassMbBlk = RK_TRUE;
    }
    output.u32Seq = ++iterator->second.seq;
    iterator->second.ready.push_back(output);
    s.cv.notify_all();
    const MPP_CHN_S source{RK_ID_AENC, 0, c};
    lock.unlock();
    return rockit_local::route_audio_stream(source, output);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_GetStream(AENC_CHN c, AUDIO_STREAM_S *output, RK_S32 ms) {
    if (!valid_aenc(c))
        return RK_ERR_AENC_INVALID_CHNID;
    if (!output)
        return RK_ERR_AENC_NULL_PTR;
    auto &s = aenc_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    const auto ready = [&] {
        const auto iterator = s.channels.find(c);
        return iterator == s.channels.end() || !iterator->second.ready.empty();
    };
    if (!wait_item(s.cv, lock, ms, ready))
        return RK_ERR_AENC_BUF_EMPTY;
    auto iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_AENC_UNEXIST;
    if (iterator->second.ready.empty())
        return RK_ERR_AENC_BUF_EMPTY;
    *output = iterator->second.ready.front();
    iterator->second.ready.pop_front();
    ++iterator->second.outstanding[output->pMbBlk];
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_ReleaseStream(AENC_CHN c, const AUDIO_STREAM_S *o) {
    if (!valid_aenc(c))
        return RK_ERR_AENC_INVALID_CHNID;
    if (!o)
        return RK_ERR_AENC_NULL_PTR;
    auto &s = aenc_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_AENC_UNEXIST;
    auto j = i->second.outstanding.find(o->pMbBlk);
    if (j == i->second.outstanding.end())
        return RK_ERR_AENC_ILLEGAL_PARAM;
    if (--j->second == 0)
        i->second.outstanding.erase(j);
    release_mb(o->pMbBlk);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_SetMute(AENC_CHN, RK_BOOL) {
    return RK_FAILURE;
}
#define ROCKIT_AENC_LOCK_CHANNEL_PTR(channel_, pointer_)                                           \
    if (!valid_aenc(channel_))                                                                     \
        return RK_ERR_AENC_INVALID_CHNID;                                                          \
    if (!(pointer_))                                                                               \
        return RK_ERR_AENC_NULL_PTR;                                                               \
    auto &s = aenc_state();                                                                        \
    std::lock_guard<std::mutex> l(s.mutex);                                                        \
    auto i = s.channels.find(channel_);                                                            \
    if (i == s.channels.end())                                                                     \
        return RK_ERR_AENC_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_SaveFile(AENC_CHN c, const AUDIO_SAVE_FILE_INFO_S *f) {
    ROCKIT_AENC_LOCK_CHANNEL_PTR(c, f);
    i->second.save = *f;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_QueryFileStatus(AENC_CHN c, AUDIO_FILE_STATUS_S *f) {
    ROCKIT_AENC_LOCK_CHANNEL_PTR(c, f);
    f->bSaving = i->second.save.bCfg;
    return RK_SUCCESS;
}
#undef ROCKIT_AENC_LOCK_CHANNEL_PTR
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_GetFd(AENC_CHN c) {
    if (!valid_aenc(c))
        return RK_ERR_AENC_INVALID_CHNID;
    auto &s = aenc_state();
    std::lock_guard<std::mutex> l(s.mutex);
    return s.channels.count(c) ? 2000 + c : RK_ERR_AENC_UNEXIST;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_RegisterEncoder(RK_S32 *h, const AENC_ENCODER_S *e) {
    if (!h || !e)
        return RK_ERR_AENC_ILLEGAL_PARAM;
    auto &s = aenc_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *h = s.nextReg++;
    s.regs.insert(*h);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AENC_UnRegisterEncoder(RK_S32 h) {
    auto &s = aenc_state();
    std::lock_guard<std::mutex> l(s.mutex);
    return s.regs.erase(h) ? RK_SUCCESS : RK_ERR_AENC_REGISTER_ERR;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_CreateChn(ADEC_CHN c, const ADEC_CHN_ATTR_S *a) {
    if (!a)
        return RK_ERR_ADEC_NULL_PTR;
    if (!valid_adec(c))
        return RK_ERR_ADEC_INVALID_CHNID;
    auto &s = adec_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.channels.count(c))
        return RK_ERR_ADEC_EXIST;
    s.channels[c].attr = *a;
    return RK_SUCCESS;
}
#define ROCKIT_ADEC_LOCK_CHANNEL(channel_)                                                        \
    if (!valid_adec(channel_))                                                                     \
        return RK_ERR_ADEC_INVALID_CHNID;                                                         \
    auto &s = adec_state();                                                                        \
    std::lock_guard<std::mutex> l(s.mutex);                                                       \
    auto i = s.channels.find(channel_);                                                           \
    if (i == s.channels.end())                                                                    \
        return RK_ERR_ADEC_UNEXIST

extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_DestroyChn(ADEC_CHN c) {
    ROCKIT_ADEC_LOCK_CHANNEL(c);
    clear_adec(i->second);
    s.channels.erase(i);
    s.cv.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_SendStream(ADEC_CHN c,
                                                    const AUDIO_STREAM_S *input,
                                                    RK_BOOL) {
    if (!valid_adec(c))
        return RK_ERR_ADEC_INVALID_CHNID;
    if (!input)
        return RK_ERR_ADEC_NULL_PTR;
    auto &s = adec_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_ADEC_UNEXIST;
    const std::size_t capacity =
        iterator->second.attr.u32BufCount ? iterator->second.attr.u32BufCount : 8;
    if (iterator->second.ready.size() >= capacity)
        return RK_ERR_ADEC_BUF_FULL;
    AUDIO_FRAME_S frame{};
    if (g711_codec(iterator->second.attr.enType)) {
        const RK_S32 result = adec_decode_stream(iterator->second, *input, &frame);
        if (result != RK_SUCCESS)
            return result;
    } else {
        if (!retain_mb(input->pMbBlk))
            return RK_ERR_ADEC_ILLEGAL_PARAM;
        frame.pMbBlk = input->pMbBlk;
        frame.u32Len = input->u32Len;
        frame.u64TimeStamp = input->u64TimeStamp;
        frame.enBitWidth = AUDIO_BIT_WIDTH_16;
        frame.enSoundMode = AUDIO_SOUND_MODE_STEREO;
        frame.s32SampleRate = static_cast<RK_S32>(iterator->second.attr.stCodecAttr.u32SampleRate);
        frame.bBypassMbBlk = RK_TRUE;
    }
    frame.u32Seq = ++iterator->second.seq;
    iterator->second.ready.push_back(frame);
    s.cv.notify_all();
    const MPP_CHN_S source{RK_ID_ADEC, 0, c};
    lock.unlock();
    return rockit_local::route_audio_frame(source, frame, 0);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_GetFrame(ADEC_CHN c,
                                                  AUDIO_FRAME_INFO_S *output,
                                                  RK_BOOL block) {
    if (!valid_adec(c))
        return RK_ERR_ADEC_INVALID_CHNID;
    if (!output)
        return RK_ERR_ADEC_NULL_PTR;
    auto &s = adec_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    const auto ready = [&] {
        const auto iterator = s.channels.find(c);
        return iterator == s.channels.end() || !iterator->second.ready.empty() ||
               iterator->second.eos;
    };
    if (!wait_item(s.cv, lock, block ? -1 : 0, ready))
        return RK_ERR_ADEC_BUF_EMPTY;
    auto iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_ADEC_UNEXIST;
    if (iterator->second.ready.empty())
        return RK_ERR_ADEC_BUF_EMPTY;
    auto *frame = new (std::nothrow) AUDIO_FRAME_S(iterator->second.ready.front());
    if (!frame)
        return RK_ERR_ADEC_NOMEM;
    iterator->second.ready.pop_front();
    iterator->second.outstanding[frame] = frame->pMbBlk;
    output->pstFrame = frame;
    output->u32Id = frame->u32Seq;
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_ReleaseFrame(ADEC_CHN c, AUDIO_FRAME_INFO_S *out) {
    if (!valid_adec(c))
        return RK_ERR_ADEC_INVALID_CHNID;
    if (!out || !out->pstFrame)
        return RK_ERR_ADEC_NULL_PTR;
    auto &s = adec_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_ADEC_UNEXIST;
    auto j = i->second.outstanding.find(out->pstFrame);
    if (j == i->second.outstanding.end())
        return RK_ERR_ADEC_ILLEGAL_PARAM;
    release_mb(j->second);
    delete out->pstFrame;
    i->second.outstanding.erase(j);
    out->pstFrame = nullptr;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_ClearChnBuf(ADEC_CHN c) {
    ROCKIT_ADEC_LOCK_CHANNEL(c);
    for (auto &f : i->second.ready)
        release_mb(f.pMbBlk);
    i->second.ready.clear();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_SendEndOfStream(ADEC_CHN c, RK_BOOL) {
    ROCKIT_ADEC_LOCK_CHANNEL(c);
    i->second.eos = true;
    s.cv.notify_all();
    return RK_SUCCESS;
}
#undef ROCKIT_ADEC_LOCK_CHANNEL
extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_QueryChnStat(ADEC_CHN c, ADEC_CHN_STATE_S *p) {
    if (!valid_adec(c))
        return RK_ERR_ADEC_INVALID_CHNID;
    if (!p)
        return RK_ERR_ADEC_NULL_PTR;
    auto &s = adec_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_ADEC_UNEXIST;
    const RK_U32 cap = i->second.attr.u32BufCount ? i->second.attr.u32BufCount : 8;
    const RK_U32 n = static_cast<RK_U32>(i->second.ready.size());
    p->bEndOfStream = i->second.eos ? RK_TRUE : RK_FALSE;
    p->u32BufferFrmNum = cap;
    p->u32BufferBusyNum = n;
    p->u32BufferFreeNum = cap - n;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_RegisterDecoder(RK_S32 *h, const ADEC_DECODER_S *d) {
    if (!h || !d)
        return RK_ERR_ADEC_NULL_PTR;
    auto &s = adec_state();
    std::lock_guard<std::mutex> l(s.mutex);
    *h = s.nextReg++;
    s.regs.insert(*h);
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_ADEC_UnRegisterDecoder(RK_S32 h) {
    auto &s = adec_state();
    std::lock_guard<std::mutex> l(s.mutex);
    return s.regs.erase(h) ? RK_SUCCESS : RK_ERR_ADEC_REGISTER_ERR;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AF_Create(AF_CHN c, const AF_ATTR_S *a) {
    if (!a)
        return RK_SUCCESS;
    if (c < 0 || c >= AF_MAX_CHN_NUM)
        return RK_ERR_AF_INVALID_CHNID;
    auto &s = af_state();
    std::lock_guard<std::mutex> l(s.mutex);
    if (s.channels.count(c))
        return RK_ERR_AF_EXIST;
    s.channels[c].attr = *a;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AF_Destroy(AF_CHN c) {
    if (!valid_af(c))
        return RK_ERR_AF_INVALID_CHNID;
    auto &s = af_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_AF_UNEXIST;
    clear_af(i->second);
    s.channels.erase(i);
    s.cv.notify_all();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AF_SendFrame(AF_CHN c,
                                                 AUDIO_FRAME_INFO_S *input,
                                                 RK_S32 timeout) {
    if (!valid_af(c))
        return RK_ERR_AF_INVALID_CHNID;
    if (!input || !input->pstFrame)
        return RK_ERR_AF_NULL_PTR;
    auto &s = af_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_AF_UNEXIST;
    const std::size_t capacity =
        iterator->second.attr.u32OutBufCount ? iterator->second.attr.u32OutBufCount : 8;
    const auto room = [&] {
        const auto current = s.channels.find(c);
        return current == s.channels.end() || current->second.ready.size() < capacity;
    };
    if (!wait_item(s.cv, lock, timeout, room))
        return RK_ERR_AF_BUF_FULL;
    iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_AF_UNEXIST;
    if (iterator->second.ready.size() >= capacity)
        return RK_ERR_AF_BUF_FULL;
    AUDIO_FRAME_S output{};
    if (af_resample_configured(iterator->second.attr)) {
        const RK_S32 result = af_resample_frame(iterator->second.attr, *input->pstFrame, &output);
        if (result != RK_SUCCESS)
            return result;
    } else {
        if (!retain_mb(input->pstFrame->pMbBlk))
            return RK_ERR_AF_ILLEGAL_PARAM;
        output = *input->pstFrame;
    }
    iterator->second.ready.push_back(output);
    s.cv.notify_all();
    const MPP_CHN_S source{RK_ID_AF, 0, c};
    lock.unlock();
    return rockit_local::route_audio_frame(source, output, timeout);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AF_GetFrame(AF_CHN c, AUDIO_FRAME_INFO_S *output, RK_S32 ms) {
    if (!valid_af(c))
        return RK_ERR_AF_INVALID_CHNID;
    if (!output)
        return RK_ERR_AF_NULL_PTR;
    auto &s = af_state();
    std::unique_lock<std::mutex> lock(s.mutex);
    const auto ready = [&] {
        const auto iterator = s.channels.find(c);
        return iterator == s.channels.end() || !iterator->second.ready.empty();
    };
    if (!wait_item(s.cv, lock, ms, ready))
        return RK_ERR_AF_NOBUF;
    auto iterator = s.channels.find(c);
    if (iterator == s.channels.end())
        return RK_ERR_AF_UNEXIST;
    if (iterator->second.ready.empty())
        return RK_ERR_AF_NOBUF;
    auto *frame = new (std::nothrow) AUDIO_FRAME_S(iterator->second.ready.front());
    if (!frame)
        return RK_ERR_AF_NOBUF;
    iterator->second.ready.pop_front();
    iterator->second.outstanding[frame] = frame->pMbBlk;
    output->pstFrame = frame;
    output->u32Id = frame->u32Seq;
    s.cv.notify_all();
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AF_ReleaseFrame(AF_CHN c, AUDIO_FRAME_INFO_S *out) {
    if (!out || !out->pstFrame)
        return RK_ERR_AF_NULL_PTR;
    auto &s = af_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.channels.find(c);
    if (i == s.channels.end())
        return RK_ERR_AF_UNEXIST;
    auto j = i->second.outstanding.find(out->pstFrame);
    if (j == i->second.outstanding.end())
        return RK_ERR_AF_ILLEGAL_PARAM;
    release_mb(j->second);
    delete out->pstFrame;
    i->second.outstanding.erase(j);
    out->pstFrame = nullptr;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_AMIX_ListContents(AUDIO_DEV) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AMIX_ListControls(AUDIO_DEV) {
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AMIX_SetControl(AUDIO_DEV d, const char *c, char *v) {
    if (!valid_dev(d) || !c || !v)
        return -3;
    auto &s = amix_state();
    std::lock_guard<std::mutex> l(s.mutex);
    s.controls[d][c] = v;
    return RK_SUCCESS;
}
extern "C" ROCKIT_API RK_S32 RK_MPI_AMIX_GetControl(AUDIO_DEV d, const char *c, char *v) {
    if (!valid_dev(d) || !c || !v)
        return -3;
    auto &s = amix_state();
    std::lock_guard<std::mutex> l(s.mutex);
    auto i = s.controls[d].find(c);
    const std::string value = i == s.controls[d].end() ? "0" : i->second;
    std::memcpy(v, value.c_str(), value.size() + 1);
    return RK_SUCCESS;
}
