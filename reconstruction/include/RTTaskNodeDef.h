#pragma once

#include "RTMediaBufferDef.h"
#include "rt_header.h"

#ifndef MKTAG
#define MKTAG(a, b, c, d) ((a) | ((b) << 8) | ((c) << 16) | ((unsigned)(d) << 24))
#endif

typedef enum _RTStubUid {
    kStubNone = 0,
    kStubDemuxerTMedia = MKTAG('f','m','p','g'),
    kStubDemuxerFile = MKTAG('f','f','i','l'),
    kStubCodecMppDecoder = MKTAG('c','m','d','c'),
    kStubCodecMppEncoder = MKTAG('c','m','e','e'),
    kStubCodecAudioDecoder = MKTAG('c','a','o','d'),
    kStubCodecAudioEncoder = MKTAG('c','a','o','e'),
    kStubSinkVideo = MKTAG('s','v','i','d'),
    kStubSinkAudio = MKTAG('s','v','i','l'),
    kStubSinkFile = MKTAG('s','f','i','l'),
    kStubLinkOutput = MKTAG('l','k','o','p'),
    kStubFilterRKRga = MKTAG('f','r','g','a'),
    kStubFilterResample = MKTAG('f','a','r','s'),
    kStubFilterVolume = MKTAG('f','a','v','l'),
    kStubfilterGLPss = MKTAG('g','l','p','s'),
    kStubfilterVpss = MKTAG('v','p','s','s'),
    kStubFilterAvs = MKTAG('f','a','v','s'),
    kStubFilterMpiVo = MKTAG('f','m','v','o'),
    kStubDeviceV4L2 = MKTAG('d','v','l','v'),
    kStubDeviceALSACap = MKTAG('d','a','l','c'),
    kStubDeviceALSAPlay = MKTAG('d','a','l','p'),
    kStubDeviceExternal = MKTAG('d','e','x','t'),
} RTStubUid;

typedef enum _RTPadType {
    RT_PAD_UNKNOWN,
    RT_PAD_SRC,
    RT_PAD_SINK,
    RT_PAD_LINK,
    RT_PAD_MAX,
} RTPadType;

typedef struct _RTCapPair {
    const char *mKey;
    const char *mValue;
} RTCapPair;

typedef struct _RTPadCaps {
    const char *mMime;
    RTPadType mPadType;
    RTMediaBufferType mBufferType;
    RTCapPair mCapPairs;
} RTPadCaps;
