/*
 * Reconstructed Rockit TGI compatibility types.
 * Based on Apache-2.0 SDK headers shipped in the Rockit source archive.
 */
#pragma once

#include <cstdint>

#define RT_NULL nullptr
#define RT_NONE "none"
#define RT_BOOL char
#define RT_TRUE 1
#define RT_FALSE 0

typedef std::int8_t INT8;
typedef std::int16_t INT16;
typedef std::int32_t INT32;
typedef std::int64_t INT64;
typedef std::uint8_t UINT8;
typedef std::uint8_t UCHAR;
typedef std::uint16_t UINT16;
typedef std::uint32_t UINT32;
typedef std::uint64_t UINT64;
typedef void *RT_PTR;
typedef void RT_VOID;
typedef float RT_FLOAT;
typedef short RT_SHORT;
typedef std::int32_t rt_status;
