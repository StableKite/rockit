#pragma once

namespace rockit_local {

// Preserve the historical 64-bit packing used by the AArch64 implementation,
// while providing a real mix on 32-bit targets where shifting size_t by 32 is
// ill-formed.  Size is intentionally a template parameter so this header can
// be regression-compiled for a 32-bit ABI without depending on host C++
// standard-library headers.
template <typename Size> constexpr Size hash_combine_word(Size seed, Size value) noexcept {
    static_assert(Size(-1) > Size(0), "hash size type must be unsigned");
    constexpr Size golden = sizeof(Size) > 4U ? static_cast<Size>(0x9e3779b97f4a7c15ULL)
                                              : static_cast<Size>(0x9e3779b9U);
    return seed ^ (value + golden + (seed << 6U) + (seed >> 2U));
}

template <typename Size, typename High, typename Low>
constexpr Size packed_pair_hash(High high, Low low) noexcept {
    static_assert(Size(-1) > Size(0), "hash size type must be unsigned");
    if constexpr (sizeof(Size) > 4U) {
        return (static_cast<Size>(high) << 32U) ^ static_cast<Size>(low);
    }
    return hash_combine_word(static_cast<Size>(high), static_cast<Size>(low));
}

template <typename Size, typename First, typename Second, typename Third>
constexpr Size packed_triplet_hash(First first, Second second, Third third) noexcept {
    static_assert(Size(-1) > Size(0), "hash size type must be unsigned");
    if constexpr (sizeof(Size) > 4U) {
        return (static_cast<Size>(first) << 32U) ^ (static_cast<Size>(second) << 16U) ^
               static_cast<Size>(third);
    }
    Size seed = hash_combine_word(static_cast<Size>(first), static_cast<Size>(second));
    return hash_combine_word(seed, static_cast<Size>(third));
}

} // namespace rockit_local
