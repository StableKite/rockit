#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define ROCKIT_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define ROCKIT_TEST_HIDDEN
#endif

extern "C" ROCKIT_TEST_HIDDEN void
rockit_test_fail_allocation_after(std::int64_t successful_points_before_failure);
extern "C" ROCKIT_TEST_HIDDEN void rockit_test_reset_allocation_failures();
extern "C" ROCKIT_TEST_HIDDEN std::uint64_t rockit_test_allocation_point_count();

bool rockit_test_should_fail_allocation() noexcept;

inline void *rockit_test_calloc(std::size_t count, std::size_t size) noexcept {
    if (rockit_test_should_fail_allocation())
        return nullptr;
    return std::calloc(count, size);
}

template <typename T, typename... Args> T *rockit_test_new_nothrow(Args &&...args) noexcept {
    if (rockit_test_should_fail_allocation())
        return nullptr;
    return new (std::nothrow) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
std::unique_ptr<T> rockit_test_make_unique_nothrow(Args &&...args) noexcept {
    return std::unique_ptr<T>(rockit_test_new_nothrow<T>(std::forward<Args>(args)...));
}
