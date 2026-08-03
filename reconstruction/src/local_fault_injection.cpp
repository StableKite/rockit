#include "local_fault_injection.hpp"

#include <atomic>

namespace {
std::atomic<std::int64_t> g_failure_countdown{-1};
std::atomic<std::uint64_t> g_allocation_points{0};
} // namespace

extern "C" void rockit_test_fail_allocation_after(std::int64_t successful_points_before_failure) {
    g_allocation_points.store(0, std::memory_order_release);
    g_failure_countdown.store(
        successful_points_before_failure < 0 ? -1 : successful_points_before_failure,
        std::memory_order_release);
}

extern "C" void rockit_test_reset_allocation_failures() {
    g_failure_countdown.store(-1, std::memory_order_release);
    g_allocation_points.store(0, std::memory_order_release);
}

extern "C" std::uint64_t rockit_test_allocation_point_count() {
    return g_allocation_points.load(std::memory_order_acquire);
}

bool rockit_test_should_fail_allocation() noexcept {
    g_allocation_points.fetch_add(1, std::memory_order_acq_rel);
    std::int64_t current = g_failure_countdown.load(std::memory_order_acquire);
    for (;;) {
        if (current < 0)
            return false;
        if (current == 0) {
            if (g_failure_countdown.compare_exchange_weak(
                    current, -1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
            continue;
        }
        if (g_failure_countdown.compare_exchange_weak(
                current, current - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return false;
        }
    }
}
