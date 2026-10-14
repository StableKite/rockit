// PRIVATE reconstructed runtime contract, NOT a vendor-public C++ ABI test.
#include "RTExecutor.h"
#include "local_fault_injection.hpp"
#include <atomic>
#include <cstdio>
int main() {
    std::atomic<int> count{0};
    { RTThreadPoolExecutor pool(3);
      for (int i=0;i<240;++i) pool.schedule([&count]{count.fetch_add(1);}); }
    if (count.load()!=240) return 1;
    { RTSingleExecutor single(RTThreadOptions{});
      for (int i=0;i<80;++i) single.schedule([&count]{count.fetch_add(1);}); }
    if (count.load()!=320) return 2;
    rockit_test_reset_allocation_failures();
    rockit_test_fail_allocation_after(2);
    if (rockit_test_should_fail_allocation() || rockit_test_should_fail_allocation()) return 3;
    if (!rockit_test_should_fail_allocation() || rockit_test_should_fail_allocation()) return 4;
    if (rockit_test_allocation_point_count()!=4) return 5;
    rockit_test_reset_allocation_failures();
    std::puts("PRIVATE_RUNTIME_PASS callbacks=320; vendor_cpp_abi=NOT_TESTED");
    return 0;
}
