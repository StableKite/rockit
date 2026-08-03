#include "rockit_internal.hpp"
#include "local_fault_injection.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "rk_debug.h"
#include "rk_errno.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_mmz.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_adec.h"
#include "rk_mpi_aenc.h"
#include "rk_mpi_af.h"
#include "rk_mpi_ai.h"
#include "rk_mpi_ao.h"
#include "rk_mpi_avs.h"
#include "rk_mpi_pvs.h"
#include "rk_mpi_venc.h"
#include "rk_mpi_vo.h"
#include "rk_mpi_vpss.h"

namespace {

constexpr std::uint64_t kBlockMagic = 0x524f434b49544d42ULL; // ROCKITMB
// The local fallback uses a 32-bit public length. The top partial page cannot
// be represented by the vendor MMZ provider and is rejected before attempting
// a host allocation. This also avoids Linux overcommit turning UINT32_MAX into
// a misleading successful opaque handle.
constexpr std::uint64_t kMaximumLocalAllocationRequest =
    static_cast<std::uint64_t>(UINT32_MAX) - 4095U;

enum class BlockStorage {
    Host,
    DmaHeap,
    External,
};

enum class AllocationPolicy {
    HostOnly,
    DmaPreferred,
};

#if defined(__linux__)
struct RockitDmaHeapAllocationData {
    std::uint64_t len;
    std::uint32_t fd;
    std::uint32_t fd_flags;
    std::uint64_t heap_flags;
};

struct RockitDmaBufSync {
    std::uint64_t flags;
};

#ifndef ROCKIT_DMA_HEAP_IOCTL_ALLOC
#define ROCKIT_DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct RockitDmaHeapAllocationData)
#endif
#ifndef ROCKIT_DMA_BUF_IOCTL_SYNC
#define ROCKIT_DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct RockitDmaBufSync)
#endif

constexpr std::uint64_t kDmaBufSyncRead = 1U << 0;
constexpr std::uint64_t kDmaBufSyncWrite = 2U << 0;
constexpr std::uint64_t kDmaBufSyncEnd = 1U << 2;
#endif

bool environment_equals(const char *name, const char *expected) noexcept {
    const char *value = std::getenv(name);
    return value && std::strcmp(value, expected) == 0;
}

bool dma_heap_required() noexcept {
    return environment_equals("ROCKIT_MMZ_PROVIDER", "dma_heap") ||
           environment_equals("ROCKIT_MMZ_PROVIDER", "required-dma_heap");
}

bool dma_heap_allowed() noexcept {
    if (environment_equals("ROCKIT_MMZ_PROVIDER", "host"))
        return false;
    if (dma_heap_required() || environment_equals("ROCKIT_MMZ_PROVIDER", "auto")) {
        return true;
    }
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

struct ChannelKey {
    int mod;
    int dev;
    int chn;

    bool operator==(const ChannelKey &other) const noexcept {
        return mod == other.mod && dev == other.dev && chn == other.chn;
    }
};

struct ChannelKeyHash {
    std::size_t operator()(const ChannelKey &key) const noexcept {
        std::size_t h = static_cast<std::size_t>(key.mod);
        h = h * 1315423911u + static_cast<std::size_t>(key.dev);
        h = h * 1315423911u + static_cast<std::size_t>(key.chn);
        return h;
    }
};

ChannelKey make_key(const MPP_CHN_S &chn) noexcept {
    return {static_cast<int>(chn.enModId), chn.s32DevId, chn.s32ChnId};
}

struct Pool;

struct Block {
    std::uint64_t magic{kBlockMagic};
    std::atomic<int> refs{1};
    std::uint64_t size{0};
    std::uint64_t length{0};
    std::uint32_t offset{0};
    std::uint32_t horizontal_stride{0};
    std::uint32_t vertical_stride{0};
    MB_POOL pool_id{MB_INVALID_POOLID};
    int unique_id{-1};
    int fd{-1};
    std::uint64_t phys{0};
    void *data{nullptr};
    std::size_t mapped_size{0};
    BlockStorage storage{BlockStorage::Host};
    bool owns_data{false};
    bool owns_fd{false};
    bool cacheable{true};
    bool in_pool_free_list{false};
    RK_MPI_MB_FREE_CB free_cb{nullptr};
    void *opaque{nullptr};
    Pool *pool{nullptr};
};

struct Pool {
    MB_POOL id{MB_INVALID_POOLID};
    MB_POOL_CONFIG_S config{};
    std::vector<Block *> blocks;
    std::vector<Block *> free_blocks;
    std::condition_variable cv;
    bool destroying{false};
};

struct State {
    std::mutex mutex;
    std::atomic<int> sys_refs{0};
    bool mb_initialized{false};
    MB_CONFIG_S common_config{};
    bool common_config_set{false};
    std::unordered_map<int, MB_CONFIG_S> module_configs;
    std::unordered_map<MB_POOL, std::unique_ptr<Pool>> pools;
    std::unordered_multimap<void *, Block *> by_data;
    std::unordered_multimap<int, Block *> by_fd;
    std::unordered_map<int, Block *> by_unique;
    std::unordered_map<std::uint64_t, Block *> by_phys;
    std::unordered_set<Block *> live_blocks;
    std::unordered_map<ChannelKey, ChannelKey, ChannelKeyHash> bind_by_dest;
    std::unordered_map<ChannelKey, std::vector<ChannelKey>, ChannelKeyHash> binds_by_source;
    std::unordered_map<ChannelKey, CHN_INPUT_MODE_E, ChannelKeyHash> input_modes;
    std::unordered_map<int, int> device_priorities;
    LOG_LEVEL_CONF_S log_config{};
    MB_POOL next_pool{0};
    int next_unique{1};
    int next_fd{1000};
    std::chrono::steady_clock::time_point pts_anchor{std::chrono::steady_clock::now()};
    std::uint64_t pts_base{0};
};

State &state() {
    static State instance;
    return instance;
}

Block *as_block_locked(State &s, MB_BLK handle) noexcept {
    auto *block = static_cast<Block *>(handle);
    if (!block || s.live_blocks.find(block) == s.live_blocks.end())
        return nullptr;
    return block->magic == kBlockMagic ? block : nullptr;
}

void register_block_locked(State &s, Block *block) {
    s.live_blocks.insert(block);
    if (block->data)
        s.by_data.emplace(block->data, block);
    if (block->fd >= 0)
        s.by_fd.emplace(block->fd, block);
    if (block->unique_id >= 0)
        s.by_unique[block->unique_id] = block;
    if (block->phys != 0)
        s.by_phys[block->phys] = block;
}

void unregister_block_locked(State &s, Block *block) {
    s.live_blocks.erase(block);
    if (block->data) {
        const auto range = s.by_data.equal_range(block->data);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == block) {
                s.by_data.erase(it);
                break;
            }
        }
    }
    if (block->fd >= 0) {
        const auto range = s.by_fd.equal_range(block->fd);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == block) {
                s.by_fd.erase(it);
                break;
            }
        }
    }
    if (block->unique_id >= 0)
        s.by_unique.erase(block->unique_id);
    if (block->phys != 0)
        s.by_phys.erase(block->phys);
}

Block *allocate_host_block_locked(
    State &s, std::uint64_t size, bool cacheable, MB_POOL pool_id, Pool *pool) {
    if (size > static_cast<std::uint64_t>(SIZE_MAX) || size > kMaximumLocalAllocationRequest)
        return nullptr;
    auto *block = rockit_test_new_nothrow<Block>();
    if (!block)
        return nullptr;
    const std::size_t allocation_size = size == 0 ? 1U : static_cast<std::size_t>(size);
    block->data = rockit_test_calloc(1, allocation_size);
    if (!block->data) {
        delete block;
        return nullptr;
    }
    block->size = size;
    block->length = size;
    block->mapped_size = allocation_size;
    block->pool_id = pool_id;
    block->pool = pool;
    block->storage = BlockStorage::Host;
    block->owns_data = true;
    block->cacheable = cacheable;
    block->unique_id = s.next_unique++;
    while (s.by_fd.find(s.next_fd) != s.by_fd.end())
        ++s.next_fd;
    block->fd = s.next_fd++;
    block->phys = reinterpret_cast<std::uintptr_t>(block->data);
    register_block_locked(s, block);
    return block;
}

#if defined(__linux__)
Block *allocate_dma_heap_block_locked(
    State &s, std::uint64_t size, bool cacheable, MB_POOL pool_id, Pool *pool) {
    if (size == 0 || size > kMaximumLocalAllocationRequest ||
        size > static_cast<std::uint64_t>(SIZE_MAX)) {
        return nullptr;
    }

    const char *const cacheable_heaps[] = {
        "/dev/dma_heap/system",
        "/dev/dma_heap/reserved",
    };
    const char *const uncached_heaps[] = {
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/system",
        "/dev/dma_heap/reserved",
    };
    const char *const *heaps = cacheable ? cacheable_heaps : uncached_heaps;
    const std::size_t heap_count = cacheable ? sizeof(cacheable_heaps) / sizeof(cacheable_heaps[0])
                                             : sizeof(uncached_heaps) / sizeof(uncached_heaps[0]);

    int buffer_fd = -1;
    for (std::size_t index = 0; index < heap_count; ++index) {
        const int heap_fd = ::open(heaps[index], O_RDWR | O_CLOEXEC);
        if (heap_fd < 0)
            continue;
        RockitDmaHeapAllocationData request{};
        request.len = size;
        request.fd_flags = O_RDWR | O_CLOEXEC;
        request.heap_flags = 0;
        const int result = ::ioctl(heap_fd, ROCKIT_DMA_HEAP_IOCTL_ALLOC, &request);
        const int saved_errno = errno;
        (void)::close(heap_fd);
        errno = saved_errno;
        if (result == 0 && static_cast<int>(request.fd) >= 0) {
            buffer_fd = static_cast<int>(request.fd);
            break;
        }
    }
    if (buffer_fd < 0)
        return nullptr;

    void *mapping = ::mmap(
        nullptr, static_cast<std::size_t>(size), PROT_READ | PROT_WRITE, MAP_SHARED, buffer_fd, 0);
    if (mapping == MAP_FAILED) {
        (void)::close(buffer_fd);
        return nullptr;
    }

    auto *block = rockit_test_new_nothrow<Block>();
    if (!block) {
        (void)::munmap(mapping, static_cast<std::size_t>(size));
        (void)::close(buffer_fd);
        return nullptr;
    }
    block->size = size;
    block->length = size;
    block->mapped_size = static_cast<std::size_t>(size);
    block->pool_id = pool_id;
    block->pool = pool;
    block->storage = BlockStorage::DmaHeap;
    block->data = mapping;
    block->fd = buffer_fd;
    block->owns_fd = true;
    block->cacheable = cacheable;
    block->unique_id = s.next_unique++;
    // Linux dma-buf intentionally does not expose a stable physical address.
    block->phys = 0;
    register_block_locked(s, block);
    return block;
}
#endif

Block *allocate_owned_block_locked(State &s,
                                   std::uint64_t size,
                                   bool cacheable,
                                   MB_POOL pool_id,
                                   Pool *pool,
                                   AllocationPolicy policy) {
#if defined(__linux__)
    if (policy == AllocationPolicy::DmaPreferred && dma_heap_allowed()) {
        if (Block *block = allocate_dma_heap_block_locked(s, size, cacheable, pool_id, pool)) {
            return block;
        }
        if (dma_heap_required())
            return nullptr;
    }
#else
    (void)policy;
#endif
    return allocate_host_block_locked(s, size, cacheable, pool_id, pool);
}

void destroy_block_locked(State &s, Block *block) {
    if (!block)
        return;
    unregister_block_locked(s, block);
    block->magic = 0;
    if (block->free_cb) {
        block->free_cb(block->opaque);
    } else if (block->storage == BlockStorage::DmaHeap) {
#if defined(__linux__)
        if (block->data && block->mapped_size != 0) {
            (void)::munmap(block->data, block->mapped_size);
        }
        if (block->owns_fd && block->fd >= 0) {
            (void)::close(block->fd);
        }
#endif
    } else if (block->owns_data) {
        std::free(block->data);
    }
    delete block;
}

RK_S32 release_block_locked(State &s, Block *block) {
    if (!block)
        return RK_ERR_MB_UNEXIST;
    const int previous = block->refs.fetch_sub(1, std::memory_order_acq_rel);
    if (previous <= 0) {
        block->refs.store(0, std::memory_order_release);
        return RK_ERR_MB_UNEXIST;
    }
    if (previous != 1)
        return RK_SUCCESS;

    if (block->pool) {
        block->offset = 0;
        block->length = block->size;
        block->horizontal_stride = 0;
        block->vertical_stride = 0;
        block->in_pool_free_list = true;
        block->pool->free_blocks.push_back(block);
        block->pool->cv.notify_one();
    } else {
        destroy_block_locked(s, block);
    }
    return RK_SUCCESS;
}

MB_BLK create_standalone(std::uint64_t size, bool cacheable, AllocationPolicy policy) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return static_cast<MB_BLK>(
        allocate_owned_block_locked(s, size, cacheable, MB_INVALID_POOLID, nullptr, policy));
}

bool binding_path_exists_locked(const State &s, const ChannelKey &from, const ChannelKey &target) {
    std::vector<ChannelKey> pending{from};
    std::unordered_set<ChannelKey, ChannelKeyHash> visited;
    while (!pending.empty()) {
        const ChannelKey current = pending.back();
        pending.pop_back();
        if (current == target)
            return true;
        if (!visited.insert(current).second)
            continue;
        const auto iterator = s.binds_by_source.find(current);
        if (iterator == s.binds_by_source.end())
            continue;
        pending.insert(pending.end(), iterator->second.begin(), iterator->second.end());
    }
    return false;
}

std::vector<MPP_CHN_S> bound_destinations(const MPP_CHN_S &source) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::vector<MPP_CHN_S> result;
    const auto iterator = s.binds_by_source.find(make_key(source));
    if (iterator == s.binds_by_source.end())
        return result;
    result.reserve(iterator->second.size());
    for (const ChannelKey &key : iterator->second) {
        result.push_back({static_cast<MOD_ID_E>(key.mod), key.dev, key.chn});
    }
    return result;
}

RK_S32 validate_range(Block *block, RK_U32 offset, RK_U32 length) noexcept {
    if (!block)
        return RK_ERR_MB_UNEXIST;
    if (static_cast<std::uint64_t>(offset) > block->size)
        return RK_ERR_MB_ILLEGAL_PARAM;
    if (length != 0 && static_cast<std::uint64_t>(offset) + length > block->size)
        return RK_ERR_MB_ILLEGAL_PARAM;
    return RK_SUCCESS;
}

RK_S32
sync_block_locked(Block *block, RK_U32 offset, RK_U32 length, RK_U32 flags, bool end) noexcept {
    const RK_S32 range_result = validate_range(block, offset, length);
    if (range_result != RK_SUCCESS)
        return range_result;
#if defined(__linux__)
    if (block->storage == BlockStorage::DmaHeap) {
        std::uint64_t direction = kDmaBufSyncRead;
        if (flags == RK_MMZ_SYNC_WRITEONLY) {
            direction = kDmaBufSyncWrite;
        } else if (flags == RK_MMZ_SYNC_RW) {
            direction = kDmaBufSyncRead | kDmaBufSyncWrite;
        }
        RockitDmaBufSync request{};
        request.flags = direction | (end ? kDmaBufSyncEnd : 0U);
        return ::ioctl(block->fd, ROCKIT_DMA_BUF_IOCTL_SYNC, &request) == 0 ? RK_SUCCESS
                                                                            : RK_FAILURE;
    }
#else
    (void)flags;
    (void)end;
#endif
    return RK_SUCCESS;
}

} // namespace

namespace rockit_local {

// The recursion guard and fan-out lifecycle are identical for every routed
// payload type.  Only the per-destination dispatch belongs at each call site.
#define ROCKIT_ROUTE_BOUND_DESTINATIONS(source, ...)                                               \
    thread_local std::vector<ChannelKey> activeSources;                                            \
    const ChannelKey sourceKey = make_key(source);                                                 \
    if (activeSources.size() >= 16 ||                                                              \
        std::find(activeSources.begin(), activeSources.end(), sourceKey) != activeSources.end()) { \
        return RK_ERR_SYS_BUSY;                                                                    \
    }                                                                                              \
    const auto destinations = bound_destinations(source);                                          \
    if (destinations.empty())                                                                      \
        return RK_SUCCESS;                                                                         \
    activeSources.push_back(sourceKey);                                                            \
    struct PopGuard {                                                                              \
        std::vector<ChannelKey> &stack;                                                            \
        ~PopGuard() {                                                                              \
            stack.pop_back();                                                                      \
        }                                                                                          \
    } guard{activeSources};                                                                        \
    for (const MPP_CHN_S &destination : destinations) {                                            \
        RK_S32 result = RK_ERR_SYS_NOT_SUPPORT;                                                    \
        __VA_ARGS__                                                                                \
        if (result != RK_SUCCESS)                                                                  \
            return result;                                                                         \
    }                                                                                              \
    return RK_SUCCESS

bool system_initialized() noexcept {
    return state().sys_refs.load(std::memory_order_acquire) > 0;
}

RK_S32 route_video_frame(const MPP_CHN_S &source, const VIDEO_FRAME_INFO_S &frame, RK_S32 timeout) {
    if (timeout < -1 || frame.stVFrame.pMbBlk == MB_INVALID_HANDLE) {
        return RK_ERR_SYS_ILLEGAL_PARAM;
    }
    ROCKIT_ROUTE_BOUND_DESTINATIONS(source, {
        switch (destination.enModId) {
        case RK_ID_VPSS:
            result =
                RK_MPI_VPSS_SendFrame(destination.s32DevId, destination.s32ChnId, &frame, timeout);
            break;
        case RK_ID_VENC:
            result = RK_MPI_VENC_SendFrame(destination.s32ChnId, &frame, timeout);
            break;
        case RK_ID_VO: {
            VIDEO_FRAME_INFO_S writable = frame;
            result =
                RK_MPI_VO_SendFrame(destination.s32DevId, destination.s32ChnId, &writable, timeout);
            break;
        }
        case RK_ID_AVS:
            result = RK_MPI_AVS_SendPipeFrame(
                destination.s32DevId, destination.s32ChnId, &frame, timeout);
            break;
        case RK_ID_PVS:
            result = RK_MPI_PVS_SendFrame(destination.s32DevId, destination.s32ChnId, &frame);
            break;
        default:
            break;
        }
    });
}

RK_S32 route_audio_frame(const MPP_CHN_S &source, const AUDIO_FRAME_S &frame, RK_S32 timeout) {
    if (timeout < -1 || frame.pMbBlk == MB_INVALID_HANDLE) {
        return RK_ERR_SYS_ILLEGAL_PARAM;
    }
    ROCKIT_ROUTE_BOUND_DESTINATIONS(source, {
        switch (destination.enModId) {
        case RK_ID_AI:
            result =
                RK_MPI_AI_SendFrame(destination.s32DevId, destination.s32ChnId, &frame, timeout);
            break;
        case RK_ID_AO:
            result =
                RK_MPI_AO_SendFrame(destination.s32DevId, destination.s32ChnId, &frame, timeout);
            break;
        case RK_ID_AENC:
            result = RK_MPI_AENC_SendFrame(destination.s32ChnId, &frame, nullptr, timeout);
            break;
        case RK_ID_AF: {
            AUDIO_FRAME_S writable = frame;
            AUDIO_FRAME_INFO_S info{};
            info.pstFrame = &writable;
            result = RK_MPI_AF_SendFrame(destination.s32ChnId, &info, timeout);
            break;
        }
        default:
            break;
        }
    });
}

RK_S32 route_audio_stream(const MPP_CHN_S &source, const AUDIO_STREAM_S &stream) {
    if (stream.pMbBlk == MB_INVALID_HANDLE)
        return RK_ERR_SYS_ILLEGAL_PARAM;
    ROCKIT_ROUTE_BOUND_DESTINATIONS(source, {
        if (destination.enModId == RK_ID_ADEC) {
            result = RK_MPI_ADEC_SendStream(destination.s32ChnId, &stream, RK_TRUE);
        }
    });
}

#undef ROCKIT_ROUTE_BOUND_DESTINATIONS

} // namespace rockit_local

namespace {
bool zero_mpp_channel(const MPP_CHN_S &channel) {
    return static_cast<int>(channel.enModId) == 0 && channel.s32DevId == 0 && channel.s32ChnId == 0;
}
} // namespace

extern "C" ROCKIT_API void
RK_LOG(RK_S32 level, RK_S32 modId, const char *fmt, const char *fname, const RK_U32 row, ...) {
    std::fprintf(stderr, "[rockit:%d:%d] %s:%u: ", level, modId, fname ? fname : "?", row);
    va_list ap;
    va_start(ap, row);
    std::vfprintf(stderr, fmt ? fmt : "", ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_Init(RK_VOID) {
    auto &s = state();
    s.sys_refs.fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lock(s.mutex);
    s.mb_initialized = true;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_Exit(RK_VOID) {
    auto &s = state();
    int current = s.sys_refs.load(std::memory_order_acquire);
    while (current > 0 &&
           !s.sys_refs.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel)) {}
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_DumpSys(const RK_CHAR *, RK_CHAR *, RK_U32) {
    // Vendor AArch64 librockit.so and rk_mpi_sys.cpp.o both implement this
    // public entry point as an unconditional RK_FAILURE and do not touch the
    // caller-provided buffer. Keep the richer local diagnostics internal.
    return RK_FAILURE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_Bind(const MPP_CHN_S *src, const MPP_CHN_S *dst) {
    if (!src || !dst)
        return RK_ERR_SYS_NULL_PTR;
    if (static_cast<int>(src->enModId) == 0 || static_cast<int>(dst->enModId) == 0)
        return RK_ERR_SYS_NOT_SUPPORT;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto sk = make_key(*src);
    const auto dk = make_key(*dst);
    if (sk == dk || binding_path_exists_locked(s, dk, sk)) {
        return RK_ERR_SYS_ILLEGAL_PARAM;
    }
    const auto existing = s.bind_by_dest.find(dk);
    if (existing != s.bind_by_dest.end()) {
        return existing->second == sk ? RK_SUCCESS : RK_ERR_SYS_BUSY;
    }
    s.bind_by_dest[dk] = sk;
    auto &destinations = s.binds_by_source[sk];
    if (std::find(destinations.begin(), destinations.end(), dk) == destinations.end()) {
        destinations.push_back(dk);
    }
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_UnBind(const MPP_CHN_S *src, const MPP_CHN_S *dst) {
    if (!src || !dst)
        return RK_ERR_SYS_NULL_PTR;
    if (zero_mpp_channel(*src) && zero_mpp_channel(*dst))
        return RK_SUCCESS;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto sk = make_key(*src);
    const auto dk = make_key(*dst);
    auto it = s.bind_by_dest.find(dk);
    if (it == s.bind_by_dest.end() || !(it->second == sk))
        return RK_ERR_SYS_NOTREADY;
    s.bind_by_dest.erase(it);
    auto source_it = s.binds_by_source.find(sk);
    if (source_it != s.binds_by_source.end()) {
        auto &v = source_it->second;
        v.erase(std::remove(v.begin(), v.end(), dk), v.end());
        if (v.empty())
            s.binds_by_source.erase(source_it);
    }
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_GetBindbyDest(const MPP_CHN_S *dst, MPP_CHN_S *src) {
    if (!dst || !src)
        return RK_ERR_SYS_ILLEGAL_PARAM;
    if (zero_mpp_channel(*dst))
        return RK_ERR_SYS_ILLEGAL_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.bind_by_dest.find(make_key(*dst));
    if (it == s.bind_by_dest.end())
        return RK_ERR_SYS_NOTREADY;
    src->enModId = static_cast<MOD_ID_E>(it->second.mod);
    src->s32DevId = it->second.dev;
    src->s32ChnId = it->second.chn;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_GetBindbySrc(const MPP_CHN_S *src, MPP_BIND_DEST_S *dest) {
    if (!src || !dest)
        return RK_ERR_SYS_ILLEGAL_PARAM;
    if (zero_mpp_channel(*src))
        return RK_SUCCESS;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.binds_by_source.find(make_key(*src));
    if (it == s.binds_by_source.end())
        return RK_ERR_SYS_NOTREADY;
    std::memset(dest, 0, sizeof(*dest));
    const std::size_t count = std::min<std::size_t>(
        it->second.size(), sizeof(dest->astMppChn) / sizeof(dest->astMppChn[0]));
    dest->u32Num = static_cast<RK_U32>(count);
    for (std::size_t i = 0; i < count; ++i) {
        dest->astMppChn[i].enModId = static_cast<MOD_ID_E>(it->second[i].mod);
        dest->astMppChn[i].s32DevId = it->second[i].dev;
        dest->astMppChn[i].s32ChnId = it->second[i].chn;
    }
    return RK_SUCCESS;
}

RK_S32 mmz_allocate_with_provider_order(MB_BLK *destination,
                                        RK_U32 length,
                                        bool cacheable,
                                        RK_S32 null_error) {
    // The vendor provider attempts to create the MB first. Consequently an
    // impossible allocation reports the MB-domain NOBUF code even when the
    // destination pointer is null. We preserve that observable ordering while
    // keeping the candidate safer on vendor-crashing null/zero requests.
    MB_BLK created = create_standalone(length, cacheable, AllocationPolicy::DmaPreferred);
    if (created == MB_INVALID_HANDLE)
        return RK_ERR_MB_NOBUF;
    if (!destination) {
        (void)RK_MPI_MB_ReleaseMB(created);
        return null_error;
    }
    *destination = created;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_MmzAlloc(MB_BLK *blk,
                                                 const RK_CHAR *,
                                                 const RK_CHAR *,
                                                 RK_U32 len) {
    return mmz_allocate_with_provider_order(blk, len, false, RK_ERR_SYS_NULL_PTR);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_MmzAlloc_Cached(MB_BLK *blk,
                                                        const RK_CHAR *,
                                                        const RK_CHAR *,
                                                        RK_U32 len) {
    return mmz_allocate_with_provider_order(blk, len, true, RK_ERR_SYS_NULL_PTR);
}

extern "C" ROCKIT_API RK_S32
RK_MPI_SYS_MmzAllocEx(MB_BLK *blk, const RK_CHAR *, const RK_CHAR *, RK_U32 len, RK_U32 flags) {
    const bool cacheable = (flags & MB_REMAP_MODE_NOCACHE) == 0;
    return mmz_allocate_with_provider_order(blk, len, cacheable, RK_ERR_SYS_NULL_PTR);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_MmzFree(MB_BLK blk) {
    return RK_MPI_MB_ReleaseMB(blk);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_MmzFlushCache(MB_BLK blk, RK_BOOL) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, blk);
    if (!block)
        return RK_ERR_MB_NULL_PTR;
    const RK_S32 start = sync_block_locked(block, 0, 0, RK_MMZ_SYNC_RW, false);
    return start == RK_SUCCESS ? sync_block_locked(block, 0, 0, RK_MMZ_SYNC_RW, true) : start;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_Malloc(MB_BLK *blk, RK_U32 len) {
    if (!blk)
        return RK_ERR_SYS_NULL_PTR;
    *blk = create_standalone(len, true, AllocationPolicy::HostOnly);
    return *blk ? RK_SUCCESS : RK_ERR_SYS_NOMEM;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_Free(MB_BLK blk) {
    return RK_MPI_MB_ReleaseMB(blk);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_CreateMB(MB_BLK *blk, MB_EXT_CONFIG_S *config) {
    if (!blk || !config)
        return RK_ERR_SYS_NULL_PTR;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto *block = rockit_test_new_nothrow<Block>();
    if (!block)
        return RK_ERR_SYS_NOMEM;
    block->size = config->u64Size;
    block->length = config->u64Size;
    block->data = config->pu8VirAddr;
    block->mapped_size =
        static_cast<std::size_t>(std::min<RK_U64>(config->u64Size, static_cast<RK_U64>(SIZE_MAX)));
    block->storage = BlockStorage::External;
    // External DMA-BUF providers do not expose a stable physical address.
    // Never report a virtual pointer as a physical address.
    block->phys = config->u64PhyAddr;
    block->fd = config->s32Fd;
    block->unique_id = s.next_unique++;
    block->free_cb = config->pFreeCB;
    block->opaque = config->pOpaque;
    register_block_locked(s, block);
    *blk = static_cast<MB_BLK>(block);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_GetCurPTS(RK_U64 *pts) {
    if (!pts)
        return RK_ERR_SYS_NULL_PTR;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - s.pts_anchor)
                             .count();
    *pts = s.pts_base + static_cast<std::uint64_t>(std::max<std::int64_t>(0, elapsed));
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_InitPTSBase(RK_U64 base) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.pts_base = base;
    s.pts_anchor = std::chrono::steady_clock::now();
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_SyncPTS(RK_U64 base) {
    return RK_MPI_SYS_InitPTSBase(base);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_SetChnInputMode(const MPP_CHN_S *chn,
                                                        CHN_INPUT_MODE_E mode) {
    if (!chn)
        return RK_ERR_SYS_NULL_PTR;
    if (mode < CHN_INPUT_MODE_NORMAL || mode >= CHN_INPUT_MODE_BUTT)
        return RK_ERR_SYS_ILLEGAL_PARAM;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.input_modes[make_key(*chn)] = mode;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_LOG_SetLevelConf(LOG_LEVEL_CONF_S *config) {
    if (!config)
        return RK_ERR_SYS_NULL_PTR;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.log_config = *config;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_LOG_GetLevelConf(LOG_LEVEL_CONF_S *config) {
    if (!config)
        return RK_ERR_SYS_NULL_PTR;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    *config = s.log_config;
    return RK_SUCCESS;
}

#define ROCKIT_LOCK_DEVICE_PRIORITY(config_)                                                       \
    if (!(config_))                                                                                \
        return RK_ERR_SYS_NULL_PTR;                                                                \
    const int device = static_cast<int>((config_)->enDeviceId);                                    \
    if (device < 1 || device > 4)                                                                  \
        return RK_ERR_SYS_NOT_SUPPORT;                                                             \
    if ((config_)->s32Level < 0 || (config_)->s32Level > 5 ||                                     \
        (device == 1 && (config_)->s32Level == 5))                                                 \
        return RK_ERR_SYS_ILLEGAL_PARAM;                                                           \
    auto &s = state();                                                                             \
    std::lock_guard<std::mutex> lock(s.mutex)

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_SetDevicePriority(DEVICE_PRIORITY_LEVEL_CONFIG_S *config) {
    ROCKIT_LOCK_DEVICE_PRIORITY(config);
    s.device_priorities[static_cast<int>(config->enDeviceId)] = config->s32Level;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_SYS_GetDevicePriority(DEVICE_PRIORITY_LEVEL_CONFIG_S *config) {
    ROCKIT_LOCK_DEVICE_PRIORITY(config);
    auto it = s.device_priorities.find(static_cast<int>(config->enDeviceId));
    config->s32Level = it == s.device_priorities.end() ? 0 : it->second;
    return RK_SUCCESS;
}
#undef ROCKIT_LOCK_DEVICE_PRIORITY

extern "C" ROCKIT_API MB_POOL RK_MPI_MB_CreatePool(MB_POOL_CONFIG_S *config) {
    // A non-null all-zero descriptor is a valid empty vendor pool. It owns no
    // blocks, but still receives a provider-local ID and can be destroyed.
    if (!config || config->u32MBCnt > MB_MAX_COUNT)
        return MB_INVALID_POOLID;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const MB_POOL id = s.next_pool++;
    auto pool = rockit_test_make_unique_nothrow<Pool>();
    if (!pool)
        return MB_INVALID_POOLID;
    pool->id = id;
    pool->config = *config;
    if (config->bPreAlloc) {
        for (RK_U32 i = 0; i < config->u32MBCnt; ++i) {
            Block *block = allocate_owned_block_locked(s,
                                                       config->u64MBSize,
                                                       config->enRemapMode != MB_REMAP_MODE_NOCACHE,
                                                       id,
                                                       pool.get(),
                                                       config->enAllocType == MB_ALLOC_TYPE_DMA
                                                           ? AllocationPolicy::DmaPreferred
                                                           : AllocationPolicy::HostOnly);
            if (!block) {
                for (Block *created : pool->blocks)
                    destroy_block_locked(s, created);
                return MB_INVALID_POOLID;
            }
            block->refs.store(0);
            block->in_pool_free_list = true;
            pool->blocks.push_back(block);
            pool->free_blocks.push_back(block);
        }
    }
    s.pools.emplace(id, std::move(pool));
    return id;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_DestroyPool(MB_POOL pool_id) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.pools.find(pool_id);
    if (it == s.pools.end())
        return static_cast<RK_S32>(UINT32_C(0xA0018040));
    Pool *pool = it->second.get();
    for (Block *block : pool->blocks) {
        if (block->refs.load(std::memory_order_acquire) > 0)
            return RK_ERR_MB_BUSY;
    }
    for (Block *block : pool->blocks)
        destroy_block_locked(s, block);
    s.pools.erase(it);
    return RK_SUCCESS;
}

extern "C" ROCKIT_API MB_BLK RK_MPI_MB_GetMB(MB_POOL pool_id, RK_U64 size, RK_BOOL block_wait) {
    auto &s = state();
    std::unique_lock<std::mutex> lock(s.mutex);
    auto it = s.pools.find(pool_id);
    if (it == s.pools.end())
        return MB_INVALID_HANDLE;
    Pool *pool = it->second.get();
    const std::uint64_t requested = size == 0 ? pool->config.u64MBSize : size;
    if (requested > pool->config.u64MBSize)
        return MB_INVALID_HANDLE;

    for (;;) {
        if (!pool->free_blocks.empty()) {
            Block *result = pool->free_blocks.back();
            pool->free_blocks.pop_back();
            result->refs.store(1, std::memory_order_release);
            result->length = requested;
            result->offset = 0;
            result->in_pool_free_list = false;
            return static_cast<MB_BLK>(result);
        }
        if (pool->blocks.size() < pool->config.u32MBCnt) {
            Block *result = allocate_owned_block_locked(
                s,
                pool->config.u64MBSize,
                pool->config.enRemapMode != MB_REMAP_MODE_NOCACHE,
                pool_id,
                pool,
                pool->config.enAllocType == MB_ALLOC_TYPE_DMA ? AllocationPolicy::DmaPreferred
                                                              : AllocationPolicy::HostOnly);
            if (!result)
                return MB_INVALID_HANDLE;
            result->length = requested;
            pool->blocks.push_back(result);
            return static_cast<MB_BLK>(result);
        }
        if (!block_wait)
            return MB_INVALID_HANDLE;
        pool->cv.wait(lock, [&] {
            return !pool->free_blocks.empty() || pool->destroying;
        });
        if (pool->destroying)
            return MB_INVALID_HANDLE;
    }
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_ReleaseMB(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? release_block_locked(s, block) : RK_ERR_MB_NULL_PTR;
}

extern "C" ROCKIT_API RK_U64 RK_MPI_MB_Handle2PhysAddr(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->phys : 0;
}

extern "C" ROCKIT_API RK_VOID *RK_MPI_MB_Handle2VirAddr(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->data : nullptr;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_Handle2Fd(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->fd : -1;
}

extern "C" ROCKIT_API MB_POOL RK_MPI_MB_Handle2PoolId(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->pool_id : MB_INVALID_POOLID;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_Handle2UniqueId(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->unique_id : 0;
}

extern "C" ROCKIT_API RK_U64 RK_MPI_MB_GetSize(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->size : UINT64_MAX;
}

extern "C" ROCKIT_API RK_U64 RK_MPI_MB_GetLength(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? static_cast<RK_U32>(block->length) : UINT32_MAX;
}

extern "C" ROCKIT_API RK_U32 RK_MPI_MB_GetOffset(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->offset : UINT32_MAX;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_SetOffset(MB_BLK handle, RK_U32 offset) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    if (!block)
        return RK_ERR_MB_NULL_PTR;
    if (offset > block->size)
        return RK_ERR_MB_ILLEGAL_PARAM;
    block->offset = offset;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API MB_BLK RK_MPI_MB_VirAddr2Handle(RK_VOID *address) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto exact = s.by_data.find(address);
    if (exact != s.by_data.end())
        return static_cast<MB_BLK>(exact->second);
    const auto target = reinterpret_cast<std::uintptr_t>(address);
    for (const auto &entry : s.by_data) {
        const auto begin = reinterpret_cast<std::uintptr_t>(entry.first);
        if (target >= begin && target < begin + entry.second->size)
            return static_cast<MB_BLK>(entry.second);
    }
    return MB_INVALID_HANDLE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_UniqueId2Fd(RK_S32 unique_id) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.by_unique.find(unique_id);
    return it == s.by_unique.end() ? -1 : it->second->fd;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_InquireUserCnt(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    return block ? block->refs.load(std::memory_order_acquire) : RK_ERR_MB_NULL_PTR;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_AddUserCnt(MB_BLK handle) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    if (!block)
        return RK_ERR_MB_NULL_PTR;
    if (block->refs.load(std::memory_order_acquire) <= 0)
        return RK_ERR_MB_UNEXIST;
    return block->refs.fetch_add(1, std::memory_order_acq_rel) + 1;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_SetModPoolConfig(MB_UID_E, const MB_CONFIG_S *) {
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_GetModPoolConfig(MB_UID_E, MB_CONFIG_S *) {
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_Init() {
    // The vendor runtime has already initialized its global MB provider before
    // public entry points are available; an explicit second initialization is
    // rejected with NOT_PERM in all three isolated oracle profiles.
    return RK_ERR_MB_NOT_PERM;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_Exit() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto &entry : s.pools) {
        for (Block *block : entry.second->blocks) {
            if (block->refs.load(std::memory_order_acquire) > 0)
                return RK_ERR_MB_BUSY;
        }
    }
    s.mb_initialized = false;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_SetConfig(const MB_CONFIG_S *config) {
    if (!config)
        return RK_ERR_MB_NULL_PTR;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.common_config = *config;
    s.common_config_set = true;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_GetConfig(MB_CONFIG_S *config) {
    if (!config)
        return RK_ERR_MB_NULL_PTR;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.common_config_set)
        return RK_ERR_MB_NOT_PERM;
    *config = s.common_config;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MB_SetBufferStride(MB_BLK handle,
                                                       RK_U32 horizontal,
                                                       RK_U32 vertical) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, handle);
    if (!block)
        return RK_ERR_MB_NULL_PTR;
    block->horizontal_stride = horizontal;
    block->vertical_stride = vertical;
    return RK_SUCCESS;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MMZ_Alloc(MB_BLK *blk, RK_U32 length, RK_U32 flags) {
    const bool cacheable = (flags & RK_MMZ_ALLOC_UNCACHEABLE) == 0;
    return mmz_allocate_with_provider_order(blk, length, cacheable, RK_ERR_MB_NULL_PTR);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MMZ_Free(MB_BLK blk) {
    return RK_MPI_MB_ReleaseMB(blk);
}

extern "C" ROCKIT_API RK_U64 RK_MPI_MMZ_Handle2PhysAddr(MB_BLK blk) {
    return RK_MPI_MB_Handle2PhysAddr(blk);
}

extern "C" ROCKIT_API RK_VOID *RK_MPI_MMZ_Handle2VirAddr(MB_BLK blk) {
    return RK_MPI_MB_Handle2VirAddr(blk);
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MMZ_Handle2Fd(MB_BLK blk) {
    return RK_MPI_MB_Handle2Fd(blk);
}

extern "C" ROCKIT_API RK_U64 RK_MPI_MMZ_GetSize(MB_BLK blk) {
    return RK_MPI_MB_GetSize(blk);
}

extern "C" ROCKIT_API MB_BLK RK_MPI_MMZ_Fd2Handle(RK_S32 fd) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.by_fd.find(fd);
    return it == s.by_fd.end() ? MB_INVALID_HANDLE : static_cast<MB_BLK>(it->second);
}

extern "C" ROCKIT_API MB_BLK RK_MPI_MMZ_VirAddr2Handle(RK_VOID *address) {
    return RK_MPI_MB_VirAddr2Handle(address);
}

extern "C" ROCKIT_API MB_BLK RK_MPI_MMZ_PhyAddr2Handle(RK_U64 address) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto exact = s.by_phys.find(address);
    if (exact != s.by_phys.end())
        return static_cast<MB_BLK>(exact->second);
    for (const auto &entry : s.by_phys) {
        if (address >= entry.first && address < entry.first + entry.second->size)
            return static_cast<MB_BLK>(entry.second);
    }
    return MB_INVALID_HANDLE;
}

extern "C" ROCKIT_API RK_S32 RK_MPI_MMZ_IsCacheable(MB_BLK blk) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Block *block = as_block_locked(s, blk);
    return block ? (block->cacheable ? 1 : 0) : RK_FAILURE;
}

#define ROCKIT_DEFINE_MMZ_BLOCK_FLUSH(name_, completed_)                                           \
    extern "C" ROCKIT_API RK_S32 name_(MB_BLK blk, RK_U32 offset, RK_U32 length, RK_U32 flags) {   \
        auto &s = state();                                                                         \
        std::lock_guard<std::mutex> lock(s.mutex);                                                 \
        Block *block = as_block_locked(s, blk);                                                    \
        return block ? sync_block_locked(block, offset, length, flags, completed_) : RK_FAILURE;   \
    }

#define ROCKIT_DEFINE_MMZ_VADDR_FLUSH(name_, completed_)                                           \
    extern "C" ROCKIT_API RK_S32 name_(RK_VOID *address, RK_U32 length, RK_U32 flags) {            \
        auto &s = state();                                                                         \
        std::lock_guard<std::mutex> lock(s.mutex);                                                 \
        const auto target = reinterpret_cast<std::uintptr_t>(address);                             \
        for (const auto &entry : s.by_data) {                                                      \
            const auto begin = reinterpret_cast<std::uintptr_t>(entry.first);                      \
            Block *block = entry.second;                                                           \
            if (target >= begin && target < begin + block->size) {                                 \
                const auto offset = static_cast<RK_U32>(target - begin);                           \
                return sync_block_locked(block, offset, length, flags, completed_);                \
            }                                                                                      \
        }                                                                                          \
        return RK_FAILURE;                                                                         \
    }

#define ROCKIT_DEFINE_MMZ_PADDR_FLUSH(name_, completed_)                                           \
    extern "C" ROCKIT_API RK_S32 name_(RK_U64 address, RK_U32 length, RK_U32 flags) {              \
        auto &s = state();                                                                         \
        std::lock_guard<std::mutex> lock(s.mutex);                                                 \
        for (const auto &entry : s.by_phys) {                                                      \
            Block *block = entry.second;                                                           \
            if (address >= entry.first && address < entry.first + block->size) {                   \
                const auto offset = static_cast<RK_U32>(address - entry.first);                    \
                return sync_block_locked(block, offset, length, flags, completed_);                \
            }                                                                                      \
        }                                                                                          \
        return RK_FAILURE;                                                                         \
    }

ROCKIT_DEFINE_MMZ_BLOCK_FLUSH(RK_MPI_MMZ_FlushCacheStart, false)
ROCKIT_DEFINE_MMZ_BLOCK_FLUSH(RK_MPI_MMZ_FlushCacheEnd, true)
ROCKIT_DEFINE_MMZ_VADDR_FLUSH(RK_MPI_MMZ_FlushCacheVaddrStart, false)
ROCKIT_DEFINE_MMZ_VADDR_FLUSH(RK_MPI_MMZ_FlushCacheVaddrEnd, true)
ROCKIT_DEFINE_MMZ_PADDR_FLUSH(RK_MPI_MMZ_FlushCachePaddrStart, false)
ROCKIT_DEFINE_MMZ_PADDR_FLUSH(RK_MPI_MMZ_FlushCachePaddrEnd, true)

#undef ROCKIT_DEFINE_MMZ_PADDR_FLUSH
#undef ROCKIT_DEFINE_MMZ_VADDR_FLUSH
#undef ROCKIT_DEFINE_MMZ_BLOCK_FLUSH
