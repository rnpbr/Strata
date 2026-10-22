// include/strata/platform/spin.hpp - the two CPU hints the host's busy-wait and hand-off loops use, on every target.
//
// The host hands work to the GPU through words in mapped host memory: it writes the data, fences, then raises a flag a
// spin kernel polls. On x86 the fence is SFENCE. On Arm the GPU of a Jetson or a DGX Spark reads that memory as an
// I/O-coherent observer outside the CPU's inner-shareable domain, so the barrier must reach the outer-shareable
// domain: DMB OSHST, the same barrier Linux uses for coherent DMA memory (dma_wmb).
//
// The other way, the GPU writes its outputs, fences (__threadfence_system) and raises a sequence word the host spins
// on. x86 never lets a later load pass an earlier one; Arm does, so the host's reads of those outputs could be served
// before its read of the word. DMB OSHLD after the spin (Linux's dma_rmb) keeps them behind it.
#pragma once

#include <atomic>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define STRATA_SPIN_X86 1
#elif defined(__aarch64__)
#define STRATA_SPIN_ARM64 1
#endif

namespace strata::platform {

/// One iteration of a busy-wait: PAUSE on x86, YIELD on Arm. It keeps the spinning core from flooding the memory
/// system and hands the pipeline to a sibling thread.
inline void spin_pause() noexcept {
#if defined(STRATA_SPIN_X86)
    _mm_pause();
#elif defined(STRATA_SPIN_ARM64)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

/// Orders every store before it ahead of every store after it, as the GPU sees them. Call it between writing data
/// the GPU reads from mapped host memory and raising the flag that releases the GPU.
inline void device_store_fence() noexcept {
#if defined(STRATA_SPIN_X86)
    _mm_sfence();
#elif defined(STRATA_SPIN_ARM64)
    __asm__ __volatile__("dmb oshst" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

/// Orders the load that saw a GPU's flag ahead of every load after it. Call it once a spin on a word the GPU raises
/// has ended, before reading what the GPU wrote to mapped host memory ahead of that word.
inline void device_load_fence() noexcept {
#if defined(STRATA_SPIN_ARM64)
    __asm__ __volatile__("dmb oshld" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);   // x86 keeps loads in order: the compiler must too
#endif
}

}  // namespace strata::platform
