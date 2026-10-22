// include/strata/core/uma_experts.hpp - the expert arena read in place by a GPU that shares the RAM (Jetson, DGX Spark).
//
// A discrete card copies the experts it keeps into VRAM and the CPU computes the rest from RAM. A GPU without memory of
// its own reads the RAM itself, as fast as it reads anything (an AGX Orin measured 172 GB/s from a registered file
// mapping against 180 GB/s from cudaMalloc), so that copy only doubles the memory the experts take and the time a start
// spends filling it. Here the pack's experts.bin is mapped read-only and registered for the GPU, and every expert is
// read where the file's pages lie: nothing is copied, nothing is computed on the CPU.
//
// The registration pins the pages, so they stay in RAM for as long as the arena is open, and it is counted against the
// memory device_free_bytes() reports (the OS still lists those pages as reclaimable cache).
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class UmaExperts {
public:
    UmaExperts() = default;
    ~UmaExperts();
    UmaExperts(const UmaExperts&) = delete;
    UmaExperts& operator=(const UmaExperts&) = delete;

    /// True when the current CUDA device has no memory of its own (cudaDevAttrIntegrated) and can register a
    /// read-only host mapping (cudaDevAttrHostRegisterReadOnlySupported).
    static bool supported();

    /// Maps `path` (the pack's experts.bin, laid out as `lay` says) and registers it for the GPU. One registration when
    /// the driver takes the whole file, else ranges of whole layers up to `chunk_bytes` each (a single call has been
    /// refused near 30 GiB on an Orin). False with `err`, nothing left mapped or registered, when any step fails.
    bool open(const std::string& path, const strata::kernels::cpu::ExpertLayout& lay, std::string& err,
              uint64_t chunk_bytes = 4ull << 30);
    void close();

    bool valid() const { return map_ != nullptr; }
    /// The GPU's address of (layer, expert)'s blob; null when out of range or closed.
    const uint8_t* device_blob(int64_t layer, int64_t expert) const;
    /// The blob's address in this process's mapping (what the CPU side and verify_slot read).
    const uint8_t* host_blob(int64_t layer, int64_t expert) const;
    /// True when (layer, expert)'s blob holds the bytes of its gate, up and down slices in the model's GGUF (`native`,
    /// the --native shard): an experts.bin cut from another model or quantization is refused with `err`.
    bool matches_gguf(const std::string& native, int64_t layer, int64_t expert, std::string& err) const;
    /// Bytes registered (the whole file).
    uint64_t bytes() const { return size_; }
    /// Registrations used: 1 for the whole file, more when it was registered in ranges.
    int ranges() const { return (int) ranges_.size(); }
    /// Seconds the mapping and registration took (the page-in from disk included).
    double open_seconds() const { return open_s_; }

private:
    struct Range {
        uint64_t off = 0, len = 0;   ///< in the file
        uint8_t* dev = nullptr;      ///< the GPU's address of `off`
    };
    const Range* range_of(uint64_t off) const;

    int fd_ = -1;
    uint8_t* map_ = nullptr;
    uint64_t size_ = 0;
    std::vector<Range> ranges_;
    strata::kernels::cpu::ExpertLayout lay_;
    double open_s_ = 0;
};

}  // namespace strata::core
