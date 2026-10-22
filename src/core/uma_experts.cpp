// src/core/uma_experts.cpp - see include/strata/core/uma_experts.hpp.
#include "strata/core/uma_experts.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

bool UmaExperts::supported() {
#if defined(__linux__)
    int dev = 0, integrated = 0, read_only = 0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) == cudaSuccess && integrated != 0 &&
           cudaDeviceGetAttribute(&read_only, cudaDevAttrHostRegisterReadOnlySupported, dev) == cudaSuccess &&
           read_only != 0;
#else
    return false;
#endif
}

UmaExperts::~UmaExperts() { close(); }

const UmaExperts::Range* UmaExperts::range_of(uint64_t off) const {
    for (const Range& r : ranges_)
        if (off >= r.off && off < r.off + r.len) return &r;
    return nullptr;
}

const uint8_t* UmaExperts::device_blob(int64_t layer, int64_t expert) const {
    if (map_ == nullptr || layer < 0 || layer >= lay_.n_layers || expert < 0 || expert >= lay_.n_expert) return nullptr;
    const uint64_t off = lay_.blob_offset(layer, expert);
    const Range* r = range_of(off);
    return r != nullptr ? r->dev + (off - r->off) : nullptr;
}

const uint8_t* UmaExperts::host_blob(int64_t layer, int64_t expert) const {
    if (map_ == nullptr || layer < 0 || layer >= lay_.n_layers || expert < 0 || expert >= lay_.n_expert) return nullptr;
    return map_ + lay_.blob_offset(layer, expert);
}

bool UmaExperts::matches_gguf(const std::string& native, int64_t layer, int64_t expert, std::string& err) const {
#if !defined(__linux__)
    (void) native; (void) layer; (void) expert;
    err = "reading the experts in place needs Linux";
    return false;
#else
    const uint8_t* blob = host_blob(layer, expert);
    if (blob == nullptr || !lay_.native || lay_.gguf_off.size() != (size_t) (3 * lay_.n_layers)) {
        err = "the pack's native_experts.txt has no GGUF offsets to check experts.bin against";
        return false;
    }
    const auto& fm = lay_.fmt[(size_t) layer];
    const uint64_t b = lay_.bytes[(size_t) layer];
    const uint64_t per[3] = {fm.up_off, fm.up_off, b - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    static const char* roles[3] = {"gate", "up", "down"};
    std::vector<uint8_t> buf;
    for (int r = 0; r < 3; ++r) {
        const std::string path = expert_gguf_file(native, lay_, layer, r);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            err = path + ": " + std::strerror(errno);
            return false;
        }
        buf.resize((size_t) per[r]);
        const uint64_t off = lay_.gguf_off[(size_t) (3 * layer + r)] + (uint64_t) expert * per[r];
        uint64_t got = 0;
        while (got < per[r]) {
            const ssize_t n = pread(fd, buf.data() + got, (size_t) (per[r] - got), (off_t) (off + got));
            if (n <= 0) break;
            got += (uint64_t) n;
        }
        ::close(fd);
        if (got != per[r]) {
            err = path + ": short read of the " + roles[r] + " slice";
            return false;
        }
        if (std::memcmp(buf.data(), blob + at[r], (size_t) per[r]) != 0) {
            err = std::string("its ") + roles[r] + " rows differ from " + path +
                  ": experts.bin was cut from another model - write it again with tools/iq_pack.py --experts-bin";
            return false;
        }
    }
    return true;
#endif
}

bool UmaExperts::open(const std::string& path, const strata::kernels::cpu::ExpertLayout& lay, std::string& err,
                      uint64_t chunk_bytes) {
#if !defined(__linux__)
    (void) path; (void) lay; (void) chunk_bytes;
    err = "reading the experts in place needs Linux";
    return false;
#else
    close();
    const auto t0 = std::chrono::steady_clock::now();
    lay_ = lay;
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        err = path + ": " + std::strerror(errno);
        return false;
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0 || (uint64_t) st.st_size < lay.total || lay.total == 0) {
        err = path + ": " + std::to_string((long long) st.st_size) + " bytes, the pack's layout needs " +
              std::to_string((unsigned long long) lay.total);
        close();
        return false;
    }
    size_ = lay.total;
    void* m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (m == MAP_FAILED) {
        err = path + ": mmap: " + std::strerror(errno);
        map_ = nullptr;
        close();
        return false;
    }
    map_ = static_cast<uint8_t*>(m);
    // The registration faults every page in; asking for all of it first lets the kernel read the file in large
    // sequential requests instead of one fault at a time.
    (void) madvise(map_, size_, MADV_WILLNEED);

    const unsigned flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    auto add = [&](uint64_t off, uint64_t len) -> bool {
        if (cudaHostRegister(map_ + off, len, flags) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        void* dev = nullptr;
        if (cudaHostGetDevicePointer(&dev, map_ + off, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            (void) cudaHostUnregister(map_ + off);
            return false;
        }
        ranges_.push_back({off, len, static_cast<uint8_t*>(dev)});
        uma_pinned_add((int64_t) len);
        return true;
    };
    if (!add(0, size_)) {
        // Whole layers per range, cut only where a layer starts on a page boundary: a blob never spans two ranges,
        // whose GPU addresses need not be adjacent.
        const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
        uint64_t start = 0;
        for (int64_t l = 1; l <= lay.n_layers; ++l) {
            const uint64_t end = l == lay.n_layers ? size_ : lay.layer_offset(l);
            const uint64_t next = l + 1 >= lay.n_layers ? size_ : lay.layer_offset(l + 1);
            if (l < lay.n_layers && !(end % page == 0 && next - start > chunk_bytes)) continue;
            if (!add(start, end - start)) {
                err = path + ": cudaHostRegister refused " + std::to_string((unsigned long long) (end - start)) +
                      " bytes at offset " + std::to_string((unsigned long long) start);
                close();
                return false;
            }
            start = end;
        }
    }
    open_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
#endif
}

void UmaExperts::close() {
#if defined(__linux__)
    for (const Range& r : ranges_) {
        (void) cudaHostUnregister(map_ + r.off);
        uma_pinned_add(-(int64_t) r.len);
    }
    ranges_.clear();
    if (map_ != nullptr) munmap(map_, size_);
    if (fd_ >= 0) ::close(fd_);
#endif
    map_ = nullptr;
    fd_ = -1;
    size_ = 0;
    open_s_ = 0;
}

}  // namespace strata::core
