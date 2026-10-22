// --uma-experts: a small native-layout experts.bin mapped and registered in place (UmaExperts), read by the GPU through
// an in-place ExpertCache, checked against the "GGUF" slices it was cut from, and refused once one byte differs.
// Needs a GPU that shares the RAM (Jetson, DGX Spark); exits 77 anywhere else.
#include "strata/core/expert_cache.hpp"
#include "strata/core/uma_experts.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

void write_file(const std::filesystem::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0 || !strata::core::UmaExperts::supported()) {
        std::puts("no GPU that shares the RAM: skipped");
        return 77;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("strata_uma_test_" + std::to_string((long long) getpid()));
    fs::create_directories(dir);

    // Two layers of four experts with different blob sizes, as a native pack has; each layer starts on a page and the
    // file is whole pages (20480 bytes), like a real experts.bin.
    strata::kernels::cpu::ExpertLayout lay;
    lay.native = true;
    lay.n_layers = 2;
    lay.n_expert = 4;
    lay.fmt.resize(2);
    lay.fmt[0].up_off = 640;  lay.fmt[0].down_off = 1280;   // blob 2048: gate 640 | up 640 | down 768
    lay.fmt[1].up_off = 1024; lay.fmt[1].down_off = 2048;   // blob 3072: gate 1024 | up 1024 | down 1024
    lay.bytes = {2048, 3072};
    lay.offset = {0, 8192};
    lay.max_blob = 3072;
    lay.total = 8192 + 4 * 3072;

    // The "GGUF": each role's tensor holds its four experts back to back, the layout iq_pack cuts experts.bin from.
    std::vector<uint8_t> gguf(4096), bin(lay.total);
    lay.gguf_off.resize(6);
    for (int64_t l = 0; l < 2; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
        const uint64_t at[3] = {0, fm.up_off, fm.down_off};
        for (int r = 0; r < 3; ++r) {
            lay.gguf_off[(size_t) (3 * l + r)] = gguf.size();
            for (int64_t e = 0; e < 4; ++e)
                for (uint64_t i = 0; i < per[r]; ++i) {
                    const uint8_t v = (uint8_t) (l * 97 + r * 31 + e * 13 + i * 7);
                    gguf.push_back(v);
                    bin[lay.blob_offset(l, e) + at[r] + i] = v;
                }
        }
    }
    const fs::path gguf_path = dir / "model.gguf", bin_path = dir / "experts.bin";
    write_file(gguf_path, gguf);
    write_file(bin_path, bin);

    std::string err;
    const uint64_t pinned_before = strata::core::uma_pinned_bytes();
    {
        strata::core::UmaExperts uma;
        if (!uma.open(bin_path.string(), lay, err)) {
            std::fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        check(uma.ranges() == 1 && uma.bytes() == lay.total, "the whole file is one registration");
        check(strata::core::uma_pinned_bytes() == pinned_before + lay.total, "the pinned bytes are counted");
        check(uma.device_blob(2, 0) == nullptr && uma.device_blob(0, 4) == nullptr, "out of range has no address");

        std::vector<const uint8_t*> blobs;
        for (int64_t l = 0; l < 2; ++l)
            for (int64_t e = 0; e < 4; ++e) {
                blobs.push_back(uma.device_blob(l, e));
                check(uma.host_blob(l, e) != nullptr && uma.host_blob(l, e)[0] == bin[lay.blob_offset(l, e)],
                      "the host mapping holds the file");
                check(uma.matches_gguf(gguf_path.string(), l, e, err), "every expert matches its GGUF slices");
            }

        strata::core::ExpertCache cache;
        check(cache.open_in_place(blobs, {2048, 3072}, 2, 4, err), "open_in_place");
        check(cache.in_place() && cache.slots() == 8, "one slot per expert");
        for (int64_t l = 0; l < 2; ++l)
            for (int64_t e = 0; e < 4; ++e) {
                const int32_t s = cache.slot_of(l, e);
                check(s == (int32_t) (l * 4 + e), "residency is the identity");
                check(cache.device_slot(s) == blobs[(size_t) s], "a slot is the expert's own address");
                check(cache.verify_slot(s, bin.data() + lay.blob_offset(l, e), err, (int64_t) lay.bytes[(size_t) l]),
                      "the GPU reads the file's bytes");
            }

        // slot 0 must be the lowest address: the kernels add slot offsets to device_slot(0)
        std::vector<const uint8_t*> swapped = blobs;
        std::swap(swapped[0], swapped[1]);
        strata::core::ExpertCache bad;
        check(!bad.open_in_place(swapped, {2048, 3072}, 2, 4, err), "slot 0 above another slot is refused");
        check(!bad.open_in_place(blobs, {2048}, 2, 4, err), "a missing layer size is refused");
        cache.close();
    }
    check(strata::core::uma_pinned_bytes() == pinned_before, "close hands the pinned bytes back");

    // one byte of layer 1, expert 2's down rows changed: that expert no longer matches, the others still do
    bin[lay.blob_offset(1, 2) + lay.fmt[1].down_off + 5] ^= 0xFF;
    write_file(bin_path, bin);
    {
        strata::core::UmaExperts uma;
        if (!uma.open(bin_path.string(), lay, err)) {
            std::fprintf(stderr, "open (changed): %s\n", err.c_str());
            return 1;
        }
        check(!uma.matches_gguf(gguf_path.string(), 1, 2, err) && err.find("down") != std::string::npos,
              "a changed byte is found, in the role it is in");
        check(uma.matches_gguf(gguf_path.string(), 1, 1, err), "a neighbour is unaffected");
    }

    // a file shorter than the layout is refused before anything is mapped
    bin.resize(lay.total - 4096);
    write_file(bin_path, bin);
    {
        strata::core::UmaExperts uma;
        check(!uma.open(bin_path.string(), lay, err) && !uma.valid(), "a short experts.bin is refused");
    }

    fs::remove_all(dir);
    if (g_fail == 0) std::puts("uma_experts_test: OK");
    return g_fail == 0 ? 0 : 1;
}
