// src/kernels/cpu/x86_absent.cpp - the x86 CPU kernels' entry points on a CPU without them (arm64).
//
// native_expert.cpp and expert_source.cpp call the AVX / AVX2 / AVX-512 rows only behind the cpu_*_ok() probes, which
// answer no on Arm, so those rows run on ggml-cpu's NEON kernels instead. The predicates here answer no as well. A
// kernel reached anyway means a call site lost its probe: it stops the engine with the kernel's name rather than
// computing garbage.
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#error "x86_absent.cpp is for CPUs without the x86 kernels; an x86 build compiles the kernels themselves"
#endif

namespace strata::kernels::cpu {
namespace {
[[noreturn]] void absent(const char* kernel) {
    std::fprintf(stderr, "strata: internal error: the x86 kernel %s was called on a CPU that does not have it\n", kernel);
    std::abort();
}
}  // namespace

bool iq128_supported(int) noexcept { return false; }
bool iq128_down_supported(int) noexcept { return false; }
int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 0; }
bool nvfp4_256_fits(int) { return false; }
bool nvfp4_512_fits(int) { return false; }

void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq128_gu_rows");
}
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq128_down_rows");
}
void q8k_quant_avx1(const float*, void*, int64_t) { absent("q8k_quant_avx1"); }

void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    absent("iq256_gu_rows_exact_one");
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_gu_rows");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_rows");
}
void iq256_gu_rows_v(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_gu_rows_v");
}
void iq256_rows_v(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq256_rows_v");
}
void q8k_quant_avx2(const float*, void*, int64_t) { absent("q8k_quant_avx2"); }
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq4nl256_down_rows");
}
void iq4nl256_down_rows_v(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq4nl256_down_rows_v");
}

void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq512_gu_rows");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("iq512_rows");
}

void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    absent("bf16_rows_dot_multi_avx1");
}
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    absent("kq256_rows");
}
void bf16_rows_dot(const uint16_t*, int, int, const float*, float*) { absent("bf16_rows_dot"); }
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) { absent("bf16_rows_dot_multi"); }

void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float,
                       float) {
    absent("nvfp4_256_gu_rows");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    absent("nvfp4_256_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float,
                       float) {
    absent("nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    absent("nvfp4_512_rows");
}

}  // namespace strata::kernels::cpu
