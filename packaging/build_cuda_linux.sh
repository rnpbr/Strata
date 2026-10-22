#!/bin/sh
# Build the Linux CUDA engine package for the release feed.
#
#   VARIANT=cuda12 packaging/build_cuda_linux.sh     # CUDA 12.9: GTX 10 (Pascal), Tesla P100, V100, and the newer cards up to sm_89
#   VARIANT=cuda13 packaging/build_cuda_linux.sh     # CUDA 13: RTX 20 and newer (sm_75 .. sm_120)
#
# On a 64-bit Arm box (linux-arm64) the same two variants are for the NVIDIA boards whose GPU shares the RAM:
#   cuda13: Jetson AGX Orin / Orin NX on JetPack 7 (sm_87), Jetson Thor (sm_110), DGX Spark (sm_121)
# Their CPU code is the portable armv8.2-a + dot product + fp16 build (every Orin, Thor and Spark core has them).
#
# Run it on a Linux box of the package's CPU architecture with the CUDA toolkit of that variant on PATH (nvcc) and a Python 3
# venv with requirements.txt and requirements-build.txt. The pinned llama.cpp comes from $GGML_DIR, else third_party/llama.cpp
# (python -c "import setup; setup.get_llama_cpp()"), else CMake fetches the same pinned commit itself.
# The package lands in $OUT (default ./out-engine-<variant>). Nothing is uploaded.
#
# The cuda12 package is the one the feed and docs/DOCKER.md offer to Pascal and Volta cards, so it needs machine code for sm_60,
# sm_61 and sm_70 and the -DSTRATA_EXPERIMENTAL_SM60=ON build (#3503: the 0.2.1 package had sm_75 / 86 / 89 only, and a P100
# container restarted in a loop with "this PC's GPU is sm_60"). packaging/package_engine.py refuses a cuda12 package without them.
set -eu
variant=${VARIANT:-cuda13}
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
case "$(uname -m)" in
  aarch64|arm64) platform=linux-arm64 ;;
  x86_64|amd64) platform=linux-x86_64 ;;
  *) echo "build_cuda_linux.sh: no CUDA engine package for a $(uname -m) CPU" >&2; exit 2 ;;
esac
case "$platform:$variant" in
  linux-x86_64:cuda12) archs=${ARCHS:-60;61;70;75;80;86;89}; mindriver=575.51; sm60=ON ;;
  linux-x86_64:cuda13) archs=${ARCHS:-75;86;89;120}; mindriver=580.65; sm60=OFF ;;
  linux-arm64:cuda13) archs=${ARCHS:-87;110;121}; mindriver=580.65; sm60=OFF ;;
  linux-arm64:*) echo "build_cuda_linux.sh: the Arm package is cuda13 (JetPack 7, DGX OS); JetPack 6's CUDA 12 is not packaged" >&2; exit 2 ;;
  *) echo "build_cuda_linux.sh: VARIANT must be cuda12 or cuda13" >&2; exit 2 ;;
esac
ggml=${GGML_DIR:-}
[ -z "$ggml" ] && [ -d "$repo/third_party/llama.cpp" ] && ggml="$repo/third_party/llama.cpp"
# nvcc jobs take 1-2 GiB each; on a board whose GPU shares the RAM, keep 2 GiB per job and 2 GiB aside
jobs=${JOBS:-$(awk -v n="$(nproc)" '/^MemAvailable:/ { j = int(($2 / 1048576 - 2) / 2); if (j < 1) j = 1; print (j < n ? j : n) }' /proc/meminfo)}
out=${OUT:-$repo/out-engine-$variant}
build=${BUILD:-$repo/build-$variant}
py=${PYTHON:-python3}
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$variant:$cudafull" in
  cuda12:12.*|cuda13:13.*) ;;
  *) echo "build_cuda_linux.sh: $variant needs the matching CUDA toolkit; nvcc here is ${cudafull:-missing}" >&2; exit 2 ;;
esac

real=""; for a in $(echo "$archs" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}
last=$(echo "$archs" | tr ';,' '  ' | awk '{print $NF}')
cd "$repo"
cmake -G Ninja -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=$sm60 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR="$ggml"
cmake --build "$build" --target strata strata_kq -j "$jobs"

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
stage=$(mktemp -d "${TMPDIR:-/tmp}/strata-engine-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp "$build/strata" "$stage/"
[ -f "$build/libstrata_kq.so" ] && cp "$build/libstrata_kq.so" "$stage/"
echo "{\"version\": \"$ver\", \"source\": \"packaging/build_cuda_linux.sh\", \"archs\": \"$archs\", \"cuda\": \"$cudafull\", \"portable\": true, \"platform\": \"$platform\"}" > "$stage/BUILD.json"
"$py" packaging/collect_libs.py --binary "$build/strata" --search /usr/local/cuda/lib64 --out "$stage" --also 'libcublasLt.so*'
arch_args=""; for a in $(echo "$archs" | tr ';,' '  '); do arch_args="$arch_args --arch sm_$a"; done
mkdir -p "$out"
# shellcheck disable=SC2086
"$py" packaging/package_engine.py --src "$stage" --backend cuda --version "$ver" --out "$out" $arch_args --arch "compute_$last" \
    --min-driver nvidia=$mindriver --variant "$variant" --platform "$platform" --build-id "release-${ver}-${variant}-${cudafull}" \
    --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
for d in "$out"/*/; do "$py" packaging/package_engine.py --verify "$d"; "$py" packaging/binary_deps.py "$d"; done
echo "built: $out"
