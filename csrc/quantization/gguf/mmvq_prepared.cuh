#pragma once

#if defined(USE_ROCM)
namespace gguf_k45_prepared {

using int4_t = int __attribute__((ext_vector_type(4)));
using int2_t = int __attribute__((ext_vector_type(2)));

__device__ int4_t load4(int4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ int load1(int4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ __forceinline__ int4_t resource(const void* ptr) {
  struct Resource {
    const void* ptr;
    uint32_t range;
    uint32_t config;
  } value{ptr, 0xffffffffu, 0x00020000u};
  return __builtin_bit_cast(int4_t, value);
}

// Private activation records: 64 int8 values, then two {half d, int16 sum}
// pairs. The integer sums belong to the quantized values, unlike Q8_1 ds.y.
template <typename scalar_t, bool Q5>
__global__ __launch_bounds__(512) void mul_mat_vec_k45_q8_1_prepared(
    const uint8_t* __restrict__ weights, const int* __restrict__ x,
    scalar_t* __restrict__ out, int cols, int rows) {
  constexpr int rows_per_block = 8;
  const int row = blockIdx.x * rows_per_block + threadIdx.x / 64;
  const int lane = threadIdx.x % 64;
  const int group = lane % 4;
  const int nb = cols / 256;
  const int padded = (cols + 511) / 512 * 512;
  extern __shared__ uint8_t shared_x[];
  for (int i = threadIdx.x; i < padded / 32 * 9; i += blockDim.x) {
    reinterpret_cast<int*>(shared_x)[i] = x[i];
  }
  __syncthreads();
  if (row >= rows) return;

  const int4_t wres = resource(weights);
  const int blocks = rows * nb;
  const int scbase = blocks * (Q5 ? 160 : 128);
  float total = 0.f;
  for (int sb = lane / 4; sb < nb; sb += 16) {
    const int base = row * nb + sb;
    const int4_t ql0 = load4(wres, base * 128 + group * 32, 0, 0);
    const int4_t ql1 = load4(wres, base * 128 + group * 32 + 16, 0, 0);
    int4_t high0, high1;
    if constexpr (Q5) {
      high0 = load4(wres, blocks * 128 + base * 32, 0, 0);
      high1 = load4(wres, blocks * 128 + base * 32 + 16, 0, 0);
    }
    const int4_t meta = load4(wres, scbase + base * 12, 0, 0);
    int sc0, sc1, min0, min1;
    gfx906_kq_scales(meta[0], meta[1], meta[2], group * 2,
                      sc0, sc1, min0, min1);
    const uint32_t dm = load1(wres, scbase + blocks * 12 + base * 4, 0, 0);
    const float d = __half2float(__ushort_as_half(dm & 65535));
    const float minimum = __half2float(__ushort_as_half(dm >> 16));

    const uint8_t* record = shared_x + (sb * 4 + group) * 72;
    int4_t a0, a1, b0, b1;
    int2_t scales;
    __builtin_memcpy(&a0, record, 16);
    __builtin_memcpy(&a1, record + 16, 16);
    __builtin_memcpy(&b0, record + 32, 16);
    __builtin_memcpy(&b1, record + 48, 16);
    __builtin_memcpy(&scales, record + 64, 8);
    int dot[2] = {};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int v0 = (uint32_t(ql0[word]) >> (half * 4)) & 0x0f0f0f0f;
        int v1 = (uint32_t(ql1[word]) >> (half * 4)) & 0x0f0f0f0f;
        if constexpr (Q5) {
          v0 |= ((uint32_t(high0[word]) >> (group * 2 + half)) << 4) & 0x10101010;
          v1 |= ((uint32_t(high1[word]) >> (group * 2 + half)) << 4) & 0x10101010;
        }
        const int u0 = half ? b0[word] : a0[word];
        const int u1 = half ? b1[word] : a1[word];
        dot[half] = __builtin_amdgcn_sdot4(
            v0, u0, __builtin_amdgcn_sdot4(v1, u1, dot[half], false), false);
      }
    }
    const float d0 = __half2float(__ushort_as_half(scales[0] & 65535));
    const float d1 = __half2float(__ushort_as_half(uint32_t(scales[1]) & 65535));
    const int sum0 = int16_t(uint32_t(scales[0]) >> 16);
    const int sum1 = int16_t(uint32_t(scales[1]) >> 16);
    const float sd = d0 * (dot[0] * sc0) + d1 * (dot[1] * sc1);
    const float sm = d0 * (sum0 * min0) + d1 * (sum1 * min1);
    total += d * sd - minimum * sm;
  }
#pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1) {
    total += VLLM_SHFL_XOR_SYNC(total, mask);
  }
  if (lane == 0) out[row] = scalar_t(total);
}

template <typename scalar_t>
static void launch(const void* weights, const void* x, scalar_t* out,
                   int cols, int rows, int64_t type, cudaStream_t stream) {
  const int shared_bytes = ((cols + 511) / 512 * 512) / 32 * 36;
  if (type == 13) {
    mul_mat_vec_k45_q8_1_prepared<scalar_t, true>
        <<<(rows + 7) / 8, 512, shared_bytes, stream>>>(
            static_cast<const uint8_t*>(weights), static_cast<const int*>(x),
            out, cols, rows);
  } else {
    mul_mat_vec_k45_q8_1_prepared<scalar_t, false>
        <<<(rows + 7) / 8, 512, shared_bytes, stream>>>(
            static_cast<const uint8_t*>(weights), static_cast<const int*>(x),
            out, cols, rows);
  }
}

}  // namespace gguf_k45_prepared
#endif
