# 二、计算层优化

在内存带宽充足的前提下，计算效率决定最终性能。计算层优化目标是让每个 CUDA Core 在每个时钟周期都在执行有效指令。

---

## 2.1 Warp Divergence（线程束分化）

### 原理

GPU 采用 **SIMT（单指令多线程）** 执行模型：同一 Warp 的 32 个线程在同一时钟周期执行**相同指令**。

当分支条件不同时，硬件通过 **Active Mask** 处理：
```
if (tid % 2 == 0) { A(); } else { B(); }
```
```
时钟周期 1：偶数线程执行 A()，奇数线程挂起（masked off）
时钟周期 2：奇数线程执行 B()，偶数线程挂起
总开销 = 2 个串行周期，吞吐量下降 50%
```

**最坏情况**：Warp 内 32 个线程走 32 条不同路径 → 退化为完全串行执行。

### 常见场景及优化

**① 基于线程 ID 的分支**
```cuda
// ❌ 奇偶分化
if (tid % 2 == 0) process_even();
else              process_odd();

// ✅ 数据重排：偶数索引数据放前半，奇数放后半
// 或用掩码计算代替分支
float result = process_even() * (tid % 2 == 0) +
               process_odd()  * (tid % 2 != 0);
```

**② 提前退出（early exit）**
```cuda
// ❌ 部分线程提前 return，同 Warp 的其他线程必须等待
if (tid >= N) return;

// ✅ 确保整个 Warp 的线程数量是 32 的整数倍，或用 predication
float val = (tid < N) ? data[tid] : 0.0f;
```

**③ 归约中的分化**
```cuda
// ❌ 前半段线程工作，后半段闲置：连续活跃线程会在同一 Warp 内
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0)        // 后期多数线程失活，Warp 内严重分化
        smem[tid] += smem[tid+s];
}

// ✅ 折半归约：活跃线程从 tid=0 连续排列，Warp 内无分化
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        smem[tid] += smem[tid + s];
    __syncthreads();
}
```

### 检测工具
- `ncu` → `smsp__thread_inst_executed_pred_on_efficiency` 指标（越接近 100% 越好）
- `nvprof` → `branch_efficiency`

---

## 2.2 Warp Shuffle（Warp 内通信）

Shuffle 指令允许 Warp 内线程**直接读取其他线程的寄存器**，无需共享内存，延迟极低（几个时钟周期）。

### 常用 Shuffle 指令（Compute Capability 3.0+）

| 函数 | 功能 |
|------|------|
| `__shfl_sync(mask, val, srcLane)` | 从指定 lane 广播 val |
| `__shfl_up_sync(mask, val, delta)` | 从低 delta 的 lane 读取 |
| `__shfl_down_sync(mask, val, delta)` | 从高 delta 的 lane 读取 |
| `__shfl_xor_sync(mask, val, laneMask)` | 蝶形交换（butterfly） |

### Warp-Level Reduce（无需共享内存）

```cuda
__device__ float warpReduceSum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val; // 结果在 lane 0
}
```

### Block-Level Reduce（结合 Shuffle + Shared Memory）

```cuda
__device__ float blockReduceSum(float val) {
    __shared__ float warpResults[32]; // 最多 32 个 warp
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceSum(val);          // 每个 warp 内归约

    if (lane == 0) warpResults[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? warpResults[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val); // 最终归约
    return val;
}
```

---

## 2.3 指令级优化

### 循环展开（Loop Unrolling）

```cuda
// 编译器自动展开
#pragma unroll 4
for (int i = 0; i < N; ++i) {
    result += a[i] * b[i];
}

// 完全展开（N 必须是编译期常量）
#pragma unroll
for (int i = 0; i < 8; ++i) { ... }

// 模板参数展开（将 BLOCK_SIZE 作为模板参数传入）
template <int BLOCK_SIZE>
__global__ void kernel() {
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE / 32; ++i) { ... }
}
```

**权衡**：展开减少循环控制开销，增加指令级并行（ILP）；但过度展开会增加寄存器压力，导致溢出。

### 快速数学函数

CUDA 提供内置的 intrinsic 函数，牺牲极小精度换取更高吞吐：

| 精确版本 | 快速版本 | 加速倍数 |
|----------|----------|----------|
| `sinf(x)` | `__sinf(x)` | ~2-4x |
| `cosf(x)` | `__cosf(x)` | ~2-4x |
| `expf(x)` | `__expf(x)` | ~2-4x |
| `logf(x)` | `__logf(x)` | ~2-4x |
| `a / b` | `a * __frcp_rn(b)` | ~2x |
| `sqrtf(x)` | `__fsqrt_rn(x)` | 略快 |

编译器级别的快速数学：
```bash
nvcc -use_fast_math  # 等价于 --ftz=true --prec-div=false --prec-sqrt=false
```

### 整数除法与取模优化

整数除法和取模非常昂贵（~20-40 cycle）。若除数是 2 的幂次：
```cuda
// ❌ 慢
int q = n / 32;
int r = n % 32;

// ✅ 位运算替代（仅适用于 2 的幂次）
int q = n >> 5;
int r = n & 31;
```

若除数固定但非 2 的幂，编译器通常自动优化（乘以倒数）。动态变化的非 2 次幂除数开销最大，应尽量避免。

### FMA（Fused Multiply-Add）

CUDA 硬件支持 FMA 指令：`a * b + c` 在一条指令内完成，不损失精度：
```cuda
// 编译器通常自动生成 FMA
float result = __fmaf_rn(a, b, c); // 显式触发 FMA
```

---

## 2.4 数值精度与混合精度

### 数据类型选择

| 类型 | 位宽 | 单 SM 吞吐（A100） | 用途 |
|------|------|-------------------|------|
| `double` | 64 | 低 | 高精度科学计算 |
| `float` | 32 | 基准 | 通用计算 |
| `__half` (fp16) | 16 | ~2x | 深度学习推理/训练 |
| `__nv_bfloat16` | 16 | ~2x | 训练（数值范围更宽） |
| `int8` | 8 | ~4x | 推理量化 |

### Tensor Core 利用

Tensor Core 专为矩阵乘法设计，A100 上 fp16 Tensor Core 吞吐是 fp32 CUDA Core 的 **~16x**。

```cuda
// 使用 WMMA API 调用 Tensor Core
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float>         c_frag;

load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag); // Tensor Core 执行
store_matrix_sync(c_ptr, c_frag, 16, mem_row_major);
```

实际工程中更推荐使用 **cuBLAS** 或 **CUTLASS** 直接调用高度优化的 GEMM。

---

## 2.5 向量化访存（Vectorized Memory Access）

使用向量类型（`float4`, `int4`, `float2`）一次加载多个元素，减少内存事务次数，提升 L/S 利用率：

```cuda
// ❌ 标量访问：4 条 LDS 指令
float a = data[4*tid+0];
float b = data[4*tid+1];
float c = data[4*tid+2];
float d = data[4*tid+3];

// ✅ 向量化：1 条 LDS.128 指令，带宽利用率更高
float4 vec = reinterpret_cast<float4*>(data)[tid];
float a = vec.x, b = vec.y, c = vec.z, d = vec.w;
```

**注意**：要求起始地址 16 字节对齐（`float4` 为 16 字节）。

---

## 2.6 Predication vs Branching

对于简单的条件赋值，编译器会自动使用 **predicated execution**（条件谓词执行）替代分支，消除分化：

```cuda
// 编译器通常会将此优化为 predicated 指令，无分化
float val = (condition) ? a : b;

// 等价的 PTX 伪代码（无跳转指令）：
// setp.eq p, condition, 1
// selp val, a, b, p
```

对于**复杂分支**（多条指令），predication 开销过高，此时数据重排（将相同分支的线程归组）更有效。
