# 四、算法层优化

算法层优化从整体计算结构入手，减少全局内存读写总量，提升数据复用率，是比微调参数更有乘数效应的优化手段。

---

## 4.1 Kernel Fusion（内核融合）

### 问题背景

多个独立 kernel 顺序执行时，每个 kernel 都需要：
1. 从全局内存读取输入
2. 执行计算
3. 将结果写回全局内存（供下一个 kernel 读取）

中间结果的**写入→读取往返**造成大量无谓的全局内存带宽消耗。

### 融合原理

将多个操作合并到一个 kernel 中，中间结果留在**寄存器**或**共享内存**中传递：

```
融合前：
[Global Mem] → Kernel A → [Global Mem] → Kernel B → [Global Mem] → Kernel C → [Global Mem]
  读 N 次                   写+读 N 次                 写+读 N 次                  写 N 次
                                                                           = 7N 次全局内存访问

融合后：
[Global Mem] → Kernel ABC → [Global Mem]
  读 N 次     (寄存器中间传递)   写 N 次
                                       = 2N 次全局内存访问
```

### 典型融合场景

**① 逐元素操作链（Element-wise fusion）**
```cuda
// ❌ 三个独立 kernel，6N 次全局内存访问
scale_kernel<<<...>>>(x, tmp1, alpha);  // y = x * alpha
bias_kernel<<<...>>>(tmp1, tmp2, b);    // y = x + b
relu_kernel<<<...>>>(tmp2, out);        // y = max(0, x)

// ✅ 融合后，2N 次全局内存访问
__global__ void scale_bias_relu(float* x, float* out, float alpha, float b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = x[i] * alpha + b;
        out[i] = fmaxf(0.0f, val);
    }
}
```

**② Conv + BN + ReLU（深度学习算子融合）**
- BN（Batch Normalization）的均值和方差在 inference 阶段是固定的，可以与卷积的权重合并
- ReLU 逐元素操作，在卷积输出写回前就可计算完毕

**③ Softmax 融合**
```
标准 Softmax = max reduction → subtract → exp → sum reduction → divide
共 5 次全局内存遍历 → 融合后只需 2 次（一次读，一次写）
```

### 融合的限制

- Block 内共享内存容量限制
- 融合后寄存器压力增大，可能降低占用率
- 不同 parallelism 的 kernel（如 reduce 和 element-wise）融合时需设计复杂的线程映射

---

## 4.2 Parallel Reduction（并行归约）

归约（Reduction）是 GPU 中最典型的通信密集模式，以求和为例从 O(N) 串行优化到 O(log N) 并行。

### 演进过程

**v0：纯全局内存归约（基线）**
```cuda
__global__ void reduce_v0(float* g_in, float* g_out, int N) {
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    // 直接在全局内存上操作，访存极慢
    for (int s = 1; s < blockDim.x; s *= 2)
        if (tid % (2*s) == 0) g_in[i] += g_in[i + s]; // Divergence!
    if (tid == 0) g_out[blockIdx.x] = g_in[i];
}
```

**v1：共享内存**
```cuda
__global__ void reduce_v1(float* g_in, float* g_out, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < N) ? g_in[i] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0)           // 仍有 Warp Divergence
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = smem[0];
}
```

**v2：折半归约（消除 Warp Divergence）**
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)                        // ✅ 活跃线程连续，无 Warp Divergence
        smem[tid] += smem[tid + s];
    __syncthreads();
}
```

**v3：每线程处理多个元素（Grid-Stride）**
```cuda
// 在写入共享内存前，先在全局内存上做串行归约，减少 Block 数量
float sum = 0.0f;
for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x)
    sum += g_in[i];
smem[tid] = sum;
__syncthreads();
// ... 折半归约 ...
```

**v4：最后 32 个元素 Warp Unroll（消除 `__syncthreads()`）**
```cuda
// 当 s <= 32 时，所有活跃线程都在同一个 Warp 内，天然同步
if (tid < 32) {
    volatile float* vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid +  8];
    vsmem[tid] += vsmem[tid +  4];
    vsmem[tid] += vsmem[tid +  2];
    vsmem[tid] += vsmem[tid +  1];
}
```

**v5：Warp Shuffle（最优 Warp 内归约）**
```cuda
// 用 __shfl_down_sync 代替共享内存，延迟更低
if (blockDim.x >= 64) {
    // 折半归约降到 32
}
// Warp 内 Shuffle 归约
val = warpReduceSum(val);
```

**v6：Block Size 作为模板参数编译期展开**
```cuda
template <int BLOCK_SIZE>
__global__ void reduce(float* in, float* out, int N) {
    // 编译器在 BLOCK_SIZE 已知时完全展开所有分支和循环
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid+256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid+128]; __syncthreads(); }
    // ...
}
```

### 性能演进对比（相对 v0）

| 版本 | 关键优化 | 相对性能 |
|------|----------|----------|
| v0 | 基线 | 1x |
| v1 | 共享内存 | ~8x |
| v2 | 折半归约（消除 Divergence） | ~10x |
| v3 | Grid-Stride（每线程多元素） | ~12x |
| v4 | Warp Unroll | ~14x |
| v5 | Shuffle | ~15x |
| v6 | 模板展开 | ~16x |

---

## 4.3 Prefix Sum（前缀和 / Scan）

Scan 是另一类重要的并行原语，与 Reduce 不同，需要产生所有中间结果。

### Blelloch Scan（Work-Efficient Parallel Scan）

两阶段：**Up-Sweep（Reduce Tree）** + **Down-Sweep**：

```
阶段一 Up-Sweep（构建求和树）：
[1, 2, 3, 4, 5, 6, 7, 8]
→ [1, 3, 3, 7, 5, 11, 7, 36]

阶段二 Down-Sweep（分发前缀和）：
→ [0, 1, 3, 6, 10, 15, 21, 28]
```

复杂度：$O(N)$ work, $O(\log N)$ span，Block 内可用共享内存完成，Block 间需要多轮 kernel。

---

## 4.4 Tiling（分块）

Tiling 是利用共享内存减少全局内存访问的核心算法模式，本质是**数据复用**。

### 核心思路

将大问题分割为小 Tile，每个 Block 负责一个 Tile：
1. 协作加载：Block 内所有线程一起将 Tile 数据从全局内存搬入共享内存（合并访问）
2. 共享内存计算：在共享内存上执行计算，避免重复访问全局内存
3. 写出结果：将结果写回全局内存

### Tile 大小选择

- **小 Tile**：共享内存消耗少，占用率高，但全局内存访问减少效果有限
- **大 Tile**：全局内存复用率高，但共享内存消耗大，占用率低
- 实践：每维 16 或 32（与 Warp 大小对齐）

---

## 4.5 Double Buffering（双缓冲流水线）

### 问题

Tiling 中，每次迭代的流程是：
```
1. 加载 Tile 到共享内存  ← 访存延迟
2. __syncthreads()
3. 在共享内存上计算
4. __syncthreads()
```
步骤 1 和步骤 3 完全串行，访存延迟无法被隐藏。

### 解决方案

使用两组缓冲区：一组计算当前 Tile，同时预取下一个 Tile：

```cuda
template <int TILE>
__global__ void gemm_dbuf(float* A, float* B, float* C, int N) {
    __shared__ float bufA[2][TILE][TILE];
    __shared__ float bufB[2][TILE][TILE];

    int cur = 0;

    // 预加载第 0 个 Tile
    load_tile(bufA[0], bufB[0], A, B, 0, ...);
    __syncthreads();

    for (int k = 1; k <= N / TILE; ++k) {
        int next = 1 - cur;

        // 异步预加载下一个 Tile（与计算重叠）
        if (k < N / TILE)
            load_tile(bufA[next], bufB[next], A, B, k, ...);

        // 计算当前 Tile
        compute_tile(bufA[cur], bufB[cur], C, ...);

        __syncthreads();
        cur = next;
    }
}
```

Ampere+ 架构支持 **异步内存拷贝（`cp.async`）**，可以真正在 DMA 层面实现加载与计算重叠：
```cuda
#include <cuda/pipeline>
cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
cuda::memcpy_async(smem, gmem, bytes, pipe);
pipe.producer_commit();
// 执行其他计算...
pipe.consumer_wait();
```

---

## 4.6 Segmented Reduction / Histogram

### Segmented Reduction

将不同长度的数组分段归约，每段独立求结果。典型实现：

- **CUB 库**：`cub::DeviceSegmentedReduce::Sum` 是生产级实现
- 自定义：每个 Block 处理一个 segment，segment 内 Block-Reduce

### Histogram

```cuda
// 共享内存私有化：每个 Block 维护局部 histogram，最后原子加合并
__global__ void histogram(int* data, int* hist, int N, int BINS) {
    extern __shared__ int local_hist[];
    if (threadIdx.x < BINS) local_hist[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(&local_hist[data[i]], 1); // 共享内存原子（块内竞争小）
    __syncthreads();

    if (threadIdx.x < BINS)
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]); // 全局合并
}
```

---

## 4.7 原子操作优化策略

原子操作是串行化的根源，需尽量减少竞争：

| 策略 | 说明 |
|------|------|
| 共享内存原子 + 全局合并 | Block 内用共享内存原子（竞争小），最后一次全局原子合并 |
| 私有化（Privatization） | 每线程/每 Warp 维护私有副本，最后合并，消除竞争 |
| Warp-Level Reduce + 原子 | Warp 内先归约为 1 个值，再原子加到全局 |
| 减少热点 | 哈希/分桶分散写目标，避免多线程竞争同一地址 |

```cuda
// ✅ Warp 归约后只有 1/32 的原子操作
float warp_sum = warpReduceSum(val);
if (lane_id == 0) atomicAdd(global_sum, warp_sum);
```
