/*
 * ============================================================
 *  Reduce Sum  V7 —— Warp Shuffle（寄存器直接通信）
 * ============================================================
 *
 *  V6 的块内归约仍依赖共享内存：
 *    - 写共享内存 → __syncthreads() → 读共享内存
 *
 *  Warp Shuffle（__shfl_down_sync）直接在同一 Warp 的线程间
 *  传递寄存器值，完全绕过共享内存：
 *    - 延迟：1~2 个时钟周期（vs 共享内存 ~20 周期）
 *    - 不需要 __syncthreads()（Warp 内天然同步）
 *    - 不消耗共享内存带宽
 *
 *  __shfl_down_sync(mask, val, offset) 语义：
 *    - 线程 tid 返回线程 (tid + offset) 的 val 值
 *    - mask：参与的线程掩码（0xFFFFFFFF = 全 Warp）
 *    - 超出 Warp 边界的线程返回自身的值
 *
 *  Warp Reduce 示意（8 线程，val=[8,3,7,1,6,5,2,4]）：
 *    offset=4: [8+6, 3+5, 7+2, 1+4,  6,  5,  2,  4]
 *              = [14,  8,  9,  5, ...]
 *    offset=2: [14+9, 8+5,  9,  5, ...]
 *              = [23, 13, ...]
 *    offset=1: [23+13, ...]
 *              = [36, ...]  ← lane 0 持有 Warp 的总和
 *
 *  完整流程：
 *    1. Grid-Stride Loop：每线程累加多个元素到寄存器
 *    2. Warp Reduce：32→1，结果在 lane 0
 *    3. 各 Warp 的 lane 0 写入共享内存（仅 blockDim/32 个值）
 *    4. 第一个 Warp 再次 Shuffle Reduce 这些值
 *    5. Block 的 thread 0 用 atomicAdd 写全局结果
 *       （atomicAdd 允许任意 gridDim，单 Pass 完成）
 *
 *  注意：需要 sm_30+ 支持 __shfl，sm_70+ 推荐用 __shfl_down_sync
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// 完整掩码：Warp 内所有 32 线程参与
#define FULL_MASK 0xFFFFFFFFu

// ─────────────────────────────────────────────────────────────
//  Warp-level reduce：用 Shuffle 将 32 线程的值折叠到 lane 0
// ─────────────────────────────────────────────────────────────
__device__ float warp_reduce_sum(float val)
{
    // offset 从 16 折半到 1，共 5 步，32→1
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val,  8);
    val += __shfl_down_sync(FULL_MASK, val,  4);
    val += __shfl_down_sync(FULL_MASK, val,  2);
    val += __shfl_down_sync(FULL_MASK, val,  1);
    return val;  // 只有 lane 0 的值有意义（其他 lane 可忽略）
}

// ─────────────────────────────────────────────────────────────
//  Kernel：Grid-Stride Loop + Warp Shuffle + Atomic 写结果
// ─────────────────────────────────────────────────────────────
__global__ void reduce_v7_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    // 共享内存只需存放每个 Warp 的局部和（blockDim/32 个值）
    extern __shared__ float warp_sums[];  // 大小 = blockDim.x / 32

    int tid     = threadIdx.x;
    int lane    = tid & 31;          // lane id within warp (tid % 32)
    int warp_id = tid >> 5;          // warp id within block (tid / 32)
    int idx     = blockIdx.x * blockDim.x + tid;
    int stride  = blockDim.x * gridDim.x;

    // ── Step 1: Grid-Stride Loop，寄存器累加 ──────────────────
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // ── Step 2: Warp Reduce（32→1，无共享内存，无同步）────────
    //   活跃线程：Warp 内全部 32 个 lane（无分叉）
    //   内存访问：无（全程寄存器间通信，__shfl_down_sync）
    sum = warp_reduce_sum(sum);

    // ── Step 3: 各 Warp 的 lane 0 将结果写共享内存 ────────────
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();  // 等待所有 Warp 写完共享内存

    // ── Step 4: 第一个 Warp 对所有 warp_sums 再做 Shuffle Reduce ──
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        // 只有前 num_warps 个 lane 有有效数据
        sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }

    // ── Step 5: Block 的 thread 0 用 atomicAdd 写最终结果 ─────
    //   使用 atomicAdd 省去二级归约，单 Pass 完成全局 Reduce
    if (tid == 0) {
        atomicAdd(output, sum);
    }
}

// ─────────────────────────────────────────────────────────────
//  主程序
// ─────────────────────────────────────────────────────────────
int main()
{
    const int N          = 1 << 24;
    const int BLOCK_SIZE = 256;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const int GRID_SIZE = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              4 * prop.multiProcessorCount);

    printf("=== Reduce V7: Warp Shuffle + Atomic，单 Pass 完成归约 ===\n");
    printf("N = %d (%.1f MB),  block=%d,  grid=%d\n",
           N, N * sizeof(float) / 1024.0f / 1024.0f, BLOCK_SIZE, GRID_SIZE);
    printf("GPU: %s,  SM 数 = %d\n\n", prop.name, prop.multiProcessorCount);

    float* h_input = (float*)malloc(N * sizeof(float));
    init_data(h_input, N);
    double cpu_result = cpu_reduce(h_input, N);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  N          * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 1          * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 共享内存：每 Warp 一个 float
    size_t smem = (BLOCK_SIZE / 32) * sizeof(float);

    // Warmup
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        reduce_v7_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        reduce_v7_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    }
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    float gpu_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_result, d_output, sizeof(float),
                          cudaMemcpyDeviceToHost));

    print_result("V7: Warp Shuffle + Atomic，单 Pass",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V6 的改进:\n");
    printf("  [+] Warp 内归约用 Shuffle（寄存器通信），无共享内存读写\n");
    printf("  [+] Shuffle 延迟 ~1 周期（vs 共享内存 ~20 周期）\n");
    printf("  [+] atomicAdd 单 Pass 完成全局归约，无需第二轮 Kernel\n");
    printf("  [+] 共享内存使用量极小（%zu bytes per Block）\n", smem);
    printf("\n优化总结（V0 → V7）:\n");
    printf("  V0: 取模 + Divergence                  （基准）\n");
    printf("  V1: 消除取模                            (+~5%%)\n");
    printf("  V2: 顺序寻址，无 Divergence/Bank冲突   (+~2x)\n");
    printf("  V3: Load 时首加，Block 数减半           (+~10%%)\n");
    printf("  V4: 展开最后一个 Warp                  (+~5%%)\n");
    printf("  V5: 模板完全展开                        (+~5%%)\n");
    printf("  V6: 每线程多元素，提升算术强度          (+~20%%)\n");
    printf("  V7: Warp Shuffle + 单 Pass              (+~10%%)\n");

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
