/*
 * ============================================================
 *  Reduce Sum  V1 —— 消除取模，交错寻址（仍有 Divergence）
 * ============================================================
 *
 *  相对 V0 的改进：
 *    将 `tid % (2*s) == 0` 替换为乘法索引：
 *      index = 2 * s * tid
 *    消除了慢速整数取模，改用廉价乘法。
 *
 *  仍存在的问题：
 *    1. Warp Divergence 仍存在（最后几轮）
 *       前期轮次（s 较大）：活跃线程连续，整 Warp 全活跃或全空闲，无 Divergence
 *       后期轮次（s >= 64）：活跃线程不足 32，Warp 0 内部出现分叉 → Divergence
 *
 *    2. Shared Memory Bank Conflict（后期轮次）
 *       s=16 时：活跃线程访问 sdata[0] 和 sdata[16]
 *                stride=16 → 映射到相同 Bank → 冲突！
 *
 *  改进索引方案示意（blockDim.x = 8，stride s=1）：
 *    tid=0: index=0, 访问 sdata[0] += sdata[1]
 *    tid=1: index=2, 访问 sdata[2] += sdata[3]
 *    tid=2: index=4, 访问 sdata[4] += sdata[5]
 *    tid=3: index=6, 访问 sdata[6] += sdata[7]
 *    tid=4..7: index >= 8，不活跃（if 屏蔽）
 *
 *    → tid=0..3 活跃，tid=4..7 空闲
 *    → 比 V0 的交替模式更"集中"，但 Warp 0=tid[0..7] 内
 *      tid=0..3 活跃、tid=4..7 空闲 → 仍有 Divergence
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Kernel：乘法索引替换取模，活跃线程更集中但仍有 Divergence
// ─────────────────────────────────────────────────────────────
__global__ void reduce_v1_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 改进：用乘法计算访问 index，避免取模
    //   V0: if (tid % (2*s) == 0)        ← 整数取模，慢
    //   V1: int index = 2*s*tid;
    //       if (index < blockDim.x)      ← 比较，快
    //
    //   活跃线程：tid = 0, 1, 2, ..., blockDim.x/(2s)-1（相邻，连续块）
    //   内存访问：sdata[2s·tid] += sdata[2s·tid + s]（下标依然跳跃，步长 2s）
    //   → 整 Warp 要么全活跃要么全空闲，大部分轮次无 Divergence
    //   → 仅最后几轮（活跃线程 < 32）Warp 0 内部才出现分叉
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * (int)tid;
        if (index < (int)blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ─────────────────────────────────────────────────────────────
//  主程序
// ─────────────────────────────────────────────────────────────
int main()
{
    const int N          = 1 << 24;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("=== Reduce V1: 消除取模，交错寻址（仍有 Divergence）===\n");
    printf("N = %d (%.1f MB),  block=%d,  grid=%d\n\n",
           N, N * sizeof(float) / 1024.0f / 1024.0f, BLOCK_SIZE, GRID_SIZE);

    float* h_input  = (float*)malloc(N         * sizeof(float));
    float* h_output = (float*)malloc(GRID_SIZE * sizeof(float));
    init_data(h_input, N);
    double cpu_result = cpu_reduce(h_input, N);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  N         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    size_t smem = BLOCK_SIZE * sizeof(float);

    for (int i = 0; i < 3; i++)
        reduce_v1_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        reduce_v1_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_result = (float)cpu_reduce(h_output, GRID_SIZE);

    print_result("V1: 乘法索引，消除取模，仍有 Divergence",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V0 的改进:\n");
    printf("  [+] 消除慢速 %% 取模，改用廉价乘法计算 index\n");
    printf("\n仍存在的问题:\n");
    printf("  [-] Warp Divergence：第一轮半数线程空闲\n");
    printf("  [-] Bank Conflict：后期 stride 增大时出现\n");
    printf("\n下一步 → reduce_v2: 顺序寻址，彻底消除 Divergence 和 Bank Conflict\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
