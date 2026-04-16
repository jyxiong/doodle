/*
 * ============================================================
 *  Reduce Sum  V0 —— 交错寻址 + 分支发散（基准版本）
 * ============================================================
 *
 *  本实现是最直观的 GPU 树状归约，展示了两个典型性能瓶颈：
 *
 *  1. 慢速取模运算（tid % (2*s)）
 *     整数取模相当于整数除法，约需 30+ 时钟周期。
 *
 *  2. Warp Divergence（Warp 分支发散）
 *     CUDA 以 32 个线程为一个 Warp 同步执行同一指令。
 *     当同一 Warp 内的线程走不同分支时，SM 必须串行执行
 *     两条路径，主动线程执行时空闲线程被屏蔽，效率减半。
 *
 *     以 stride=1 为例，Warp 0 含 tid=0..31：
 *       活跃：tid=0,2,4,...,30（满足 tid%2==0）
 *       空闲：tid=1,3,5,...,31
 *     同一 Warp 内存在分叉 → Divergence!
 *
 *  归约过程（blockDim.x = 8 的示意）：
 *
 *    初始 sdata: [8, 3, 7, 1, 6, 5, 2, 4]
 *    stride=1:   [11, -, 8, -, 11, -, 6, -]   (0+=1, 2+=3, 4+=5, 6+=7)
 *    stride=2:   [19, -, -, -, 17, -, -, -]   (0+=2, 4+=6)
 *    stride=4:   [36, -, -, -, -,  -, -, -]   (0+=4)
 *    结果 = sdata[0] = 36 ✓
 *
 *  后续版本将逐步消除这些问题。
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Kernel：交错寻址，stride 从 1 翻倍到 blockDim.x/2
// ─────────────────────────────────────────────────────────────
__global__ void reduce_v0_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    // 线程在 Block 内的索引
    int tid = threadIdx.x;
    // 全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存，越界补 0
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    // 确保所有线程完成加载
    __syncthreads();

    // 树状归约：stride 每轮翻倍
    //   问题 1：tid % (2*s) 是整数取模，极慢
    //   问题 2：同一 Warp 内满足 / 不满足条件的线程同时存在 → Divergence
    //
    //   活跃线程：tid = 0, 2s, 4s, ...（跳跃，每 2s 个里取 1 个）
    //   内存访问：sdata[tid] += sdata[tid+s]（下标也跳跃，步长 2s）
    //   → 同一 Warp（tid 连续）内奇偶交替活跃/空闲 → Divergence 每轮都有
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) { // 每 2*s 个线程有一个活跃
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 确保所有线程完成本轮，再开始下一轮
    }

    // Thread 0 将本块结果写回全局内存
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ─────────────────────────────────────────────────────────────
//  主程序
// ─────────────────────────────────────────────────────────────
int main()
{
    const int N          = 1 << 24;   // 16 M 个 float
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("=== Reduce V0: 交错寻址 + 分支发散 ===\n");
    printf("N = %d (%.1f MB),  block=%d,  grid=%d\n\n",
           N, N * sizeof(float) / 1024.0f / 1024.0f, BLOCK_SIZE, GRID_SIZE);

    // ── 分配主机内存 ──────────────────────────────────────────
    float* h_input  = (float*)malloc(N          * sizeof(float));
    float* h_output = (float*)malloc(GRID_SIZE  * sizeof(float));
    init_data(h_input, N);

    double cpu_result = cpu_reduce(h_input, N);

    // ── 分配设备内存 ──────────────────────────────────────────
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  N         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, GRID_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    size_t smem = BLOCK_SIZE * sizeof(float);

    // ── Warmup ────────────────────────────────────────────────
    for (int i = 0; i < 3; i++)
        reduce_v0_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Benchmark ─────────────────────────────────────────────
    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        reduce_v0_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    // ── 取回结果，CPU 完成最后一级归约（partial sums → final sum）──
    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double partial_sum = cpu_reduce(h_output, GRID_SIZE);
    float  gpu_result  = (float)partial_sum;

    print_result("V0: 交错寻址 + % 取模 + Divergence",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n性能瓶颈分析:\n");
    printf("  [1] tid %% (2*s) 整数取模  —— 约 30 周期，远慢于加法\n");
    printf("  [2] Warp Divergence        —— stride=1 时半数线程空闲\n");
    printf("  [3] Bank Conflict（后期）  —— stride 增大时访问模式发散\n");
    printf("\n下一步 → reduce_v1: 消除取模，改用乘法计算 index\n");

    // ── 释放资源 ──────────────────────────────────────────────
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
