/*
 * ============================================================
 *  Reduce Sum  V2 —— 顺序寻址（消除 Divergence 和 Bank Conflict）
 * ============================================================
 *
 *  核心变化：将 stride 的遍历方向从"从小到大"改为"从大到小"
 *
 *             V0/V1（交错）        V2（顺序）
 *    stride:  1, 2, 4, 8, ...     N/2, N/4, ..., 4, 2, 1
 *    访问:    sdata[0]+=sdata[1]   sdata[0]+=sdata[N/2]
 *             sdata[2]+=sdata[3]   sdata[1]+=sdata[N/2+1]
 *
 *  改进 1 —— 消除 Warp Divergence
 *    条件改为 `if (tid < s)`，活跃线程为 tid=0,1,...,s-1（连续）
 *    s >= 32 时，整个 Warp 内所有线程要么全活跃要么全空闲
 *             → 无 Divergence！
 *
 *  改进 2 —— 消除 Shared Memory Bank Conflict
 *    V1 后期 stride 大，访问 sdata[0] 和 sdata[stride]，
 *    stride 是 32 的倍数时映射同一 Bank → 冲突。
 *    V2 中活跃线程 tid=0..s-1 访问 sdata[tid] 和 sdata[tid+s]，
 *    相邻线程访问相邻地址 → 无 Bank Conflict。
 *
 *  归约过程（blockDim.x = 8 的示意）：
 *
 *    初始 sdata: [a0, a1, a2, a3, a4, a5, a6, a7]
 *    stride=4:   [a0+a4, a1+a5, a2+a6, a3+a7, a4, a5, a6, a7]  (tid 0..3 活跃)
 *    stride=2:   [a0+a4+a2+a6, a1+a5+a3+a7, ...]                (tid 0..1 活跃)
 *    stride=1:   [sum, ...]                                       (tid 0   活跃)
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Kernel：stride 从 blockDim.x/2 向下折半
// ─────────────────────────────────────────────────────────────
__global__ void reduce_v2_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 顺序寻址：stride 从大到小
    //   活跃线程：tid = 0, 1, ..., s-1（相邻，连续块，随 s 缩减）
    //   内存访问：sdata[tid] += sdata[tid+s]（相邻线程访问相邻地址，步长 1）
    //   → s >= 32：整 Warp 全活跃或全空闲 → 无 Divergence
    //   → 无 Bank Conflict：相邻线程访问连续 Bank
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
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

    printf("=== Reduce V2: 顺序寻址（无 Divergence，无 Bank Conflict）===\n");
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
        reduce_v2_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        reduce_v2_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_result = (float)cpu_reduce(h_output, GRID_SIZE);

    print_result("V2: 顺序寻址，无 Divergence，无 Bank Conflict",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V1 的改进:\n");
    printf("  [+] 无 Warp Divergence：活跃线程连续，整 Warp 同路径\n");
    printf("  [+] 无 Bank Conflict：相邻线程访问相邻共享内存地址\n");
    printf("\n仍存在的问题:\n");
    printf("  [-] 线程利用率低：第一轮 50%% 线程立即空闲\n");
    printf("      -> %d 本可以做有用工作的线程在 Load 阶段就被浪费\n",
           GRID_SIZE * BLOCK_SIZE / 2);
    printf("\n下一步 → reduce_v3: 全局 Load 时顺便加法，线程利用率翻倍\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
