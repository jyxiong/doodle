/*
 * ============================================================
 *  Reduce Sum  V3 —— 全局 Load 时首次相加（减少空闲线程）
 * ============================================================
 *
 *  观察 V2 的问题：
 *    第一轮（stride = blockDim.x/2）只有 50% 的线程做加法，
 *    另外 50% 的线程仅仅执行了一次 Load 就空闲了。
 *
 *    如果把这 50% 空闲线程的 Load 也变成有用工作，就能：
 *      - 用一半的 Block 数完成同样的工作
 *      - 每个 Block 加载 2*blockDim.x 个元素
 *      - Thread 0 加载 input[blockIdx*blockDim + tid]
 *                      + input[blockIdx*blockDim + tid + blockDim]
 *
 *  示意（blockDim.x = 4，处理 8 个元素）：
 *
 *    V2（8个元素，2个Block，每块4线程）：
 *      Block0: load [a0,a1,a2,a3] → reduce → s0
 *      Block1: load [a4,a5,a6,a7] → reduce → s1
 *      CPU: s0 + s1 = sum
 *
 *    V3（8个元素，1个Block，每块4线程）：
 *      Block0: load [a0+a4, a1+a5, a2+a6, a3+a7] → reduce → sum
 *      ──────────────────────────────────────────────
 *      Block 数减半，每线程多做一次加法
 *      但全局内存读取量相同（都读了 8 个元素）
 *
 *  收益：
 *    - Block 数减半 → Grid 减小 → 更少的 Block 调度开销
 *    - 树状归约深度不变（log2(blockDim)），但起始有效数据量翻倍
 *    - 线程利用率从 50% → 100%（在 Load 阶段）
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Kernel：每个 Block 处理 2 * blockDim.x 个元素
// ─────────────────────────────────────────────────────────────
__global__ void reduce_v3_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    // 每个 Block 覆盖的全局起始索引（跨度翻倍）
    int idx  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 加载两个元素并立即相加后存入共享内存
    //   若越界则补 0（边界安全）
    float a = (idx < n) ? input[idx] : 0.0f;
    float b = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    // 后续与 V2 相同：顺序寻址树状归约
    //   活跃线程：tid = 0, 1, ..., s-1（相邻）
    //   内存访问：sdata[tid] += sdata[tid+s]（相邻，步长 1）
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
    // 每个 Block 处理 2*BLOCK_SIZE 个元素，所以 Grid 减半
    const int GRID_SIZE  = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    printf("=== Reduce V3: Load 时首次相加，线程利用率翻倍 ===\n");
    printf("N = %d (%.1f MB),  block=%d,  grid=%d  (V2 的一半)\n\n",
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
        reduce_v3_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        reduce_v3_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_result = (float)cpu_reduce(h_output, GRID_SIZE);

    print_result("V3: Load 时首次相加，Block 数减半",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V2 的改进:\n");
    printf("  [+] Grid 减半（%d → %d Block），Block 调度开销降低\n",
           (N + BLOCK_SIZE - 1) / BLOCK_SIZE, GRID_SIZE);
    printf("  [+] Load 阶段线程利用率 50%% → 100%%\n");
    printf("  [+] 全局内存读取量不变，但计算量/访存量比提升\n");
    printf("\n仍存在的问题:\n");
    printf("  [-] 最后 5~6 轮归约（s <= 32）只有 1 个 Warp 活跃\n");
    printf("      但仍在调用 __syncthreads()（不必要的同步开销）\n");
    printf("\n下一步 → reduce_v4: 展开最后一个 Warp，消除多余同步\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
