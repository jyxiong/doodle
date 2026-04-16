/*
 * ============================================================
 *  Reduce Sum  V5 —— 模板完全展开（消除所有循环开销）
 * ============================================================
 *
 *  V4 的主循环在编译期无法确定迭代次数（blockDim.x 是运行期值），
 *  编译器无法展开，每轮仍有：
 *    - 循环计数器递减
 *    - 条件判断
 *    - 分支跳转
 *
 *  优化思路：
 *    将 blockDim.x 作为模板参数传入，编译器在编译期就知道其值，
 *    可以：
 *      1. 将 `if (tid < s)` 中的 s 视为常量 → 死代码消除
 *      2. 完全展开 for 循环 → 无循环控制开销
 *      3. 潜在的更好的指令排布和 IPC 优化
 *
 *  模板展开原理：
 *    `if (blockSize >= 512)` 在编译期求值
 *      - 若 blockSize < 512，整个 if 块被编译器直接删除
 *      - 若 blockSize >= 512，该块保留并内联
 *    → 生成的机器码针对特定 blockSize 最优化
 *
 *  调度方式（在 main 中手动分发）：
 *    switch (blockSize) {
 *      case 256: reduce_v5<256><<<...>>>(...); break;
 *      case 128: reduce_v5<128><<<...>>>(...); break;
 *      ...
 *    }
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Warp 内展开（与 V4 相同，使用 volatile）
// ─────────────────────────────────────────────────────────────
__device__ void warp_reduce_v5(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

// ─────────────────────────────────────────────────────────────
//  Kernel：模板参数 blockSize，编译期完全展开归约循环
// ─────────────────────────────────────────────────────────────
template <unsigned int blockSize>
__global__ void reduce_v5_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;

    float a = (idx           < n) ? input[idx]            : 0.0f;
    float b = (idx + blockSize < n) ? input[idx + blockSize] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    // 编译期展开：每个 if 判断在编译时已知结果
    //   活跃线程：每步为 tid = 0..s-1（相邻，s 从 blockSize/2 折半到 32）
    //   内存访问：sdata[tid] += sdata[tid+s]（相邻，步长 1）
    //   blockSize 不满足的分支直接被编译器删除（zero overhead）
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) { sdata[tid] += sdata[tid +  64]; }
        __syncthreads();
    }

    // 最后一个 Warp 展开（与 V4 相同）
    if (tid < 32) {
        warp_reduce_v5(sdata, tid);
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ─────────────────────────────────────────────────────────────
//  主程序：依据 BLOCK_SIZE 分发正确的模板实例
// ─────────────────────────────────────────────────────────────
template <unsigned int blockSize>
void launch_reduce(const float* d_input, float* d_output, int n, int grid)
{
    size_t smem = blockSize * sizeof(float);
    reduce_v5_kernel<blockSize><<<grid, blockSize, smem>>>(d_input, d_output, n);
}

int main()
{
    const int N          = 1 << 24;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE  = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    printf("=== Reduce V5: 模板完全展开，消除循环控制开销 ===\n");
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

    // Warmup
    for (int i = 0; i < 3; i++)
        launch_reduce<BLOCK_SIZE>(d_input, d_output, N, GRID_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        launch_reduce<BLOCK_SIZE>(d_input, d_output, N, GRID_SIZE);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_result = (float)cpu_reduce(h_output, GRID_SIZE);

    print_result("V5: 模板展开，零循环开销",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V4 的改进:\n");
    printf("  [+] 主归约循环完全展开，无循环计数/比较/跳转指令\n");
    printf("  [+] 编译器可做更激进的指令调度和 ILP 优化\n");
    printf("  [+] 不满足条件的 if 块直接从机器码中删除\n");
    printf("\n仍存在的问题:\n");
    printf("  [-] 每线程仅处理 2 个元素（Load/Compute 比仍偏低）\n");
    printf("      GPU 访存延迟约 400 周期，需要足够算术指令来隐藏\n");
    printf("\n下一步 → reduce_v6: 每线程处理多个元素（提升算术强度）\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
