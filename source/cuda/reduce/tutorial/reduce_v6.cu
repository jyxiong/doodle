/*
 * ============================================================
 *  Reduce Sum  V6 —— 每线程处理多个元素（Grid-Stride Loop）
 * ============================================================
 *
 *  V5 中每线程仅处理 2 个全局内存元素，访存延迟（~400 周期）
 *  远大于两次加法（约 2 周期），Compute-to-Memory 比值很低。
 *
 *  优化思路：让每个线程通过 Grid-Stride Loop 处理多个元素，
 *  在寄存器中累加，最后才写入共享内存。
 *
 *  Grid-Stride Loop 模式：
 *    int stride = blockDim.x * gridDim.x;  // 整个 Grid 的总线程数
 *    for (int i = idx; i < n; i += stride) {
 *        sum += input[i];
 *    }
 *
 *  收益：
 *    1. 算术强度提升：每次全局内存 Load 对应多次寄存器加法
 *    2. 访存延迟隐藏：多次 Load 流水线执行，隐藏带宽延迟
 *    3. Grid 大小可固定为较小值（如 SM 数 × 若干），
 *       避免调度数千个 Block 的开销
 *    4. 更好的线程局部性（连续 stride 访问，访存合并）
 *
 *  以 N=16M, blockDim=256, gridDim=1024 为例：
 *    每线程处理的元素数 = 16M / (256 × 1024) = 64 个元素
 *    64 次 Load + 64 次加法，比 2 次 Load + 1 次加法强 32 倍
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Warp 展开（与 V4/V5 相同）
// ─────────────────────────────────────────────────────────────
__device__ void warp_reduce_v6(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

// ─────────────────────────────────────────────────────────────
//  Kernel：Grid-Stride Loop + 模板展开归约
// ─────────────────────────────────────────────────────────────
template <unsigned int blockSize>
__global__ void reduce_v6_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    int tid    = threadIdx.x;
    int idx    = blockIdx.x * blockSize + threadIdx.x;
    int stride = blockSize * gridDim.x;  // 整个 Grid 的步长

    // Grid-Stride Loop：每线程累加多个全局内存元素到寄存器
    //   活跃线程：全部线程（所有 tid 均参与，无空闲）
    //   内存访问：input[tid], input[tid+stride], ...（连续步长，访存合并）
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // 后续块内归约：模板展开（与 V5 相同）
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

    if (tid < 32) {
        warp_reduce_v6(sdata, tid);
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

    // Grid 大小：通常设为 SM 数 × 几倍，这里用较小的 Grid
    // 让每线程处理更多元素（约 N / (BLOCK_SIZE × GRID_SIZE) ≈ 64）
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    // 每 SM 启动 4 个 Block，控制总 Block 数在合理范围
    const int GRID_SIZE = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              4 * prop.multiProcessorCount);

    int elems_per_thread = (N + GRID_SIZE * BLOCK_SIZE - 1) / (GRID_SIZE * BLOCK_SIZE);

    printf("=== Reduce V6: Grid-Stride Loop，每线程处理多元素 ===\n");
    printf("N = %d (%.1f MB),  block=%d,  grid=%d\n",
           N, N * sizeof(float) / 1024.0f / 1024.0f, BLOCK_SIZE, GRID_SIZE);
    printf("SM 数 = %d,  每线程约处理 %d 个元素\n\n",
           prop.multiProcessorCount, elems_per_thread);

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
        reduce_v6_kernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, smem>>>(
            d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        reduce_v6_kernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, smem>>>(
            d_input, d_output, N);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_result = (float)cpu_reduce(h_output, GRID_SIZE);

    print_result("V6: Grid-Stride Loop，高算术强度",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V5 的改进:\n");
    printf("  [+] 每线程处理 ~%d 个元素，算术强度大幅提升\n", elems_per_thread);
    printf("  [+] 访存延迟被更多计算指令有效隐藏\n");
    printf("  [+] Grid 更小（%d），Block 调度开销降低\n", GRID_SIZE);
    printf("  [+] Load 访存合并依然完美（连续 stride 访问）\n");
    printf("\n下一步 → reduce_v7: Warp Shuffle，用寄存器通信替代共享内存\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
