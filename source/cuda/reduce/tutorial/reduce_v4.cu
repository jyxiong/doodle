/*
 * ============================================================
 *  Reduce Sum  V4 —— 展开最后一个 Warp（消除多余同步）
 * ============================================================
 *
 *  观察 V3 的问题：
 *    当 stride s <= 32 时，Block 内只有 <= 32 个线程活跃，
 *    即只有 1 个 Warp 在做实际工作。
 *    但循环仍在调用 `__syncthreads()`，而单个 Warp 内的线程
 *    在同一 SM 上以锁步（lockstep）方式执行，根本不需要同步！
 *
 *  优化思路：
 *    手动展开最后 6 步（s = 32, 16, 8, 4, 2, 1），
 *    完全省去这 6 次 `__syncthreads()` 调用。
 *
 *  注意：`__syncwarp()`
 *    展开后没有 `__syncthreads()`，编译器可能把共享内存读写
 *    优化到寄存器里，导致其他线程看到的是旧值。
 *    `__syncwarp()` 在每步之间插入 Warp 级内存屏障，
 *    强制编译器将所有共享内存写入对 Warp 内所有线程可见。
 *    相比 `volatile`，`__syncwarp()` 是 sm_70+ 的推荐做法，
 *    语义更清晰，且不阻止编译器对寄存器做其他优化。
 *
 *  展开示意（blockDim.x = 256，最后一个 Warp = tid 0..31）：
 *
 *    循环部分（s = 128, 64）：照常同步
 *    展开部分（s = 32..1）：只有 tid < 32 的线程执行，无需同步
 *      sdata[tid] += sdata[tid+32]  // s=32
 *      sdata[tid] += sdata[tid+16]  // s=16
 *      sdata[tid] += sdata[tid+ 8]  // s=8
 *      sdata[tid] += sdata[tid+ 4]  // s=4
 *      sdata[tid] += sdata[tid+ 2]  // s=2
 *      sdata[tid] += sdata[tid+ 1]  // s=1
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ─────────────────────────────────────────────────────────────
//  Warp 内展开辅助函数（__syncwarp 插入内存屏障，sm_70+ 推荐）
// ─────────────────────────────────────────────────────────────
__device__ void warp_reduce_v4(float* sdata, int tid)
{
    // 每步写入共享内存后调用 __syncwarp()，
    // 确保 Warp 内所有线程看到最新写入再进行下一步读取
    sdata[tid] += sdata[tid + 32]; __syncwarp();
    sdata[tid] += sdata[tid + 16]; __syncwarp();
    sdata[tid] += sdata[tid +  8]; __syncwarp();
    sdata[tid] += sdata[tid +  4]; __syncwarp();
    sdata[tid] += sdata[tid +  2]; __syncwarp();
    sdata[tid] += sdata[tid +  1];
}

// ─────────────────────────────────────────────────────────────
//  Kernel：循环归约 + 最后一个 Warp 展开
// ─────────────────────────────────────────────────────────────
__global__ void reduce_v4_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float a = (idx              < n) ? input[idx]              : 0.0f;
    float b = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    // 主循环：s 从 blockDim.x/2 折半，直到 s > 32
    //   活跃线程：tid = 0, 1, ..., s-1（相邻）
    //   内存访问：sdata[tid] += sdata[tid+s]（相邻，步长 1）
    //   （s <= 32 时只剩 1 个 Warp，交给展开部分处理）
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 展开：省去所有 __syncthreads()
    //   活跃线程：tid = 0, 1, ..., 31（1 个 Warp，Warp 内锁步无需同步）
    //   内存访问：sdata[tid] += sdata[tid+32/16/8/4/2/1]（6步，逐步折半）
    if (tid < 32) {
        warp_reduce_v4(sdata, tid);
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
    const int GRID_SIZE  = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    printf("=== Reduce V4: 展开最后一个 Warp，消除多余 __syncthreads() ===\n");
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
        reduce_v4_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        reduce_v4_kernel<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_input, d_output, N);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_result = (float)cpu_reduce(h_output, GRID_SIZE);

    print_result("V4: 展开最后 Warp，省去 6 次同步",
                 N, elapsed, gpu_result, cpu_result);

    printf("\n相比 V3 的改进:\n");
    printf("  [+] 最后 6 次 __syncthreads() → 无（Warp 锁步无需同步）\n");
    printf("  [+] 减少指令数，提升 IPC（每周期指令数）\n");
    printf("  [+] __syncwarp() 替代 volatile，语义更清晰（sm_70+ 推荐）\n");
    printf("\n仍存在的问题:\n");
    printf("  [-] 主循环仍有 __syncthreads()（blockDim.x/2 > 32 的轮次）\n");
    printf("  [-] 循环控制开销（计数、比较、跳转）\n");
    printf("\n下一步 → reduce_v5: 模板完全展开，消除所有循环开销\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
