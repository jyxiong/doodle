/*
 * ============================================================
 *  GEMM  V1 —— Shared Memory Tiling（分块提升数据复用率）
 * ============================================================
 *
 *  V0 的核心问题：
 *    计算 C[m][n] 需要从全局内存读 A 的整行（K 个元素）和
 *    B 的整列（K 个元素），同一 Block 内 TILE×TILE 个线程
 *    读取大量重叠数据，全局内存带宽成为瓶颈。
 *
 *  Tiling 思想：
 *    将 K 维度切成步长为 TILE 的小块，每次只取 A 的一个
 *    TILE×TILE 子块和 B 的一个 TILE×TILE 子块，
 *    加载到 Shared Memory，全 Block 内线程协作复用。
 *
 *  每次迭代（tile 步骤）：
 *    1. Block 内 TILE×TILE 个线程协作加载：
 *         As[ty][tx] = A[row][t*TILE + tx]   ← A 的一行片段
 *         Bs[ty][tx] = B[t*TILE + ty][col]   ← B 的一列片段
 *    2. __syncthreads()，确保共享内存加载完毕
 *    3. 每线程对 TILE 个元素做点积：sum += As[ty][k] * Bs[k][tx]
 *    4. __syncthreads()，确保计算完毕再加载下一块
 *
 *  数据复用分析（TILE=16，Block=16×16=256 线程）：
 *    全局内存读取：
 *      A：TILE×TILE 个元素，被 Block 内 TILE 列线程共享 → 复用 TILE 次
 *      B：TILE×TILE 个元素，被 Block 内 TILE 行线程共享 → 复用 TILE 次
 *    算术强度 ≈ TILE FLOP/Byte（V0 的 TILE=16 倍）
 *
 *  Shared Memory 布局（各维 TILE=16）：
 *    As[TILE][TILE]  —— A 的子块，ty 对应 row，tx 对应 k
 *    Bs[TILE][TILE]  —— B 的子块，ty 对应 k，tx 对应 col
 *
 *  示意（TILE=2，输出 C 的 2×2 子块）：
 *
 *    t=0: As=A[0:2][0:2], Bs=B[0:2][0:2]
 *         sum[0][0] += As[0][0]*Bs[0][0] + As[0][1]*Bs[1][0]
 *    t=1: As=A[0:2][2:4], Bs=B[2:4][0:2]
 *         sum[0][0] += As[0][0]*Bs[0][0] + As[0][1]*Bs[1][0]
 *    ...累加完所有 tile 后写入 C
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

#define TILE 16

// ─────────────────────────────────────────────────────────────
//  Kernel：Shared Memory Tiling
// ─────────────────────────────────────────────────────────────
__global__ void gemm_v1_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    // Block 内线程坐标
    int tx = threadIdx.x;   // 列方向
    int ty = threadIdx.y;   // 行方向

    // 当前线程对应的输出矩阵坐标
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    // Shared Memory：A 和 B 各一个 TILE×TILE 子块
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    // 沿 K 方向分块迭代
    // 每次迭代加载 A[row][t*TILE:t*TILE+TILE] 和 B[t*TILE:t*TILE+TILE][col]
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 协作加载 A 的子块到 As
        //   当前线程负责 A[row][t*TILE+tx]
        int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // 协作加载 B 的子块到 Bs
        //   当前线程负责 B[t*TILE+ty][col]
        int b_row = t * TILE + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // 等待所有线程加载完毕
        __syncthreads();

        // 计算本 tile 的点积贡献
        // As[ty][k]：A 子块第 ty 行第 k 个元素
        // Bs[k][tx]：B 子块第 k 行第 tx 个元素
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 确保所有线程完成计算后再加载下一 tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ─────────────────────────────────────────────────────────────
//  主程序
// ─────────────────────────────────────────────────────────────
int main()
{
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

    printf("=== GEMM V1: Shared Memory Tiling (TILE=%d) ===\n", TILE);
    printf("M=%d, N=%d, K=%d\n\n", M, N, K);

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float* h_A     = (float*)malloc(sA);
    float* h_B     = (float*)malloc(sB);
    float* h_C     = (float*)calloc(M * N, sizeof(float));
    float* h_C_ref = (float*)calloc(M * N, sizeof(float));

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    printf("计算 CPU 参考结果...\n");
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sC));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    for (int i = 0; i < 3; i++)
        gemm_v1_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        gemm_v1_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    bool correct = verify_gemm(h_C, h_C_ref, M, N);

    print_result("V1: Shared Memory Tiling", M, N, K, elapsed, correct);

    printf("\n相比 V0 的改进:\n");
    printf("  [+] TILE×TILE 子块加载到 Shared Memory，复用率提升 TILE 倍\n");
    printf("  [+] 算术强度：1 → %d FLOP/Byte\n", TILE);
    printf("  [+] 全局内存读取次数：2*M*N*K → 2*M*N*K/TILE\n");
    printf("\n仍存在的问题:\n");
    printf("  [-] 每线程只计算一个输出元素，寄存器利用率偏低\n");
    printf("  [-] Shared Memory 带宽可能有 Bank Conflict\n");
    printf("\n下一步 → gemm_v2: Thread Coarsening，每线程计算多个输出元素\n");

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
