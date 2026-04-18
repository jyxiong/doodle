/*
 * ============================================================
 *  GEMM  V0 —— 朴素实现（每线程计算一个输出元素）
 * ============================================================
 *
 *  C(M×N) = A(M×K) × B(K×N)，所有矩阵行优先存储。
 *
 *  思路：
 *    将输出矩阵 C 的元素 C[m][n] 分配给一个线程。
 *    该线程遍历 K 个元素，依次从 A 的第 m 行和 B 的第 n 列
 *    读取数据相乘累加。
 *
 *  线程/Block 映射：
 *    blockDim = (TILE, TILE)，每个线程负责一个 C[m][n]
 *    gridDim  = (ceil(N/TILE), ceil(M/TILE))
 *               x → 列方向（N），y → 行方向（M）
 *
 *  内存访问分析：
 *    计算 C[m][n] 需要：
 *      A 的第 m 行：A[m][0..K-1]  → 访问 K 个全局内存元素
 *      B 的第 n 列：B[0..K-1][n]  → 访问 K 个全局内存元素
 *
 *    全局内存读取量：2 * M * N * K 个 float
 *    计算量：2 * M * N * K 次浮点运算（乘-加）
 *    算术强度 ≈ 1 FLOP/Byte（极低，内存受限）
 *
 *  性能瓶颈：
 *    1. B 矩阵按列访问 —— stride = N，不连续，无法合并（uncoalesced）
 *       同一 Warp（相邻线程 n 连续）访问 B[k][n], B[k][n+1], ...
 *       反而是合并的，A 矩阵才有问题（同一 Warp 同一 m，所有 k 相同）。
 *       实际上两者都对同一 cache line 有大量重复读取。
 *    2. 无数据复用：相邻线程读取 A/B 的大量重叠数据，
 *       每次都从全局内存（L2/DRAM）读取，cache 利用率低。
 *
 *  后续版本将通过 Shared Memory Tiling 大幅提升数据复用率。
 *
 *  归约过程示意（4×4 矩阵，TILE=2）：
 *
 *    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + ...  ← Thread(0,0)
 *    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + ...  ← Thread(1,0)
 *    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + ...  ← Thread(0,1)
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// Block 内线程组织：TILE×TILE
#define TILE 16

// ─────────────────────────────────────────────────────────────
//  Kernel：每线程计算 C 的一个元素
// ─────────────────────────────────────────────────────────────
__global__ void gemm_v0_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    // 当前线程负责的输出行列
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // n 方向（列）
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // m 方向（行）

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A[row][k]：row 固定，k 变化 → 同 Warp 内多个线程访问同一地址（广播）
        // B[k][col]：col 随线程变化 → 同 Warp 内连续线程访问连续地址 → 合并
        sum += A[row * K + k] * B[k * N + col];
    }

    // C = alpha * A*B + beta * C
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

// ─────────────────────────────────────────────────────────────
//  主程序
// ─────────────────────────────────────────────────────────────
int main()
{
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

    printf("=== GEMM V0: 朴素实现（每线程一个输出元素）===\n");
    printf("M=%d, N=%d, K=%d,  alpha=%.1f, beta=%.1f\n\n", M, N, K, alpha, beta);

    // ── 分配主机内存 ──────────────────────────────────────────
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

    // CPU 参考结果（矩阵较小，直接算）
    printf("计算 CPU 参考结果...\n");
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta);

    // ── 分配设备内存 ──────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sC));

    // ── 配置 Grid / Block ────────────────────────────────────
    dim3 block(TILE, TILE);                              // 16×16 = 256 线程
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // ── Warmup ────────────────────────────────────────────────
    for (int i = 0; i < 3; i++)
        gemm_v0_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Benchmark ─────────────────────────────────────────────
    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        gemm_v0_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    // ── 取回结果并验证 ────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    bool correct = verify_gemm(h_C, h_C_ref, M, N);

    print_result("V0: 朴素实现", M, N, K, elapsed, correct);

    printf("\n性能瓶颈分析:\n");
    printf("  [1] 无 Shared Memory 复用\n");
    printf("      同一 Warp 中 16 线程共享 A[row][k]（广播可以，但重复读）\n");
    printf("      相邻 Block 的线程大量重叠读取 A 和 B 的同一 cache line\n");
    printf("  [2] 算术强度 ≈ 1 FLOP/Byte，远低于 GPU 峰值计算:访存比\n");
    printf("  [3] 全局内存带宽是主要瓶颈，计算单元利用率极低\n");
    printf("\n下一步 → gemm_v1: Shared Memory Tiling，大幅提升数据复用率\n");

    // ── 释放资源 ──────────────────────────────────────────────
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
