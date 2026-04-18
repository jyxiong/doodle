/*
 * ============================================================
 *  GEMM  V2 —— Thread Coarsening（每线程计算 BN×BM 个输出）
 * ============================================================
 *
 *  V1 的问题：
 *    每线程只计算 C 的 1 个元素，消耗 TILE 次 Shared Memory 读，
 *    寄存器层面没有充分复用 A 和 B 的数据。
 *
 *  Thread Coarsening 思想：
 *    让每个线程计算输出矩阵 C 的 BN×BM（= 8×8）个元素。
 *    线程数不变，Block 覆盖更大的输出区域（BN*TILE × BM*TILE）。
 *
 *    Block 大小：BK×BK = 8×8（参与加载的线程网格）
 *    每个线程负责：BN×BM = 8×8 = 64 个输出元素
 *    Block 输出区域：BK*BN × BK*BM = 64×64
 *
 *  关键收益（寄存器级复用）：
 *    加载一次 As[ty][k]（1 个寄存器值），用于计算 BM 个输出列
 *    加载一次 Bs[k][tx]（1 个寄存器值），用于计算 BN 个输出行
 *    → 寄存器级数据复用，避免重复读取 Shared Memory
 *
 *  内循环结构：
 *    for k in [0, BK):
 *        regA[m] = As[ty*BN + m][k]  for m in [0, BN)
 *        regB[n] = Bs[k][tx*BM + n]  for n in [0, BM)
 *        for m in [0, BN):
 *            for n in [0, BM):
 *                accum[m][n] += regA[m] * regB[n]
 *
 *    每次从 Shared Memory 读 BN+BM 个值，完成 BN*BM 次 FMA
 *    算术强度（寄存器层）：BN*BM / (BN+BM) ≈ BN/2 = 4
 *
 *  参数约定（本版本）：
 *    BK = TILE = 16  （K 方向 tile 大小 / Block 内线程数）
 *    BN = BM  = 8    （每线程在 M/N 方向计算的元素数）
 *    Block 输出：BK*BN × BK*BM = 128 × 128
 *    Block 线程：BK × BK = 16 × 16 = 256
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// Block 内线程维度
#define BK   16   // Block 内线程数（BK×BK）
// 每线程计算的输出维度
#define BN   8    // 每线程在行方向（M）计算 BN 个元素
#define BM   8    // 每线程在列方向（N）计算 BM 个元素

// Block 覆盖的输出尺寸
#define BLOCK_ROW  (BK * BN)   // 128
#define BLOCK_COL  (BK * BM)   // 128
// Shared Memory tile 在 K 方向的大小（等于 BK）
#define TILE_K     BK          // 16

// ─────────────────────────────────────────────────────────────
//  Kernel
// ─────────────────────────────────────────────────────────────
__global__ void gemm_v2_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    int tx = threadIdx.x;   // [0, BK)  → 列方向
    int ty = threadIdx.y;   // [0, BK)  → 行方向

    // 当前线程负责输出 C 的起始坐标
    int row_start = blockIdx.y * BLOCK_ROW + ty * BN;  // M 方向
    int col_start = blockIdx.x * BLOCK_COL + tx * BM;  // N 方向

    // 累加寄存器：BN×BM 个输出元素
    float accum[BN][BM] = {};

    // Shared Memory：A 的 BLOCK_ROW×TILE_K 子块，B 的 TILE_K×BLOCK_COL 子块
    __shared__ float As[BLOCK_ROW][TILE_K];   // 128×16
    __shared__ float Bs[TILE_K][BLOCK_COL];   // 16×128

    // 每线程在加载阶段负责写入 As/Bs 的位置
    // BLOCK_ROW = BK*BN, BLOCK_COL = BK*BM
    // 总线程数 = BK*BK，每线程负责加载多个位置
    // As 总元素：BLOCK_ROW*TILE_K = 128*16 = 2048，线程数 256 → 每线程 8 个
    const int thread_id   = ty * BK + tx;    // 0..255
    const int as_per_thread = (BLOCK_ROW * TILE_K) / (BK * BK);  // 8
    const int bs_per_thread = (TILE_K * BLOCK_COL) / (BK * BK);  // 8

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // ── 协作加载 As（BLOCK_ROW×TILE_K = 128×16）──────────
        // 将 2048 个元素分配给 256 线程，每线程连续加载 8 个
        for (int i = 0; i < as_per_thread; i++) {
            int elem_idx = thread_id * as_per_thread + i;  // 全局元素编号
            int as_row   = elem_idx / TILE_K;
            int as_col   = elem_idx % TILE_K;
            int a_r = blockIdx.y * BLOCK_ROW + as_row;
            int a_c = t * TILE_K + as_col;
            As[as_row][as_col] = (a_r < M && a_c < K) ? A[a_r * K + a_c] : 0.0f;
        }

        // ── 协作加载 Bs（TILE_K×BLOCK_COL = 16×128）─────────
        for (int i = 0; i < bs_per_thread; i++) {
            int elem_idx = thread_id * bs_per_thread + i;
            int bs_row   = elem_idx / BLOCK_COL;
            int bs_col   = elem_idx % BLOCK_COL;
            int b_r = t * TILE_K + bs_row;
            int b_c = blockIdx.x * BLOCK_COL + bs_col;
            Bs[bs_row][bs_col] = (b_r < K && b_c < N) ? B[b_r * N + b_c] : 0.0f;
        }

        __syncthreads();

        // ── 寄存器级计算：BN×BM 个外积累加 ──────────────────
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // 加载 A 的一列片段到寄存器（BN 个值）
            float regA[BN];
            #pragma unroll
            for (int m = 0; m < BN; m++) {
                regA[m] = As[ty * BN + m][k];
            }
            // 加载 B 的一行片段到寄存器（BM 个值）
            float regB[BM];
            #pragma unroll
            for (int n = 0; n < BM; n++) {
                regB[n] = Bs[k][tx * BM + n];
            }
            // 外积：BN×BM 次 FMA
            #pragma unroll
            for (int m = 0; m < BN; m++) {
                #pragma unroll
                for (int n = 0; n < BM; n++) {
                    accum[m][n] += regA[m] * regB[n];
                }
            }
        }

        __syncthreads();
    }

    // ── 写回 C ───────────────────────────────────────────────
    #pragma unroll
    for (int m = 0; m < BN; m++) {
        #pragma unroll
        for (int n = 0; n < BM; n++) {
            int r = row_start + m;
            int c = col_start + n;
            if (r < M && c < N) {
                C[r * N + c] = alpha * accum[m][n] + beta * C[r * N + c];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  主程序
// ─────────────────────────────────────────────────────────────
int main()
{
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

    printf("=== GEMM V2: Thread Coarsening (BN=%d, BM=%d per thread) ===\n", BN, BM);
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block 覆盖: %d×%d,  每线程: %d×%d 元素\n\n",
           BLOCK_ROW, BLOCK_COL, BN, BM);

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

    dim3 block(BK, BK);
    dim3 grid((N + BLOCK_COL - 1) / BLOCK_COL,
              (M + BLOCK_ROW - 1) / BLOCK_ROW);

    for (int i = 0; i < 3; i++)
        gemm_v2_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        gemm_v2_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    bool correct = verify_gemm(h_C, h_C_ref, M, N);

    print_result("V2: Thread Coarsening", M, N, K, elapsed, correct);

    printf("\n相比 V1 的改进:\n");
    printf("  [+] 每线程计算 %d×%d=%d 个输出，寄存器数据复用率大幅提升\n",
           BN, BM, BN * BM);
    printf("  [+] Shared Memory 读取次数减少（BN+BM 次读取完成 BN*BM 次 FMA）\n");
    printf("  [+] Block 覆盖 %d×%d，更大的输出 tile\n", BLOCK_ROW, BLOCK_COL);
    printf("\n仍存在的问题:\n");
    printf("  [-] 全局内存加载仍为逐元素（1 float/次），未向量化\n");
    printf("\n下一步 → gemm_v3: float4 向量化加载，提升内存带宽利用率\n");

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
