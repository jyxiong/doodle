/*
 * ============================================================
 *  GEMM  V3 —— float4 向量化加载（提升内存带宽利用率）
 * ============================================================
 *
 *  V2 的问题：
 *    全局内存加载仍以 1 float（4 Bytes）为单位，
 *    GPU 内存控制器每次能够传输 128 Bytes（32 float），
 *    单次 float 读取等于浪费了 31/32 的带宽配额。
 *
 *  float4 加载思想：
 *    使用 `float4`（16 Bytes）一次抓取 4 个连续浮点数。
 *    同等数量的内存事务获取 4× 的数据量。
 *    在满足 16 Byte 对齐（地址 % 16 == 0）的情况下，
 *    编译器生成 LDG.E.128 指令，充分利用 L2/HBM 带宽。
 *
 *  向量化加载条件：
 *    1. 数据地址必须 16 Byte 对齐（malloc/cudaMalloc 天然满足）
 *    2. 连续线程访问连续地址（已满足：行优先存储，tx 方向连续）
 *    3. 加载宽度是 4 的倍数（约束 K 和 N 为 4 的倍数，BLOCK_COL 为 4 的倍数）
 *
 *  加载方式对比：
 *    V2：As[as_row][as_col] = A[a_r * K + a_c]   ← 1 float / 指令
 *    V3：*(float4*)&As[as_row][as_col] = *(float4*)&A[a_r*K+a_c]  ← 4 float / 指令
 *
 *  Shared Memory 填充（Padding）：
 *    V2 中 As[BLOCK_ROW][TILE_K] 的 TILE_K=16，
 *    连续线程访问 As[row0..row3][col] 可能遇到 Bank Conflict。
 *    给 As 的第二维加 1 padding（As[BLOCK_ROW][TILE_K+4]），
 *    错开 Bank 映射，消除冲突（float4 步长 4，加 4 刚好错开）。
 *
 *  参数（与 V2 相同）：
 *    BK=16, BN=8, BM=8
 *    Block: 16×16=256 线程，每线程 8×8=64 输出元素
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

#define BK   16
#define BN   8
#define BM   8

#define BLOCK_ROW  (BK * BN)   // 128
#define BLOCK_COL  (BK * BM)   // 128
#define TILE_K     BK          // 16

// Padding 避免 Bank Conflict（float4 方向）
#define AS_PAD  4              // As 第二维 padding
#define BS_PAD  4              // Bs 第二维 padding

// ─────────────────────────────────────────────────────────────
//  Kernel
// ─────────────────────────────────────────────────────────────
__global__ void gemm_v3_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = blockIdx.y * BLOCK_ROW + ty * BN;
    int col_start = blockIdx.x * BLOCK_COL + tx * BM;

    float accum[BN][BM] = {};

    // 添加 padding 的 Shared Memory
    __shared__ float As[BLOCK_ROW][TILE_K + AS_PAD];   // 128×20
    __shared__ float Bs[TILE_K][BLOCK_COL + BS_PAD];   // 16×132

    const int thread_id = ty * BK + tx;   // 0..255

    // as_per_thread / bs_per_thread 用 float4 单位计算
    // As 总 float4 个数：BLOCK_ROW * TILE_K / 4 = 128*16/4 = 512
    // 每线程负责 512/256 = 2 个 float4 = 8 个 float
    const int AS_FLOAT4_TOTAL = BLOCK_ROW * TILE_K / 4;   // 512
    const int BS_FLOAT4_TOTAL = TILE_K * BLOCK_COL / 4;   // 512
    const int as_f4_per_thread = AS_FLOAT4_TOTAL / (BK * BK);  // 2
    const int bs_f4_per_thread = BS_FLOAT4_TOTAL / (BK * BK);  // 2

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // ── 向量化加载 As ────────────────────────────────────
        // 每次用 float4 读取 A 的 4 个列方向元素
        for (int i = 0; i < as_f4_per_thread; i++) {
            int f4_idx  = thread_id * as_f4_per_thread + i;
            // As 布局：BLOCK_ROW 行，TILE_K 列；float4 沿列方向打包
            // 每 4 column 为一个 float4，共 TILE_K/4 = 4 个 float4 每行
            int as_row  = f4_idx / (TILE_K / 4);
            int as_col4 = f4_idx % (TILE_K / 4);  // float4 列索引
            int a_r = blockIdx.y * BLOCK_ROW + as_row;
            int a_c = t * TILE_K + as_col4 * 4;

            float4 val = {0.0f, 0.0f, 0.0f, 0.0f};
            if (a_r < M && a_c + 3 < K) {
                val = *reinterpret_cast<const float4*>(&A[a_r * K + a_c]);
            } else if (a_r < M) {
                // 边界处理：逐个加载
                val.x = (a_c     < K) ? A[a_r * K + a_c    ] : 0.0f;
                val.y = (a_c + 1 < K) ? A[a_r * K + a_c + 1] : 0.0f;
                val.z = (a_c + 2 < K) ? A[a_r * K + a_c + 2] : 0.0f;
                val.w = (a_c + 3 < K) ? A[a_r * K + a_c + 3] : 0.0f;
            }
            As[as_row][as_col4 * 4 + 0] = val.x;
            As[as_row][as_col4 * 4 + 1] = val.y;
            As[as_row][as_col4 * 4 + 2] = val.z;
            As[as_row][as_col4 * 4 + 3] = val.w;
        }

        // ── 向量化加载 Bs ────────────────────────────────────
        // float4 沿行方向（BLOCK_COL）打包，每 4 列一个 float4
        for (int i = 0; i < bs_f4_per_thread; i++) {
            int f4_idx  = thread_id * bs_f4_per_thread + i;
            // Bs 布局：TILE_K 行，BLOCK_COL 列
            int bs_row  = f4_idx / (BLOCK_COL / 4);
            int bs_col4 = f4_idx % (BLOCK_COL / 4);
            int b_r = t * TILE_K + bs_row;
            int b_c = blockIdx.x * BLOCK_COL + bs_col4 * 4;

            float4 val = {0.0f, 0.0f, 0.0f, 0.0f};
            if (b_r < K && b_c + 3 < N) {
                val = *reinterpret_cast<const float4*>(&B[b_r * N + b_c]);
            } else if (b_r < K) {
                val.x = (b_c     < N) ? B[b_r * N + b_c    ] : 0.0f;
                val.y = (b_c + 1 < N) ? B[b_r * N + b_c + 1] : 0.0f;
                val.z = (b_c + 2 < N) ? B[b_r * N + b_c + 2] : 0.0f;
                val.w = (b_c + 3 < N) ? B[b_r * N + b_c + 3] : 0.0f;
            }
            Bs[bs_row][bs_col4 * 4 + 0] = val.x;
            Bs[bs_row][bs_col4 * 4 + 1] = val.y;
            Bs[bs_row][bs_col4 * 4 + 2] = val.z;
            Bs[bs_row][bs_col4 * 4 + 3] = val.w;
        }

        __syncthreads();

        // ── 寄存器级计算（同 V2）──────────────────────────
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float regA[BN];
            #pragma unroll
            for (int m = 0; m < BN; m++) regA[m] = As[ty * BN + m][k];

            float regB[BM];
            #pragma unroll
            for (int n = 0; n < BM; n++) regB[n] = Bs[k][tx * BM + n];

            #pragma unroll
            for (int m = 0; m < BN; m++)
                #pragma unroll
                for (int n = 0; n < BM; n++)
                    accum[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // ── 写回 C（float4 向量化写）──────────────────────────────
    #pragma unroll
    for (int m = 0; m < BN; m++) {
        int r = row_start + m;
        if (r >= M) continue;

        // 每 4 个 n 打包成 float4 写回
        #pragma unroll
        for (int n = 0; n < BM; n += 4) {
            int c = col_start + n;
            if (c + 3 < N) {
                float4 out;
                out.x = alpha * accum[m][n]     + beta * C[r * N + c    ];
                out.y = alpha * accum[m][n + 1] + beta * C[r * N + c + 1];
                out.z = alpha * accum[m][n + 2] + beta * C[r * N + c + 2];
                out.w = alpha * accum[m][n + 3] + beta * C[r * N + c + 3];
                *reinterpret_cast<float4*>(&C[r * N + c]) = out;
            } else {
                for (int nn = n; nn < BM && (col_start + nn) < N; nn++) {
                    C[r * N + col_start + nn] = alpha * accum[m][nn]
                                              + beta  * C[r * N + col_start + nn];
                }
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

    printf("=== GEMM V3: float4 向量化加载 + Shared Memory Padding ===\n");
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

    dim3 block(BK, BK);
    dim3 grid((N + BLOCK_COL - 1) / BLOCK_COL,
              (M + BLOCK_ROW - 1) / BLOCK_ROW);

    for (int i = 0; i < 3; i++)
        gemm_v3_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        gemm_v3_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    bool correct = verify_gemm(h_C, h_C_ref, M, N);

    print_result("V3: float4 向量化 + Padding", M, N, K, elapsed, correct);

    printf("\n相比 V2 的改进:\n");
    printf("  [+] float4 加载：每次内存事务传输 16 Bytes，减少指令数 4×\n");
    printf("  [+] LDG.E.128 指令充分利用 L2/HBM 带宽\n");
    printf("  [+] Shared Memory Padding 消除 Bank Conflict\n");
    printf("\n下一步 → gemm_v4: Register Tiling（外积累加 + 更大寄存器块）\n");

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
