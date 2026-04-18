/*
 * ============================================================
 *  GEMM  V4 —— Register Tiling（更大寄存器块 + Warp 级优化）
 * ============================================================
 *
 *  V3 的瓶颈：
 *    每个 Warp（32 线程）计算的输出区域有限，
 *    Shared Memory 到寄存器的加载存在冗余（多线程读同一地址）。
 *
 *  本版本采用 Warp-level Tiling：
 *    将 Block 的输出区域先按 Warp 划分，再按线程划分。
 *    每个 Warp 负责一个 WARP_ROW × WARP_COL 的输出子矩阵。
 *    Warp 内 32 线程再各自负责 TM × TN 个元素。
 *
 *  参数设计：
 *    Block:     128 × 128 输出
 *    Warp 数量：4 × 4 = 16 Warps per Block
 *    每 Warp:   32 × 32 输出（WARP_ROW=32, WARP_COL=32）
 *    Warp 维度：8×4 线程（Lane 排布）
 *    每 Lane:   4 × 8 = 32 输出元素（TM=4, TN=8）
 *
 *    Block = 16 Warps × 32 threads = 512 threads
 *
 *  外积累加（Outer Product）：
 *    加载一次 As 列片段（16 个 float → 寄存器 regA[16]）
 *    加载一次 Bs 行片段（16 个 float → 寄存器 regB[16]）
 *    外积：16×16 = 256 次 FMA / k 步
 *    算术强度（寄存器层）：256 / (16+16) = 8 FLOP/Byte
 *
 *  Warp 内线程排布（8列×4行）：
 *    lane_col = lane_id % 8    （0..7）
 *    lane_row = lane_id / 8    （0..3）
 *    每 lane 负责 regA[4]（连续行）和 regB[8]（连续列）
 *
 * ============================================================
 */

#include "utils.cuh"
#include <stdio.h>

// ── 宏定义 ───────────────────────────────────────────────
// Tile K 大小（Shared Memory 在 K 方向的宽度）
#define BK    16

// Block 输出尺寸
#define BLOCK_M   128
#define BLOCK_N   128

// Warp 输出尺寸
#define WARP_M    32
#define WARP_N    32

// 每线程输出尺寸
#define TM    4
#define TN    8

// Block 内 Warp 数量
#define WARPS_M   (BLOCK_M / WARP_M)   // 4
#define WARPS_N   (BLOCK_N / WARP_N)   // 4
#define WARPS_PER_BLOCK   (WARPS_M * WARPS_N)   // 16

// 每 Warp 内线程布局：(WARP_N/TN) 列 × (WARP_M/TM) 行 = 4×8 = 32
#define LANES_N   (WARP_N / TN)   // 4
#define LANES_M   (WARP_M / TM)   // 8

// Shared Memory padding
#define AS_PAD  4
#define BS_PAD  4

// ─────────────────────────────────────────────────────────────
//  Kernel
// ─────────────────────────────────────────────────────────────
__global__ void gemm_v4_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    // Warp 和 Lane 的 ID
    int warp_id  = threadIdx.x / 32;              // 0..15
    int lane_id  = threadIdx.x % 32;              // 0..31

    // Warp 在 Block 内的行列坐标
    int warp_row = warp_id / WARPS_N;             // 0..3
    int warp_col = warp_id % WARPS_N;             // 0..3

    // Lane 在 Warp 内的行列坐标（8列×4行）
    int lane_col = lane_id % LANES_N;             // 0..3
    int lane_row = lane_id / LANES_N;             // 0..7

    // 当前线程输出元素的起始全局坐标
    int row_start = blockIdx.y * BLOCK_M
                  + warp_row * WARP_M
                  + lane_row * TM;
    int col_start = blockIdx.x * BLOCK_N
                  + warp_col * WARP_N
                  + lane_col * TN;

    // 寄存器累加数组
    float accum[TM][TN] = {};

    // Shared Memory
    __shared__ float As[BLOCK_M][BK + AS_PAD];     // 128×20
    __shared__ float Bs[BK][BLOCK_N + BS_PAD];     // 16×132

    // 加载参数（全展开为 1D 线程）
    const int THREADS   = WARPS_PER_BLOCK * 32;    // 512
    const int thread_id = threadIdx.x;

    // As 总 float4：128*16/4=512，每线程 512/512=1
    // Bs 总 float4：16*128/4=512，每线程 1
    for (int t = 0; t < (K + BK - 1) / BK; t++) {

        // ── 加载 As（BLOCK_M × BK = 128×16）──────────────
        // float4 加载，行方向：As[row][col*4..(col+1)*4)
        {
            int f4_total = BLOCK_M * BK / 4;  // 512
            int f4_per_t = (f4_total + THREADS - 1) / THREADS;
            for (int i = 0; i < f4_per_t; i++) {
                int f4_idx = thread_id * f4_per_t + i;
                if (f4_idx >= f4_total) break;
                int as_row  = f4_idx / (BK / 4);
                int as_col4 = f4_idx % (BK / 4);
                int a_r = blockIdx.y * BLOCK_M + as_row;
                int a_c = t * BK + as_col4 * 4;
                float4 val = {0.f, 0.f, 0.f, 0.f};
                if (a_r < M && a_c + 3 < K) {
                    val = *reinterpret_cast<const float4*>(&A[a_r * K + a_c]);
                } else if (a_r < M) {
                    val.x = (a_c     < K) ? A[a_r*K+a_c  ] : 0.f;
                    val.y = (a_c+1   < K) ? A[a_r*K+a_c+1] : 0.f;
                    val.z = (a_c+2   < K) ? A[a_r*K+a_c+2] : 0.f;
                    val.w = (a_c+3   < K) ? A[a_r*K+a_c+3] : 0.f;
                }
                As[as_row][as_col4*4+0] = val.x;
                As[as_row][as_col4*4+1] = val.y;
                As[as_row][as_col4*4+2] = val.z;
                As[as_row][as_col4*4+3] = val.w;
            }
        }

        // ── 加载 Bs（BK × BLOCK_N = 16×128）──────────────
        {
            int f4_total = BK * BLOCK_N / 4;  // 512
            int f4_per_t = (f4_total + THREADS - 1) / THREADS;
            for (int i = 0; i < f4_per_t; i++) {
                int f4_idx = thread_id * f4_per_t + i;
                if (f4_idx >= f4_total) break;
                int bs_row  = f4_idx / (BLOCK_N / 4);
                int bs_col4 = f4_idx % (BLOCK_N / 4);
                int b_r = t * BK + bs_row;
                int b_c = blockIdx.x * BLOCK_N + bs_col4 * 4;
                float4 val = {0.f, 0.f, 0.f, 0.f};
                if (b_r < K && b_c + 3 < N) {
                    val = *reinterpret_cast<const float4*>(&B[b_r * N + b_c]);
                } else if (b_r < K) {
                    val.x = (b_c     < N) ? B[b_r*N+b_c  ] : 0.f;
                    val.y = (b_c+1   < N) ? B[b_r*N+b_c+1] : 0.f;
                    val.z = (b_c+2   < N) ? B[b_r*N+b_c+2] : 0.f;
                    val.w = (b_c+3   < N) ? B[b_r*N+b_c+3] : 0.f;
                }
                Bs[bs_row][bs_col4*4+0] = val.x;
                Bs[bs_row][bs_col4*4+1] = val.y;
                Bs[bs_row][bs_col4*4+2] = val.z;
                Bs[bs_row][bs_col4*4+3] = val.w;
            }
        }

        __syncthreads();

        // ── 寄存器计算（外积累加）──────────────────────────
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[TM], regB[TN];

            // 加载 A 列片段到寄存器
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[warp_row * WARP_M + lane_row * TM + m][k];
            }
            // 加载 B 行片段到寄存器
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][warp_col * WARP_N + lane_col * TN + n];
            }
            // 外积 FMA
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    accum[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // ── 写回 C（float4 向量化，TN=8 可打两次 float4）────────
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int r = row_start + m;
        if (r >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int c = col_start + n;
            if (c + 3 < N) {
                float4 old = *reinterpret_cast<float4*>(&C[r * N + c]);
                float4 out;
                out.x = alpha * accum[m][n]     + beta * old.x;
                out.y = alpha * accum[m][n + 1] + beta * old.y;
                out.z = alpha * accum[m][n + 2] + beta * old.z;
                out.w = alpha * accum[m][n + 3] + beta * old.w;
                *reinterpret_cast<float4*>(&C[r * N + c]) = out;
            } else {
                for (int nn = n; nn < TN && (col_start + nn) < N; nn++) {
                    C[r * N + col_start + nn] =
                        alpha * accum[m][nn] + beta * C[r * N + col_start + nn];
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

    printf("=== GEMM V4: Register Tiling + Warp-level Tiling ===\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block: %d×%d, Warps: %d×%d, Per-lane: %d×%d\n\n",
           BLOCK_M, BLOCK_N, WARPS_M, WARPS_N, TM, TN);

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

    // 每 Block WARPS_PER_BLOCK * 32 = 512 线程（1D）
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    for (int i = 0; i < 3; i++)
        gemm_v4_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        gemm_v4_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    bool correct = verify_gemm(h_C, h_C_ref, M, N);

    print_result("V4: Register Tiling + Warp Tiling", M, N, K, elapsed, correct);

    printf("\n相比 V3 的改进:\n");
    printf("  [+] Warp 级 Tiling：更好的 Shared Memory 局部性\n");
    printf("  [+] 每线程 %d×%d=%d 输出，寄存器外积复用率更高\n", TM, TN, TM*TN);
    printf("  [+] 外积 FMA：每读 %d+%d=%d 个值完成 %d 次 FMA\n",
           TM, TN, TM+TN, TM*TN);
    printf("\n下一步 → gemm_v5: Double Buffering，隐藏 Shared Memory 加载延迟\n");

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
