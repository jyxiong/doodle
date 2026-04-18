/*
 * ============================================================
 *  GEMM  V5 —— Double Buffering（隐藏 Shared Memory 加载延迟）
 * ============================================================
 *
 *  V4 的执行时间线（每个 tile 步骤）：
 *
 *    [Load As/Bs from global → Shared]  ← L2/DRAM 延迟 200~400 周期
 *    __syncthreads()
 *    [Compute: register outer product]
 *    __syncthreads()
 *
 *    计算和加载串行执行，计算单元在加载阶段完全空闲。
 *
 *  Double Buffering 思想（软件流水线）：
 *    分配两倍的 Shared Memory（As[2][...], Bs[2][...]）。
 *    在计算第 t 步的同时，预加载第 t+1 步的数据到另一个缓冲区。
 *
 *  执行时间线（Double Buffer）：
 *
 *    预加载 t=0 → smem[0]
 *    ─────────────────────────────────────
 *    t=0: 计算 smem[0]  ||  预加载 t=1 → smem[1]
 *    t=1: 计算 smem[1]  ||  预加载 t=2 → smem[0]
 *    t=2: 计算 smem[0]  ||  预加载 t=3 → smem[1]
 *    ...
 *    t=T-1: 计算 smem[T-1 % 2]（无预加载）
 *
 *    加载和计算重叠 → 隐藏全局内存延迟！
 *
 *  实现关键点：
 *    1. cp.async（sm_80+ / Ampere+）：
 *       异步内存拷贝，不阻塞 SM 继续执行计算指令。
 *       用 `cuda::memcpy_async` 或 `cp.async` PTX 内联。
 *    2. __pipeline_commit() + __pipeline_wait_prior(0)：
 *       提交本批次异步拷贝，等待前一批次完成（保留1批次在途）。
 *    3. 缓冲区轮换：write_buf / read_buf 交替 0/1。
 *
 *  注意：
 *    cp.async 需要 sm_80+（Ampere, A100, RTX 30xx 等）。
 *    对于 sm_75 及以下，用 ldg.cs + __syncthreads() 退化实现。
 *
 *  本版本用 `__pipeline` 接口（cuda/pipeline.h，CUDA 11.1+）：
 *    cuda::pipeline<cuda::thread_scope_block> pipe;
 *    cuda::memcpy_async(dst, src, size, pipe);
 *    pipe.commit();
 *    pipe.wait_prior<1>();  // 等待除最新1批外的所有批次完成
 *
 * ============================================================
 */

#include "utils.cuh"
#include <cuda/barrier>
#include <stdio.h>

// ── 宏定义（与 V4 相同）──────────────────────────────────
#define BK    16
#define BLOCK_M   128
#define BLOCK_N   128
#define WARP_M    32
#define WARP_N    32
#define TM    4
#define TN    8
#define WARPS_M   (BLOCK_M / WARP_M)
#define WARPS_N   (BLOCK_N / WARP_N)
#define WARPS_PER_BLOCK   (WARPS_M * WARPS_N)
#define LANES_N   (WARP_N / TN)
#define LANES_M   (WARP_M / TM)

#define AS_PAD 4
#define BS_PAD 4

// ─────────────────────────────────────────────────────────────
//  Kernel
// ─────────────────────────────────────────────────────────────
__global__ void gemm_v5_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float*       __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    // Warp 和 Lane 坐标
    int warp_id  = threadIdx.x / 32;
    int lane_id  = threadIdx.x % 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int lane_col = lane_id % LANES_N;
    int lane_row = lane_id / LANES_N;

    int row_start = blockIdx.y * BLOCK_M
                  + warp_row * WARP_M + lane_row * TM;
    int col_start = blockIdx.x * BLOCK_N
                  + warp_col * WARP_N + lane_col * TN;

    float accum[TM][TN] = {};

    // Double Buffer Shared Memory
    __shared__ float As[2][BLOCK_M][BK + AS_PAD];
    __shared__ float Bs[2][BK][BLOCK_N + BS_PAD];

    // Block-scope barrier（每个 buffer 一个）
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];
    if (threadIdx.x == 0) {
        init(&bar[0], WARPS_PER_BLOCK * 32);
        init(&bar[1], WARPS_PER_BLOCK * 32);
    }
    __syncthreads();

    const int THREADS   = WARPS_PER_BLOCK * 32;
    const int thread_id = threadIdx.x;
    const int num_tiles = (K + BK - 1) / BK;

    // ── 辅助 Lambda：异步加载单个 tile 到指定缓冲区 ──────
    // 使用 __pipeline_memcpy_async（PTX cp.async，sm_80+）
    auto async_load = [&](int tile_idx, int buf) {
        if (tile_idx >= num_tiles) return;
        // 加载 As
        {
            int f4_total = BLOCK_M * BK / 4;
            int f4_per_t = (f4_total + THREADS - 1) / THREADS;
            for (int i = 0; i < f4_per_t; i++) {
                int f4_idx  = thread_id * f4_per_t + i;
                if (f4_idx >= f4_total) break;
                int as_row  = f4_idx / (BK / 4);
                int as_col4 = f4_idx % (BK / 4);
                int a_r = blockIdx.y * BLOCK_M + as_row;
                int a_c = tile_idx * BK + as_col4 * 4;
                if (a_r < M && a_c + 3 < K) {
                    cuda::memcpy_async(
                        &As[buf][as_row][as_col4 * 4],
                        &A[a_r * K + a_c],
                        cuda::aligned_size_t<16>(sizeof(float4)),
                        bar[buf]);
                } else {
                    // 边界：同步写（fallback）
                    As[buf][as_row][as_col4*4+0] = (a_r<M && a_c  <K) ? A[a_r*K+a_c  ] : 0.f;
                    As[buf][as_row][as_col4*4+1] = (a_r<M && a_c+1<K) ? A[a_r*K+a_c+1] : 0.f;
                    As[buf][as_row][as_col4*4+2] = (a_r<M && a_c+2<K) ? A[a_r*K+a_c+2] : 0.f;
                    As[buf][as_row][as_col4*4+3] = (a_r<M && a_c+3<K) ? A[a_r*K+a_c+3] : 0.f;
                }
            }
        }
        // 加载 Bs
        {
            int f4_total = BK * BLOCK_N / 4;
            int f4_per_t = (f4_total + THREADS - 1) / THREADS;
            for (int i = 0; i < f4_per_t; i++) {
                int f4_idx  = thread_id * f4_per_t + i;
                if (f4_idx >= f4_total) break;
                int bs_row  = f4_idx / (BLOCK_N / 4);
                int bs_col4 = f4_idx % (BLOCK_N / 4);
                int b_r = tile_idx * BK + bs_row;
                int b_c = blockIdx.x * BLOCK_N + bs_col4 * 4;
                if (b_r < K && b_c + 3 < N) {
                    cuda::memcpy_async(
                        &Bs[buf][bs_row][bs_col4 * 4],
                        &B[b_r * N + b_c],
                        cuda::aligned_size_t<16>(sizeof(float4)),
                        bar[buf]);
                } else {
                    Bs[buf][bs_row][bs_col4*4+0] = (b_r<K && b_c  <N) ? B[b_r*N+b_c  ] : 0.f;
                    Bs[buf][bs_row][bs_col4*4+1] = (b_r<K && b_c+1<N) ? B[b_r*N+b_c+1] : 0.f;
                    Bs[buf][bs_row][bs_col4*4+2] = (b_r<K && b_c+2<N) ? B[b_r*N+b_c+2] : 0.f;
                    Bs[buf][bs_row][bs_col4*4+3] = (b_r<K && b_c+3<N) ? B[b_r*N+b_c+3] : 0.f;
                }
            }
        }
    };

    // ── 预加载第一个 tile ─────────────────────────────────
    async_load(0, 0);
    bar[0].arrive_and_wait();  // 等待 buf=0 加载完毕

    // ── 主循环：计算 t 的同时预加载 t+1 ─────────────────
    for (int t = 0; t < num_tiles; t++) {
        int read_buf  = t & 1;
        int write_buf = 1 - read_buf;

        // 预加载下一个 tile 到 write_buf
        async_load(t + 1, write_buf);

        // ── 寄存器计算（与 V4 完全相同）──────────────────
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[TM], regB[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                regA[m] = As[read_buf][warp_row*WARP_M + lane_row*TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                regB[n] = Bs[read_buf][k][warp_col*WARP_N + lane_col*TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    accum[m][n] += regA[m] * regB[n];
        }

        // 等待下一个 tile 加载完毕（若存在）
        if (t + 1 < num_tiles) {
            bar[write_buf].arrive_and_wait();
        }
    }

    // ── 写回 C ───────────────────────────────────────────
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
                out.x = alpha * accum[m][n]   + beta * old.x;
                out.y = alpha * accum[m][n+1] + beta * old.y;
                out.z = alpha * accum[m][n+2] + beta * old.z;
                out.w = alpha * accum[m][n+3] + beta * old.w;
                *reinterpret_cast<float4*>(&C[r * N + c]) = out;
            } else {
                for (int nn = n; nn < TN && (col_start + nn) < N; nn++) {
                    C[r*N+col_start+nn] =
                        alpha * accum[m][nn] + beta * C[r*N+col_start+nn];
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
    // 检查 sm_80+ 是否可用
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== GEMM V5: Double Buffering (cp.async / pipeline) ===\n");
    printf("Device: %s  (sm_%d%d)\n\n",
           prop.name, prop.major, prop.minor);
    if (prop.major < 8) {
        printf("[警告] sm_%d%d < sm_80，cp.async 可能退化为同步加载，\n"
               "       Double Buffering 效果受限。\n\n",
               prop.major, prop.minor);
    }

    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

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

    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    for (int i = 0; i < 3; i++)
        gemm_v5_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int RUNS = 20;
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < RUNS; i++)
        gemm_v5_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    timer.stop();
    float elapsed = timer.elapsed_ms() / RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    bool correct = verify_gemm(h_C, h_C_ref, M, N);

    print_result("V5: Double Buffering", M, N, K, elapsed, correct);

    printf("\n相比 V4 的改进:\n");
    printf("  [+] Double Buffer：计算与加载流水并行，隐藏全局内存延迟\n");
    printf("  [+] cp.async（sm_80+）：异步加载不阻塞 SM 的计算流\n");
    printf("  [+] Shared Memory 消耗翻倍（As/Bs 各×2），换取延迟隐藏\n");
    printf("\n完整优化路径总结:\n");
    printf("  V0: 朴素  → V1: Shared Mem Tiling  → V2: Thread Coarsening\n");
    printf("  V3: float4  → V4: Register/Warp Tiling  → V5: Double Buffering\n");

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
