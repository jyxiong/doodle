/*
 * ============================================================
 *  3DGS Tutorial  V4 —— CUDA Tile-based 光栅化
 * ============================================================
 *
 *  目标：将 V3 的 CPU 渲染器改造成 GPU 并行版本，
 *        复现 3DGS 论文中核心的三个 GPU Pass。
 *
 *  ────────────────────────────────────────────────────────
 *  1. 为什么需要 Tile-based 渲染？
 *  ────────────────────────────────────────────────────────
 *  V3 的 CPU 版本：O(pixels × gaussians) — 每像素遍历所有高斯
 *
 *  GPU 版本的关键优化：
 *    - 将屏幕划分为 TILE×TILE 的小块（本文件用 16×16）
 *    - 每个高斯只影响它 2D bbox 覆盖的 tile（局部性）
 *    - 每个 tile 对应一个 CUDA Block，16×16 线程各负责一个像素
 *    - 同一 Block 内的线程协作地批量加载当前 tile 的高斯数据
 *      到 shared memory，大幅减少全局内存访问
 *
 *  ────────────────────────────────────────────────────────
 *  2. 三个 GPU Pass
 *  ────────────────────────────────────────────────────────
 *    Pass 1 — Preprocess（N 个线程，每线程处理一个高斯）
 *      ├─ 3D → camera → 2D 投影 & 深度
 *      ├─ EWA 2D 协方差
 *      ├─ 计算 2D 包围盒 → tile 范围
 *      └─ 写出：(tile_key, depth) 对 —— 供 Pass 2 排序
 *
 *    Pass 2 — Radix Sort
 *      排序键 = tile_id * 一个大数 + 深度编码
 *      → 同一 tile 内的高斯按深度由近到远连续排列
 *
 *    Pass 3 — Forward Rendering（每 tile 一个 Block）
 *      ├─ 每次从全局内存批量加载 BLOCK_SIZE 个高斯到 shared memory
 *      ├─ 每个线程（像素）对 shared memory 里的高斯做 Alpha Blend
 *      └─ 写出最终 RGB
 *
 *  ────────────────────────────────────────────────────────
 *  3. 关键数据布局
 *  ────────────────────────────────────────────────────────
 *  preprocess 输出（SOA 格式，更好的内存合并访问）：
 *    d_mu2d[N]         : float2（屏幕中心）
 *    d_depth[N]        : float（相机空间 z）
 *    d_cov2d_inv[N][3] : float3（a,b,c 对称 2×2 逆矩阵）
 *    d_color[N]        : float3（RGB）
 *    d_opacity[N]      : float
 *    d_tile_bounds[N]  : int4（tile x/y 范围）
 *
 *  sort 后的键值对：
 *    d_keys[total_splats]    : uint64（高 32 位 tile_id，低 32 位深度编码）
 *    d_values[total_splats]  : uint32（原始高斯索引）
 *
 *  tile 信息：
 *    d_tile_start[num_tiles] : int（该 tile 在排序后数组中的起始位置）
 *    d_tile_end[num_tiles]   : int（结束位置）
 *
 * ============================================================
 */

#include "utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// ─────────────────────────────────────────────────────────────
//  常量
// ─────────────────────────────────────────────────────────────
#define TILE_SIZE  16        // 每个 Tile 的边长（像素）
#define BLOCK_SIZE 256       // Pass 3 中 shared memory 批次大小

// ─────────────────────────────────────────────────────────────
//  场景定义（与 V3 相同）
// ─────────────────────────────────────────────────────────────
struct GaussianData {
    float3 mu_world;
    Quat   quat;
    float3 scale;
    float  opacity;
    float3 color;
};

// ─────────────────────────────────────────────────────────────
//  Pass 1 Kernel：Preprocess
//
//  每个线程处理一个高斯，输出：
//    - 2D 中心 & 深度
//    - 2D 协方差逆（a, b, c）存为 float3
//    - 颜色 & 不透明度
//    - 覆盖的 tile 范围（int4: x_min, y_min, x_max, y_max）
//    - 排序键（uint64）和排序总数量（atomicAdd 到 d_num_splats）
// ─────────────────────────────────────────────────────────────

// 将深度 z∈[0.1, 100] 编码为整数（排序用）
__device__ inline uint32_t depth_to_key(float z)
{
    // z = 0.1 → 0,  z = 100 → UINT32_MAX
    float normalized = (z - 0.1f) / (100.f - 0.1f);
    normalized = fminf(fmaxf(normalized, 0.f), 1.f);
    return (uint32_t)(normalized * (float)0xFFFFFFFFu);
}

__global__ void preprocess_kernel(
    const float3* __restrict__ mu_world,
    const Quat*   __restrict__ quats,
    const float3* __restrict__ scales,
    const float*  __restrict__ opacities,
    const float3* __restrict__ colors,
    int N,
    // 相机参数（直接传入，避免结构体对齐问题）
    float fx, float fy, float cx, float cy, int W, int H,
    float Rcw[9], float3 tcw,
    // 输出
    float2*  d_mu2d,
    float*   d_depth,
    float3*  d_cov2d_inv,   // (a, b, c) of 2x2 inverse
    float3*  d_color_out,
    float*   d_opacity_out,
    int4*    d_tile_bounds,
    uint64_t* d_sort_keys,
    uint32_t* d_sort_values,
    int*     d_num_splats     // atomicAdd 计数器
)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // ── 变换到相机空间 ─────────────────────────────────────
    float3 pw = mu_world[gid];
    float3 pc = {
        Rcw[0]*pw.x + Rcw[1]*pw.y + Rcw[2]*pw.z + tcw.x,
        Rcw[3]*pw.x + Rcw[4]*pw.y + Rcw[5]*pw.z + tcw.y,
        Rcw[6]*pw.x + Rcw[7]*pw.y + Rcw[8]*pw.z + tcw.z
    };

    // 深度剔除
    if (pc.z < 0.1f) return;

    // ── 透视投影 ───────────────────────────────────────────
    float2 mu2d = {
        fx * (pc.x / pc.z) + cx,
        fy * (pc.y / pc.z) + cy
    };

    // ── EWA 2D 协方差 ──────────────────────────────────────
    Mat3 Rmat{};
    Rmat[0][0]=Rcw[0]; Rmat[0][1]=Rcw[1]; Rmat[0][2]=Rcw[2];
    Rmat[1][0]=Rcw[3]; Rmat[1][1]=Rcw[4]; Rmat[1][2]=Rcw[5];
    Rmat[2][0]=Rcw[6]; Rmat[2][1]=Rcw[7]; Rmat[2][2]=Rcw[8];

    // Jacobian
    float x = pc.x, y = pc.y, z = pc.z;
    Mat3 J{};
    J[0][0] =  fx/z;   J[0][2] = -fx*x/(z*z);
    J[1][1] =  fy/z;   J[1][2] = -fy*y/(z*z);

    Mat3 T = J * Rmat;

    Cov3d cov3d = build_cov3d(quats[gid], scales[gid]);
    Mat3  cov_mat = cov3d.to_mat3();
    Mat3  cov_full = T * cov_mat * transpose(T);

    // 取 2×2 + 低通滤波
    Mat2 cov2d = { cov_full[0][0]+0.3f, cov_full[0][1], cov_full[1][1]+0.3f };

    // 检查行列式（退化高斯）
    float det = cov2d.det();
    if (det < 1e-6f) return;

    Mat2 inv = cov2d.inv();

    // ── 2D 包围盒（3σ 半径）→ tile 范围 ───────────────────
    // 用协方差特征值估算椭圆半径（粗略：取迹的一半作为上界）
    float radius = ceilf(3.f * sqrtf(fmaxf(cov2d.a, cov2d.c)));

    int tx_min = (int)((mu2d.x - radius) / TILE_SIZE);
    int ty_min = (int)((mu2d.y - radius) / TILE_SIZE);
    int tx_max = (int)((mu2d.x + radius) / TILE_SIZE);
    int ty_max = (int)((mu2d.y + radius) / TILE_SIZE);

    int tiles_x = (W + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (H + TILE_SIZE - 1) / TILE_SIZE;

    tx_min = max(tx_min, 0);
    ty_min = max(ty_min, 0);
    tx_max = min(tx_max, tiles_x - 1);
    ty_max = min(ty_max, tiles_y - 1);

    // 屏幕外剔除
    if (tx_min > tx_max || ty_min > ty_max) return;

    // ── 写出预处理结果 ─────────────────────────────────────
    d_mu2d[gid]       = mu2d;
    d_depth[gid]      = pc.z;
    d_cov2d_inv[gid]  = {inv.a, inv.b, inv.c};
    d_color_out[gid]  = colors[gid];
    d_opacity_out[gid]= opacities[gid];
    d_tile_bounds[gid]= {tx_min, ty_min, tx_max, ty_max};

    // ── 为每个覆盖的 tile 生成一个排序键 ──────────────────
    for (int ty = ty_min; ty <= ty_max; ++ty) {
        for (int tx = tx_min; tx <= tx_max; ++tx) {
            int tile_id    = ty * tiles_x + tx;
            uint32_t dkey  = depth_to_key(pc.z);
            // 高 32 位：tile_id；低 32 位：深度（小 = 近）
            uint64_t key   = ((uint64_t)tile_id << 32) | dkey;

            int slot = atomicAdd(d_num_splats, 1);
            d_sort_keys[slot]   = key;
            d_sort_values[slot] = (uint32_t)gid;
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Pass 2 辅助 Kernel：标记每个 tile 在排序后数组中的起始/结束位置
// ─────────────────────────────────────────────────────────────
__global__ void identify_tile_ranges(
    const uint64_t* __restrict__ keys,
    int            total_splats,
    int*           d_tile_start,
    int*           d_tile_end,
    int            num_tiles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_splats) return;

    uint32_t tile_id = (uint32_t)(keys[i] >> 32);
    if (tile_id >= (uint32_t)num_tiles) return;

    // 当前是该 tile 的第一个 splat
    if (i == 0 || (uint32_t)(keys[i-1] >> 32) != tile_id)
        d_tile_start[tile_id] = i;

    // 当前是该 tile 的最后一个 splat
    if (i == total_splats-1 || (uint32_t)(keys[i+1] >> 32) != tile_id)
        d_tile_end[tile_id] = i + 1;
}

// ─────────────────────────────────────────────────────────────
//  Pass 3 Kernel：前向渲染
//
//  gridDim  = (tiles_x, tiles_y)
//  blockDim = (TILE_SIZE, TILE_SIZE)
//
//  每个 Block 负责一个 Tile，每个线程负责一个像素。
//  所有线程协作地以 BLOCK_SIZE 为批次，从全局内存加载高斯到
//  shared memory，然后每个像素对这批高斯做 Alpha Blend。
// ─────────────────────────────────────────────────────────────
__global__ void forward_render_kernel(
    const float2* __restrict__ d_mu2d,
    const float3* __restrict__ d_cov2d_inv,
    const float3* __restrict__ d_color,
    const float*  __restrict__ d_opacity,
    const uint32_t* __restrict__ d_sorted_indices,  // 排序后的高斯索引
    const int*    __restrict__ d_tile_start,
    const int*    __restrict__ d_tile_end,
    int W, int H,
    float3 bg_color,
    float* d_output   // W×H×3，行主序 RGB
)
{
    // Tile 坐标
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tiles_x = gridDim.x;
    int tile_id = tile_y * tiles_x + tile_x;

    // 像素坐标
    int px = tile_x * TILE_SIZE + threadIdx.x;
    int py = tile_y * TILE_SIZE + threadIdx.y;
    bool inside = (px < W && py < H);

    // 该 tile 的高斯范围
    int start = d_tile_start[tile_id];
    int end   = d_tile_end[tile_id];
    if (start == end) {
        // 空 tile，填背景色
        if (inside) {
            int idx = (py * W + px) * 3;
            d_output[idx+0] = bg_color.x;
            d_output[idx+1] = bg_color.y;
            d_output[idx+2] = bg_color.z;
        }
        return;
    }

    // ── Shared memory 批次加载 ──────────────────────────────
    // 布局：每个批次 BLOCK_SIZE 个高斯，每个高斯存 (mu2d, cov2d_inv, color, opacity)
    __shared__ float2 sh_mu2d    [BLOCK_SIZE];
    __shared__ float3 sh_cov_inv [BLOCK_SIZE];
    __shared__ float3 sh_color   [BLOCK_SIZE];
    __shared__ float  sh_opacity [BLOCK_SIZE];

    float T     = 1.f;
    float3 C    = {0.f, 0.f, 0.f};
    bool done   = false;   // 像素是否已不透明饱和

    int local_id = threadIdx.y * TILE_SIZE + threadIdx.x;   // 块内线性 id

    // ── 按批次遍历该 tile 的所有高斯 ───────────────────────
    for (int batch_start = start; batch_start < end; batch_start += BLOCK_SIZE) {
        // ─ 协作加载本批次数据到 shared memory ─────────────
        int load_count = min(BLOCK_SIZE, end - batch_start);
        if (local_id < load_count) {
            int gi = d_sorted_indices[batch_start + local_id];
            sh_mu2d   [local_id] = d_mu2d[gi];
            sh_cov_inv[local_id] = d_cov2d_inv[gi];
            sh_color  [local_id] = d_color[gi];
            sh_opacity[local_id] = d_opacity[gi];
        }
        __syncthreads();

        // ─ 每个像素对本批次高斯做 Alpha Blend ──────────────
        if (inside && !done) {
            for (int j = 0; j < load_count; ++j) {
                float2 delta = {(float)px - sh_mu2d[j].x,
                                (float)py - sh_mu2d[j].y};

                // 马氏距离平方（对称 2×2 二次型）
                float a = sh_cov_inv[j].x;
                float b = sh_cov_inv[j].y;
                float c = sh_cov_inv[j].z;
                float maha = a*delta.x*delta.x + 2.f*b*delta.x*delta.y + c*delta.y*delta.y;

                if (maha > 9.f) continue;   // 3σ 截断

                float gauss = expf(-0.5f * maha);
                float alpha = fminf(0.99f, sh_opacity[j] * gauss);

                if (alpha < 1.f / 255.f) continue;

                C = C + (T * alpha) * sh_color[j];
                T *= (1.f - alpha);

                if (T < 1e-4f) { done = true; break; }
            }
        }
        __syncthreads();
    }

    // ── 写出最终颜色 ────────────────────────────────────────
    if (inside) {
        float3 final_c = C + T * bg_color;
        int idx = (py * W + px) * 3;
        d_output[idx+0] = final_c.x;
        d_output[idx+1] = final_c.y;
        d_output[idx+2] = final_c.z;
    }
}

// ─────────────────────────────────────────────────────────────
//  场景构建（同 V3）
// ─────────────────────────────────────────────────────────────
std::vector<GaussianData> make_scene() {
    std::vector<GaussianData> gs;
    Quat q = {1.f, 0.f, 0.f, 0.f};
    gs.push_back({ {-1.6f,  0.f, 0.f}, q, {0.4f,0.4f,0.4f}, 0.95f, {0.95f, 0.20f, 0.20f} });
    gs.push_back({ {-0.8f,  0.f, 0.f}, q, {0.4f,0.4f,0.4f}, 0.95f, {0.95f, 0.60f, 0.10f} });
    gs.push_back({ { 0.0f,  0.f, 0.f}, q, {0.4f,0.4f,0.4f}, 0.95f, {0.90f, 0.90f, 0.10f} });
    gs.push_back({ { 0.8f,  0.f, 0.f}, q, {0.4f,0.4f,0.4f}, 0.95f, {0.20f, 0.85f, 0.20f} });
    gs.push_back({ { 1.6f,  0.f, 0.f}, q, {0.4f,0.4f,0.4f}, 0.95f, {0.20f, 0.40f, 0.95f} });
    gs.push_back({ {0.f, -0.7f, 0.5f}, q, {2.5f,0.05f,1.5f}, 0.6f, {0.8f, 0.8f, 0.8f} });
    gs.push_back({ {0.f,  0.5f,-0.5f}, q, {0.6f,0.3f,0.3f}, 0.7f, {0.8f, 0.2f, 0.8f} });
    gs.push_back({ {0.f,  0.5f, 0.3f}, q, {0.6f,0.3f,0.3f}, 0.7f, {0.2f, 0.8f, 0.8f} });
    return gs;
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────
int main()
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  3DGS Tutorial V4 — CUDA Tile-based 光栅化   ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    // ── 相机参数 ──────────────────────────────────────────────
    const int W = 256, H = 256;
    const float fx=200.f, fy=200.f, cx=W*0.5f, cy=H*0.5f;
    float Rcw_h[9] = {1,0,0, 0,1,0, 0,0,1};
    float3 tcw_h   = {0.f, 0.f, 3.f};
    float3 bg_color = {0.05f, 0.05f, 0.1f};

    const int tiles_x = (W + TILE_SIZE-1) / TILE_SIZE;
    const int tiles_y = (H + TILE_SIZE-1) / TILE_SIZE;
    const int num_tiles = tiles_x * tiles_y;

    // ── 场景数据从 CPU 拷贝到 GPU ────────────────────────────
    auto scene = make_scene();
    int N = (int)scene.size();
    printf("\n场景: %d 个高斯\n", N);

    std::vector<float3> h_mu(N), h_scale(N), h_color(N);
    std::vector<Quat>   h_quat(N);
    std::vector<float>  h_opac(N);
    for (int i = 0; i < N; ++i) {
        h_mu[i]    = scene[i].mu_world;
        h_quat[i]  = scene[i].quat;
        h_scale[i] = scene[i].scale;
        h_opac[i]  = scene[i].opacity;
        h_color[i] = scene[i].color;
    }

    float3 *d_mu, *d_scale, *d_color_in;
    Quat   *d_quat;
    float  *d_opac_in;
    CUDA_CHECK(cudaMalloc(&d_mu,       N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_quat,     N*sizeof(Quat)));
    CUDA_CHECK(cudaMalloc(&d_scale,    N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_opac_in,  N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_color_in, N*sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_mu,       h_mu.data(),    N*sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quat,     h_quat.data(),  N*sizeof(Quat),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale,    h_scale.data(), N*sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opac_in,  h_opac.data(),  N*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_color_in, h_color.data(), N*sizeof(float3), cudaMemcpyHostToDevice));

    // ── 预处理输出缓冲区 ──────────────────────────────────────
    float2 *d_mu2d;    float *d_depth; float3 *d_cov2d_inv;
    float3 *d_color_p; float *d_opac_p;
    int4   *d_tile_bounds;
    CUDA_CHECK(cudaMalloc(&d_mu2d,      N*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_depth,     N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cov2d_inv, N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_color_p,   N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_opac_p,    N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tile_bounds, N*sizeof(int4)));

    // (tile数 * 高斯数) 作为排序键值的上界
    int max_splats = N * num_tiles;
    uint64_t *d_sort_keys;    uint32_t *d_sort_vals;
    CUDA_CHECK(cudaMalloc(&d_sort_keys, max_splats*sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sort_vals, max_splats*sizeof(uint32_t)));

    int *d_num_splats;
    CUDA_CHECK(cudaMalloc(&d_num_splats, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_num_splats, 0, sizeof(int)));

    float *d_Rcw;
    CUDA_CHECK(cudaMalloc(&d_Rcw, 9*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Rcw, Rcw_h, 9*sizeof(float), cudaMemcpyHostToDevice));

    // ────────────────────────────────────────────────────────
    //  Pass 1 — Preprocess
    // ────────────────────────────────────────────────────────
    CpuTimer timer; timer.start();
    {
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        preprocess_kernel<<<grid, block>>>(
            d_mu, d_quat, d_scale, d_opac_in, d_color_in, N,
            fx, fy, cx, cy, W, H, d_Rcw, tcw_h,
            d_mu2d, d_depth, d_cov2d_inv, d_color_p, d_opac_p,
            d_tile_bounds, d_sort_keys, d_sort_vals, d_num_splats
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    int num_splats;
    CUDA_CHECK(cudaMemcpy(&num_splats, d_num_splats, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Pass 1 Preprocess: %d splats (%.2f ms)\n", num_splats, timer.elapsed_ms());

    // ────────────────────────────────────────────────────────
    //  Pass 2 — Radix Sort（使用 Thrust）
    // ────────────────────────────────────────────────────────
    timer.start();
    {
        thrust::device_ptr<uint64_t> key_ptr(d_sort_keys);
        thrust::device_ptr<uint32_t> val_ptr(d_sort_vals);
        thrust::sort_by_key(
            thrust::device,
            key_ptr, key_ptr + num_splats,
            val_ptr
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("Pass 2 Sort:       %d splats sorted (%.2f ms)\n", num_splats, timer.elapsed_ms());

    // 找出每个 tile 的起始/结束
    int *d_tile_start, *d_tile_end;
    CUDA_CHECK(cudaMalloc(&d_tile_start, num_tiles*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tile_end,   num_tiles*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tile_start, 0, num_tiles*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tile_end,   0, num_tiles*sizeof(int)));
    {
        dim3 block(256);
        dim3 grid((num_splats + 255) / 256);
        if (num_splats > 0)
            identify_tile_ranges<<<grid, block>>>(
                d_sort_keys, num_splats, d_tile_start, d_tile_end, num_tiles
            );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ────────────────────────────────────────────────────────
    //  Pass 3 — Forward Rendering
    // ────────────────────────────────────────────────────────
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, W*H*3*sizeof(float)));

    timer.start();
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(tiles_x, tiles_y);
        forward_render_kernel<<<grid, block>>>(
            d_mu2d, d_cov2d_inv, d_color_p, d_opac_p,
            d_sort_vals,
            d_tile_start, d_tile_end,
            W, H, bg_color,
            d_output
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("Pass 3 Render:     %dx%d image (%.2f ms)\n", W, H, timer.elapsed_ms());

    // ── 拷贝结果到 CPU 并写出 PPM ────────────────────────────
    std::vector<float> h_output(W * H * 3);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          W*H*3*sizeof(float), cudaMemcpyDeviceToHost));
    save_ppm("3dgs_v4_output.ppm", h_output.data(), W, H);

    // ── 释放资源 ──────────────────────────────────────────────
    cudaFree(d_mu); cudaFree(d_quat); cudaFree(d_scale);
    cudaFree(d_opac_in); cudaFree(d_color_in);
    cudaFree(d_mu2d); cudaFree(d_depth); cudaFree(d_cov2d_inv);
    cudaFree(d_color_p); cudaFree(d_opac_p); cudaFree(d_tile_bounds);
    cudaFree(d_sort_keys); cudaFree(d_sort_vals); cudaFree(d_num_splats);
    cudaFree(d_Rcw); cudaFree(d_tile_start); cudaFree(d_tile_end);
    cudaFree(d_output);

    printf("\n[完成] 打开 3dgs_v4_output.ppm，结果应与 V3 基本一致\n");
    printf("\n下一步：\n");
    printf("  - 将更多高斯（~100k）加入场景，观察性能提升\n");
    printf("  - 添加反向传播 Kernel（backward pass），实现可微分渲染\n");
    printf("  - 实现 Densification 和 Pruning（自适应高斯球密度控制）\n");
    return 0;
}
