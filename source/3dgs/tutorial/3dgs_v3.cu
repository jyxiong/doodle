/*
 * ============================================================
 *  3DGS Tutorial  V3 —— CPU 端到端渲染器（Alpha 合成）
 * ============================================================
 *
 *  目标：把前三章的知识串联起来，在 CPU 上实现一个完整的
 *        3DGS 前向渲染管线，输出可查看的 PPM 图像。
 *
 *  ────────────────────────────────────────────────────────
 *  1. Alpha Blending（前向合成，front-to-back）
 *  ────────────────────────────────────────────────────────
 *  3DGS 使用 front-to-back（从近到远）Alpha 混合：
 *
 *      T₀ = 1（初始透明度为 1.0，完全透过）
 *
 *  对深度排序后的第 i 个高斯：
 *
 *      贡献权重：     wᵢ = αᵢ · Tᵢ
 *      累积颜色：     C  += wᵢ · cᵢ
 *      更新透明度：   Tᵢ₊₁ = Tᵢ · (1 - αᵢ)
 *
 *  最终像素颜色：C + T_final · background_color
 *
 *  其中每个高斯对像素 p 的 alpha 贡献：
 *      αᵢ = opacity_i · exp( -½ · Δpᵀ Σ'ᵢ⁻¹ Δp )
 *      Δp = p_pixel - μ'ᵢ       （μ'ᵢ 是第 i 个高斯的屏幕坐标）
 *
 *  提前截止：当 Tᵢ < 0.0001 时，该像素不透明度已饱和，停止。
 *
 *  ────────────────────────────────────────────────────────
 *  2. 完整渲染管线（CPU 版）
 *  ────────────────────────────────────────────────────────
 *  Step 1  Preprocess（每个高斯）
 *           ├─ 3D 中心 → 相机空间 → 深度 z
 *           ├─ 透视投影 → 2D 屏幕坐标 μ'
 *           ├─ EWA → 2D 协方差 Σ'
 *           └─ SH 颜色（从视角方向）
 *
 *  Step 2  按深度排序（从近到远）
 *
 *  Step 3  Per-pixel Alpha accumulate
 *           对图像中每个像素，遍历排好序的高斯，做 Alpha Blend
 *
 *  Step 4  写出 PPM 图像文件
 *
 *  ────────────────────────────────────────────────────────
 *  3. 场景设计（可修改）
 *  ────────────────────────────────────────────────────────
 *  场景包含若干彩色高斯球，排列成简单图案：
 *    - 5 个球排成一排（x = -2, -1, 0, 1, 2）
 *    - 每个球颜色不同（红、橙、黄、绿、蓝）
 *    - 相机在 z=-4 处，朝 +z 方向看
 *
 * ============================================================
 */

#include "utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// ─────────────────────────────────────────────────────────────
//  高斯基本属性（世界空间，手工设定）
// ─────────────────────────────────────────────────────────────
struct GaussianData {
    float3 mu_world;     // 世界空间中心
    Quat   quat;         // 旋转（四元数）
    float3 scale;        // 三轴缩放
    float  opacity;      // 基础不透明度 ∈ (0, 1)
    float3 color;        // 固定 RGB 颜色（本版本不用 SH，直接给颜色）
};

// ─────────────────────────────────────────────────────────────
//  预处理结果（投影后）
// ─────────────────────────────────────────────────────────────
struct Projected {
    float2 mu2d;       // 屏幕坐标（像素）
    float  depth;      // 相机空间 z
    Mat2   cov2d;      // 2D 协方差
    Mat2   cov2d_inv;  // 2D 协方差逆
    float3 color;
    float  opacity;
    bool   valid;      // 是否在相机前方且在屏幕内
};

// ─────────────────────────────────────────────────────────────
//  Step 1：Preprocess — 3D → 2D 投影
// ─────────────────────────────────────────────────────────────
Projected preprocess(const GaussianData& g, const Camera& cam)
{
    Projected p;
    p.color   = g.color;
    p.opacity = g.opacity;
    p.valid   = false;

    // 变换到相机空间
    float3 p_cam = cam.world_to_cam(g.mu_world);

    // 剔除：在相机后面
    if (p_cam.z < 0.1f) return p;

    // 投影到像素
    p.mu2d  = cam.project(p_cam);
    p.depth = p_cam.z;

    // 剔除：屏幕外（留一点余量用于覆盖屏幕边缘的高斯）
    const float margin = 3.f * sqrtf(fmaxf(g.scale.x, g.scale.y)) * cam.fx / p_cam.z;
    if (p.mu2d.x < -margin || p.mu2d.x > cam.width  + margin) return p;
    if (p.mu2d.y < -margin || p.mu2d.y > cam.height + margin) return p;

    // EWA 2D 协方差
    Cov3d cov3d = build_cov3d(g.quat, g.scale);
    p.cov2d     = compute_cov2d(cov3d, cam, p_cam);
    p.cov2d_inv = p.cov2d.inv();
    p.valid     = true;
    return p;
}

// ─────────────────────────────────────────────────────────────
//  Step 3：Per-pixel Alpha Blending（front-to-back）
// ─────────────────────────────────────────────────────────────
float3 render_pixel(int px, int py,
                    const std::vector<Projected>& sorted,
                    float3 bg_color)
{
    float T = 1.f;            // 累积透明度（初始完全透过）
    float3 C = {0.f, 0.f, 0.f};

    for (const auto& g : sorted) {
        if (!g.valid) continue;

        // 像素到高斯中心的偏移（屏幕空间）
        float2 delta = {(float)px - g.mu2d.x, (float)py - g.mu2d.y};

        // 马氏距离平方
        float maha = g.cov2d_inv.quad(delta);

        // 高斯响应（截断：maha > 9 则跳过，3σ 以外贡献极小）
        if (maha > 9.f) continue;

        float gauss = expf(-0.5f * maha);
        float alpha = fminf(0.99f, g.opacity * gauss);

        // alpha 太小则跳过
        if (alpha < 1.f / 255.f) continue;

        // Front-to-back 混合
        C = C + (T * alpha) * g.color;
        T *= (1.f - alpha);

        // 提前截止：不透明度饱和
        if (T < 1e-4f) break;
    }

    // 添加背景颜色（剩余透明度 × 背景）
    return C + T * bg_color;
}

// ─────────────────────────────────────────────────────────────
//  构建演示场景
// ─────────────────────────────────────────────────────────────
std::vector<GaussianData> make_scene()
{
    std::vector<GaussianData> gs;
    Quat q_id = {1.f, 0.f, 0.f, 0.f};

    // 5 个不同颜色的高斯球，排成一排
    //                  位置              缩放         透明度   颜色(RGB)
    gs.push_back({ {-1.6f,  0.f, 0.f}, q_id, {0.4f,0.4f,0.4f}, 0.95f, {0.95f, 0.20f, 0.20f} }); // 红
    gs.push_back({ {-0.8f,  0.f, 0.f}, q_id, {0.4f,0.4f,0.4f}, 0.95f, {0.95f, 0.60f, 0.10f} }); // 橙
    gs.push_back({ { 0.0f,  0.f, 0.f}, q_id, {0.4f,0.4f,0.4f}, 0.95f, {0.90f, 0.90f, 0.10f} }); // 黄
    gs.push_back({ { 0.8f,  0.f, 0.f}, q_id, {0.4f,0.4f,0.4f}, 0.95f, {0.20f, 0.85f, 0.20f} }); // 绿
    gs.push_back({ { 1.6f,  0.f, 0.f}, q_id, {0.4f,0.4f,0.4f}, 0.95f, {0.20f, 0.40f, 0.95f} }); // 蓝

    // 背景：一个大的白色扁平高斯板（模拟地面/背景）
    gs.push_back({ {0.f, -0.7f, 0.5f}, q_id, {2.5f,0.05f,1.5f}, 0.6f, {0.8f, 0.8f, 0.8f} });

    // 两个重叠的半透明高斯，测试遮挡
    gs.push_back({ {0.f, 0.5f, -0.5f}, q_id, {0.6f,0.3f,0.3f}, 0.7f, {0.8f, 0.2f, 0.8f} }); // 紫，近（z=-0.5+3=2.5）
    gs.push_back({ {0.f, 0.5f,  0.3f}, q_id, {0.6f,0.3f,0.3f}, 0.7f, {0.2f, 0.8f, 0.8f} }); // 青，远

    return gs;
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────
int main()
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  3DGS Tutorial V3 — CPU Alpha 合成渲染器     ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    // ── 相机设置 ──────────────────────────────────────────────
    const int W = 256, H = 256;
    Camera cam;
    cam.width  = W; cam.height = H;
    cam.fx     = 200.f; cam.fy = 200.f;
    cam.cx     = W * 0.5f; cam.cy = H * 0.5f;
    cam.R_cw   = Mat3::identity();
    cam.t_cw   = {0.f, 0.f, 3.f};   // 相机在 (0,0,-3) 看向 +z

    float3 bg_color = {0.05f, 0.05f, 0.1f};   // 深蓝背景

    // ── 构建场景 ──────────────────────────────────────────────
    auto scene = make_scene();
    printf("\n场景: %zu 个高斯球\n", scene.size());

    // ── Step 1: Preprocess ────────────────────────────────────
    CpuTimer timer;
    timer.start();

    std::vector<Projected> projected;
    projected.reserve(scene.size());
    for (const auto& g : scene)
        projected.push_back(preprocess(g, cam));

    printf("Step 1 Preprocess: %zu 个有效投影\n",
           std::count_if(projected.begin(), projected.end(),
                         [](const Projected& p){ return p.valid; }));

    // ── Step 2: 按深度排序（front-to-back，从小到大）─────────
    std::vector<int> order(scene.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){
        return projected[a].depth < projected[b].depth;
    });

    std::vector<Projected> sorted;
    sorted.reserve(scene.size());
    for (int i : order) sorted.push_back(projected[i]);

    printf("Step 2 深度排序（从近至远）:\n");
    for (int i = 0; i < (int)sorted.size(); ++i) {
        const auto& p = sorted[i];
        if (!p.valid) continue;
        printf("  [%d] depth=%.2f  μ'=(%.1f, %.1f)\n",
               i, p.depth, p.mu2d.x, p.mu2d.y);
    }

    // ── Step 3: 逐像素渲染 ────────────────────────────────────
    std::vector<float> img(W * H * 3);

    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float3 c = render_pixel(px, py, sorted, bg_color);
            img[(py*W+px)*3+0] = c.x;
            img[(py*W+px)*3+1] = c.y;
            img[(py*W+px)*3+2] = c.z;
        }
    }

    double ms = timer.elapsed_ms();
    printf("\nStep 3 渲染 %dx%d 图像: %.1f ms\n", W, H, ms);
    printf("  平均 %.2f ms/pixel\n", ms / (W*H));

    // ── Step 4: 写出 PPM ──────────────────────────────────────
    save_ppm("3dgs_v3_output.ppm", img.data(), W, H);

    printf("\n[完成] 用图像查看器打开 3dgs_v3_output.ppm\n");
    printf("  下一步：3dgs_v4.cu — 用 CUDA 加速到实时性能\n");
    return 0;
}
