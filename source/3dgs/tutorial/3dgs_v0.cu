/*
 * ============================================================
 *  3DGS Tutorial  V0 —— 3D 高斯数学基础
 * ============================================================
 *
 *  目标：理解一个 3D Gaussian 在数学上的含义。
 *
 *  ────────────────────────────────────────────────────────
 *  1. 高斯函数定义
 *  ────────────────────────────────────────────────────────
 *  一维高斯：
 *      g(x) = exp( -x² / (2σ²) )
 *
 *  三维各向同性高斯（以原点为中心）：
 *      G(x) = exp( -½ · xᵀ Σ⁻¹ x )
 *
 *  其中 Σ 为协方差矩阵（正定对称 3×3）：
 *    - 对角线  Σ[i][i]  → 沿第 i 轴的方差（越大 → 越宽）
 *    - 非对角线 Σ[i][j] → 两轴之间的相关性（控制椭球朝向）
 *
 *  ────────────────────────────────────────────────────────
 *  2. 从旋转 + 缩放构建 Σ
 *  ────────────────────────────────────────────────────────
 *  3DGS 里每个高斯球用 (四元数 q, 缩放 s) 表示形状：
 *
 *      Σ = R · S · Sᵀ · Rᵀ
 *
 *  其中：
 *      R = quat_to_mat3(q)        ← SO(3) 旋转矩阵
 *      S = diag(sx, sy, sz)       ← 沿自身坐标轴的半径
 *
 *  几何直觉：
 *    - 先在局部坐标系里定义一个轴对齐椭球（由 S 决定）
 *    - 再用 R 把椭球旋转到世界坐标系里
 *
 *  令 M = R·S，则 Σ = M·Mᵀ
 *
 *  ────────────────────────────────────────────────────────
 *  3. 本文件演示内容
 *  ────────────────────────────────────────────────────────
 *  [Demo 1]  沿三个坐标轴对轴对齐高斯球采样，
 *            验证 scale 决定半径（G 降到 e⁻⁰·⁵ ≈ 0.607 处的距离 ≈ scale）
 *
 *  [Demo 2]  旋转高斯球后，主轴方向随之旋转，
 *            验证 R 对 Σ 的影响
 *
 *  [Demo 3]  两个高斯叠加（逐点相乘），
 *            演示高斯乘积仍是高斯（3DGS 不需要此特性，但有助于理解）
 *
 * ============================================================
 */

#include "utils.cuh"
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────
//  在点 x 处计算以 μ 为中心、协方差 cov3d 的 3D 高斯值
//
//  G(x) = exp( -½ · (x-μ)ᵀ Σ⁻¹ (x-μ) )
//
//  注：不带归一化常数 (2π)^(3/2) |Σ|^(1/2)，
//      3DGS 里的 alpha 混合不需要归一化
// ─────────────────────────────────────────────────────────────
float eval_gaussian3d(float3 mu, const Cov3d& cov, float3 x)
{
    float3 d = x - mu;                   // 位移向量

    Mat3 cov_mat  = cov.to_mat3();
    Mat3 cov_inv  = inverse(cov_mat);    // 协方差逆矩阵

    // 计算二次型 dᵀ Σ⁻¹ d
    float3 tmp = cov_inv * d;
    float  quad = dot3(d, tmp);

    return expf(-0.5f * quad);
}

// ─────────────────────────────────────────────────────────────
//  【辅助】打印沿某轴方向的高斯剖面（ASCII 条形图）
// ─────────────────────────────────────────────────────────────
void print_profile(const char* label, float3 mu, const Cov3d& cov,
                   float3 axis_dir, float range, int steps)
{
    printf("\n── %s ──\n", label);
    printf("  距离      G(x)    图形\n");
    for (int i = 0; i <= steps; ++i) {
        float t = -range + 2.f * range * i / steps;
        float3 x = mu + t * axis_dir;
        float g = eval_gaussian3d(mu, cov, x);

        // 打印条形图（宽度 20 字符）
        int bar = (int)(g * 20.f + 0.5f);
        printf("  %+6.2f  %6.4f  |", t, g);
        for (int j = 0; j < bar; ++j) putchar('#');
        putchar('\n');
    }
}

// ─────────────────────────────────────────────────────────────
//  Demo 1 —— 轴对齐高斯球：scale 决定三个半轴长度
//
//  使用单位四元数（无旋转），scale = (2, 1, 0.5)
//  预期：沿 x 轴衰减最慢（宽），沿 z 轴衰减最快（窄）
// ─────────────────────────────────────────────────────────────
void demo_axis_aligned()
{
    printf("\n==============================\n");
    printf("  Demo 1: 轴对齐高斯球\n");
    printf("  scale = (2.0, 1.0, 0.5)\n");
    printf("  四元数 = 单位（无旋转）\n");
    printf("==============================\n");

    float3 mu      = {0.f, 0.f, 0.f};
    Quat   q_ident = {1.f, 0.f, 0.f, 0.f};   // 单位四元数
    float3 scale   = {2.f, 1.f, 0.5f};

    Cov3d cov = build_cov3d(q_ident, scale);

    printf("[协方差矩阵 Σ（对角元 ≈ scale²）]\n");
    Mat3 cov_mat = cov.to_mat3();
    printf("  [ %6.3f  %6.3f  %6.3f ]\n", cov_mat[0][0], cov_mat[0][1], cov_mat[0][2]);
    printf("  [ %6.3f  %6.3f  %6.3f ]\n", cov_mat[1][0], cov_mat[1][1], cov_mat[1][2]);
    printf("  [ %6.3f  %6.3f  %6.3f ]\n", cov_mat[2][0], cov_mat[2][1], cov_mat[2][2]);
    printf("\n期望：Σ[0][0]=%.1f, Σ[1][1]=%.1f, Σ[2][2]=%.2f\n",
           scale.x*scale.x, scale.y*scale.y, scale.z*scale.z);

    // 沿 x 轴剖面（应宽）
    print_profile("沿 X 轴（scale=2.0，最宽）",
                  mu, cov, {1.f,0.f,0.f}, 4.f, 16);

    // 沿 y 轴剖面
    print_profile("沿 Y 轴（scale=1.0，中等）",
                  mu, cov, {0.f,1.f,0.f}, 4.f, 16);

    // 沿 z 轴剖面（应窄）
    print_profile("沿 Z 轴（scale=0.5，最窄）",
                  mu, cov, {0.f,0.f,1.f}, 4.f, 16);
}

// ─────────────────────────────────────────────────────────────
//  Demo 2 —— 旋转高斯球：R 改变椭球朝向
//
//  使用绕 Z 轴 45° 旋转的四元数，scale = (2, 0.5, 0.5)
//  预期：旋转前，沿 X 轴宽；旋转后，主轴转 45°，沿 (1,1,0) 方向最宽
// ─────────────────────────────────────────────────────────────
void demo_rotation()
{
    printf("\n==============================\n");
    printf("  Demo 2: 旋转高斯球\n");
    printf("  绕 Z 轴旋转 45°，scale = (2, 0.5, 0.5)\n");
    printf("==============================\n");

    float3 mu    = {0.f, 0.f, 0.f};
    float3 scale = {2.f, 0.5f, 0.5f};

    // ─ 旋转前（单位四元数）──────────────────────────────────
    Quat q_none = {1.f, 0.f, 0.f, 0.f};
    Cov3d cov_before = build_cov3d(q_none, scale);

    float g_x  = eval_gaussian3d(mu, cov_before, {1.f, 0.f, 0.f});
    float g_y  = eval_gaussian3d(mu, cov_before, {0.f, 1.f, 0.f});
    float g_xy = eval_gaussian3d(mu, cov_before, {0.707f, 0.707f, 0.f});

    printf("\n[旋转前]\n");
    printf("  G(1,0,0) = %.4f  ← 沿主轴（宽），期望接近 e^(-0.5/4) ≈ %.4f\n",
           g_x, expf(-0.5f / (scale.x*scale.x)));
    printf("  G(0,1,0) = %.4f  ← 垂直主轴（窄），期望接近 e^(-0.5/0.25) ≈ %.4f\n",
           g_y, expf(-0.5f / (scale.y*scale.y)));
    printf("  G(√2/2,√2/2,0) = %.4f\n", g_xy);

    // ─ 旋转后（绕 Z 轴 45°：θ=45°, axis=(0,0,1)）────────────
    //  q = (cos(θ/2), 0, 0, sin(θ/2)) = (cos22.5°, 0, 0, sin22.5°)
    float half  = 3.14159265f / 8.f;   // 22.5° = π/8
    Quat q_rot  = {cosf(half), 0.f, 0.f, sinf(half)};
    Cov3d cov_after = build_cov3d(q_rot, scale);

    g_x  = eval_gaussian3d(mu, cov_after, {1.f, 0.f, 0.f});
    g_y  = eval_gaussian3d(mu, cov_after, {0.f, 1.f, 0.f});
    g_xy = eval_gaussian3d(mu, cov_after, {0.707f, 0.707f, 0.f});

    printf("\n[旋转后（绕 Z 轴 45°）]\n");
    printf("  G(1,0,0)           = %.4f\n", g_x);
    printf("  G(0,1,0)           = %.4f\n", g_y);
    printf("  G(√2/2,√2/2,0)     = %.4f  ← 期望最大值（主轴方向）\n", g_xy);
    printf("\n  验证：旋转后沿 (1,1,0)/√2 方向的响应应 ≈ 旋转前沿 (1,0,0) 的响应\n");
}

// ─────────────────────────────────────────────────────────────
//  Demo 3 —— 不透明度 × 高斯 = 该高斯对像素 alpha 的贡献
//
//  3DGS 中每个高斯有一个学习到的基础不透明度 o ∈ (0,1)。
//  对像素 p 的实际 alpha 贡献：
//
//      α = o · G2D(p)          （G2D 是投影后的 2D 高斯）
//
//  本 demo 在 3D 空间演示相同概念：在不同距离下 α 如何衰减
// ─────────────────────────────────────────────────────────────
void demo_opacity()
{
    printf("\n==============================\n");
    printf("  Demo 3: 不透明度 × 高斯响应 = α 贡献\n");
    printf("==============================\n");

    float3 mu    = {0.f, 0.f, 0.f};
    Quat   q     = {1.f, 0.f, 0.f, 0.f};
    float3 scale = {1.f, 1.f, 1.f};   // 球形高斯，半径 1
    float  o     = 0.8f;               // 基础不透明度

    Cov3d cov = build_cov3d(q, scale);

    printf("\n  基础不透明度 o = %.2f，scale = (1,1,1)\n", o);
    printf("  沿 X 轴采样 α = o × G(x):\n\n");
    printf("  %-8s %-8s %-8s\n", "x", "G(x)", "α=o·G(x)");
    printf("  %-8s %-8s %-8s\n", "------", "------", "--------");

    for (float x = 0.0f; x <= 3.0f; x += 0.5f) {
        float g   = eval_gaussian3d(mu, cov, {x, 0.f, 0.f});
        float alp = o * g;
        printf("  %-8.2f %-8.4f %-8.4f\n", x, g, alp);
    }
    printf("\n  结论：距离中心 1σ（x=1）处 G ≈ e^(-0.5) ≈ 0.6065\n");
    printf("         距离中心 2σ（x=2）处 G ≈ e^(-2)   ≈ 0.1353\n");
    printf("         距离中心 3σ（x=3）处 G ≈ e^(-4.5) ≈ 0.0111\n");
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────
int main()
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  3DGS Tutorial V0 — 3D 高斯数学基础          ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    demo_axis_aligned();
    demo_rotation();
    demo_opacity();

    printf("\n\n[完成] 理解以上输出后，继续阅读 3dgs_v1.cu\n");
    printf("  下一步：将 3D 高斯投影到相机平面（EWA Splatting）\n");
    return 0;
}
