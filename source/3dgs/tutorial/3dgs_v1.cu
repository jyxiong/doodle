/*
 * ============================================================
 *  3DGS Tutorial  V1 —— 相机投影 & EWA Splatting
 * ============================================================
 *
 *  目标：将 3D 高斯球"投影"到 2D 屏幕，得到 2D 高斯椭圆。
 *
 *  ────────────────────────────────────────────────────────
 *  1. 相机模型
 *  ────────────────────────────────────────────────────────
 *  内参矩阵 K（针孔相机）：
 *      K = [ fx   0   cx ]
 *          [  0  fy   cy ]
 *          [  0   0    1 ]
 *
 *  外参（世界 → 相机变换）：
 *      p_cam = R_cw · p_world + t_cw
 *
 *  透视投影（相机坐标 → 像素）：
 *      u = fx · (x/z) + cx
 *      v = fy · (y/z) + cy
 *
 *  ────────────────────────────────────────────────────────
 *  2. 为什么需要 EWA？
 *  ────────────────────────────────────────────────────────
 *  透视投影是非线性的（做了除以 z 的操作），
 *  严格把 3D 高斯函数投影到 2D 需要进行积分，计算量很大。
 *
 *  EWA（Elliptical Weighted Average）采用局部线性化（一阶泰勒展开）：
 *  在高斯球中心 μ 处，用 Jacobian J 描述 3D→2D 的局部变形，
 *  得到 2D 协方差的近似：
 *
 *      Σ' = J · W · Σ · Wᵀ · Jᵀ   （取 2×2 左上块）
 *
 *  其中：
 *      W = R_cw（相机旋转部分，W ∈ ℝ³ˣ³）
 *
 *      J 是透视投影 π: ℝ³ → ℝ² 在 p_cam = (x,y,z) 处的 Jacobian（2×3）：
 *
 *          ∂u/∂x = fx/z         ∂u/∂y = 0            ∂u/∂z = -fx·x/z²
 *          ∂v/∂x = 0            ∂v/∂y = fy/z          ∂v/∂z = -fy·y/z²
 *
 *  ────────────────────────────────────────────────────────
 *  3. 2D 高斯的 alpha 贡献
 *  ────────────────────────────────────────────────────────
 *  投影后，对像素 p = (px, py) 的 alpha 贡献：
 *
 *      α = opacity · exp( -½ · Δpᵀ (Σ')⁻¹ Δp )
 *
 *      Δp = p - μ'      （μ' 为高斯在屏幕上的中心，单位：像素）
 *
 *  ────────────────────────────────────────────────────────
 *  4. 本文件演示
 *  ────────────────────────────────────────────────────────
 *  [Demo 1]  将一个轴对齐的 3D 高斯球投影到正对相机的场景，
 *            验证投影中心的像素坐标正确
 *
 *  [Demo 2]  倾斜视角下，圆球投影为椭圆（Σ' 不再是各向同性）
 *
 *  [Demo 3]  改变焦距（fx/fy），观察 2D 协方差如何缩放
 *
 *  [Demo 4]  在一个 32×32 的小屏幕上渲染单个 2D 高斯
 *            ASCII 输出，直观呈现 Splatting 效果
 *
 * ============================================================
 */

#include "utils.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────
//  构建一个正对相机的标准场景（相机内参）
// ─────────────────────────────────────────────────────────────
Camera make_frontal_camera(int W, int H)
{
    Camera cam;
    cam.width  = W;
    cam.height = H;
    cam.fx     = (float)W;          // fov ≈ 53°（fx = W ≈ 2·tan(fov/2)·f）
    cam.fy     = (float)W;
    cam.cx     = W * 0.5f;
    cam.cy     = H * 0.5f;

    // 相机在 (0,0,-3) 看向原点，无旋转（R_cw = I，t_cw = (0,0,3)）
    cam.R_cw   = Mat3::identity();
    cam.t_cw   = {0.f, 0.f, 3.f};
    return cam;
}

// ─────────────────────────────────────────────────────────────
//  Demo 1 —— 正对相机投影，验证中心点像素坐标
// ─────────────────────────────────────────────────────────────
void demo_frontal_projection()
{
    printf("\n==============================\n");
    printf("  Demo 1: 正对相机投影\n");
    printf("==============================\n");

    int W = 256, H = 256;
    Camera cam = make_frontal_camera(W, H);

    // 原点处的高斯球（球形，scale = 0.5）
    float3 mu_world = {0.f, 0.f, 0.f};
    Quat   q        = {1.f, 0.f, 0.f, 0.f};
    float3 scale    = {0.5f, 0.5f, 0.5f};

    Cov3d cov3d = build_cov3d(q, scale);

    // 变换到相机空间
    float3 p_cam = cam.world_to_cam(mu_world);
    printf("相机空间坐标: (%.2f, %.2f, %.2f)\n", p_cam.x, p_cam.y, p_cam.z);

    // 投影到像素
    float2 uv = cam.project(p_cam);
    printf("投影像素坐标: (%.1f, %.1f)\n", uv.x, uv.y);
    printf("期望中心像素: (%.1f, %.1f)\n", cam.cx, cam.cy);

    // 计算 2D 协方差
    Mat2 cov2d = compute_cov2d(cov3d, cam, p_cam);
    printf("\n2D 协方差 Σ':\n");
    printf("  [ %6.3f  %6.3f ]\n", cov2d.a, cov2d.b);
    printf("  [ %6.3f  %6.3f ]\n", cov2d.b, cov2d.c);
    printf("\n球形 scale=0.5，fx=%g，z=%.1f\n", cam.fx, p_cam.z);
    printf("期望 Σ'[0][0] = (fx/z)² · scale² + 0.3 = (%.1f/%.1f)² · %.2f + 0.3 = %.3f\n",
           cam.fx, p_cam.z,
           scale.x * scale.x,
           (cam.fx/p_cam.z)*(cam.fx/p_cam.z)*scale.x*scale.x + 0.3f);
}

// ─────────────────────────────────────────────────────────────
//  Demo 2 —— 侧面视角：圆球投影为椭圆
// ─────────────────────────────────────────────────────────────
void demo_oblique_projection()
{
    printf("\n==============================\n");
    printf("  Demo 2: 斜角视角\n");
    printf("  高斯球在 (1, 0, 0)，相机看向原点\n");
    printf("==============================\n");

    int W = 256, H = 256;

    // 建一个正视角相机，高斯球的主轴（scale_x 方向）正对屏幕
    Camera cam;
    cam.width = W; cam.height = H;
    cam.fx    = (float)W; cam.fy = (float)W;
    cam.cx    = W * 0.5f; cam.cy = H * 0.5f;
    // 相机绕 Y 轴旋转 θ，R_cw 是世界→相机的旋转
    // 为简单示意，使用旋转后的固定矩阵
    cam.R_cw       = Mat3::identity();  // 暂不旋转相机，只平移
    cam.t_cw       = {0.f, 0.f, 3.f};

    // 高斯球：轴对齐，scale = (1, 0.2, 0.2)（细长棒，主轴沿 X）
    float3 mu_world = {0.f, 0.f, 0.f};
    Quat   q        = {1.f, 0.f, 0.f, 0.f};
    float3 scale    = {1.f, 0.2f, 0.2f};

    Cov3d cov3d = build_cov3d(q, scale);
    float3 p_cam = cam.world_to_cam(mu_world);

    Mat2 cov2d = compute_cov2d(cov3d, cam, p_cam);

    printf("\n3D 协方差（scale=(1, 0.2, 0.2)，轴沿 X 方向）:\n");
    Mat3 c = cov3d.to_mat3();
    printf("  [ %5.2f  %5.2f  %5.2f ]\n", c[0][0], c[0][1], c[0][2]);
    printf("  [ %5.2f  %5.2f  %5.2f ]\n", c[1][0], c[1][1], c[1][2]);
    printf("  [ %5.2f  %5.2f  %5.2f ]\n\n", c[2][0], c[2][1], c[2][2]);

    printf("2D 协方差（正对相机）Σ':\n");
    printf("  [ %6.3f  %6.3f ]\n", cov2d.a, cov2d.b);
    printf("  [ %6.3f  %6.3f ]\n", cov2d.b, cov2d.c);
    printf("\n主轴比例 Σ'[0][0] / Σ'[1][1] = %.2f （应 ≈ (scale_x/scale_y)² ≈ %.1f）\n",
           cov2d.a / cov2d.c,
           (scale.x/scale.y) * (scale.x/scale.y));
}

// ─────────────────────────────────────────────────────────────
//  Demo 3 —— 焦距影响 2D 协方差大小
//
//  Σ' ≈ (fx/z)² · Σ_x
//  焦距翻倍 → 投影高斯像素大小翻倍（像素空间中更大的椭圆）
// ─────────────────────────────────────────────────────────────
void demo_focal_length()
{
    printf("\n==============================\n");
    printf("  Demo 3: 焦距对投影的影响\n");
    printf("==============================\n");

    float3 mu_world = {0.f, 0.f, 0.f};
    Quat   q        = {1.f, 0.f, 0.f, 0.f};
    float3 scale    = {0.5f, 0.5f, 0.5f};
    Cov3d  cov3d    = build_cov3d(q, scale);

    float fxs[] = {128.f, 256.f, 512.f};
    printf("\n  %-8s  %-8s  %-8s\n", "fx", "Σ'[0][0]", "半径(px)≈√Σ'");
    printf("  %-8s  %-8s  %-8s\n", "------", "--------", "----------");
    for (float fx : fxs) {
        Camera cam;
        cam.fx = cam.fy = fx; cam.cx = cam.cy = 128.f;
        cam.width = cam.height = 256;
        cam.R_cw = Mat3::identity(); cam.t_cw = {0.f, 0.f, 3.f};

        float3 p_cam = cam.world_to_cam(mu_world);
        Mat2   cov2d = compute_cov2d(cov3d, cam, p_cam);
        printf("  %-8.0f  %-8.3f  %-8.3f\n", fx, cov2d.a, sqrtf(cov2d.a));
    }
    printf("\n结论：焦距翻倍，Σ'[0][0] 约为原来 4 倍（像素半径翻倍）\n");
}

// ─────────────────────────────────────────────────────────────
//  Demo 4 —— ASCII 渲染单个 2D 高斯（Splatting 效果）
// ─────────────────────────────────────────────────────────────
void demo_ascii_splat()
{
    printf("\n==============================\n");
    printf("  Demo 4: ASCII Splatting 效果\n");
    printf("  32×32 屏幕，单个 2D 高斯椭圆\n");
    printf("==============================\n");

    const int W = 32, H = 32;

    Camera cam = make_frontal_camera(W, H);
    float3 mu  = {0.f, 0.f, 0.f};

    // 稍微倾斜的椭球：绕 Z 轴旋转 30°，scale = (3px, 1px后换算）
    float half = 3.14159265f / 12.f;  // 15° = π/12
    Quat   q   = {cosf(half), 0.f, 0.f, sinf(half)};
    float3 scale = {0.15f, 0.05f, 0.05f};  // 世界空间，投影后会被 fx 放大

    Cov3d  cov3d = build_cov3d(q, scale);
    float3 p_cam = cam.world_to_cam(mu);
    float2 mu2d  = cam.project(p_cam);
    Mat2   cov2d = compute_cov2d(cov3d, cam, p_cam);
    Mat2   cov2d_inv = cov2d.inv();

    float opacity = 0.95f;

    // 字符亮度映射（从暗到亮）
    const char* chars = " .:-=+*#@";
    int num_chars = 9;

    printf("\n");
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float2 delta = { (float)px - mu2d.x, (float)py - mu2d.y };
            float  maha  = cov2d_inv.quad(delta);
            float  g     = expf(-0.5f * maha);
            float  alpha = opacity * g;

            int idx = (int)(alpha * (num_chars - 1) + 0.5f);
            idx = idx < 0 ? 0 : (idx >= num_chars ? num_chars-1 : idx);
            putchar(chars[idx]);
        }
        putchar('\n');
    }
    printf("\n（dim→ ' ' = 0, '@' = 1.0）\n");
    printf("\n[完成] 继续阅读 3dgs_v2.cu — 球谐函数颜色\n");
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────
int main()
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  3DGS Tutorial V1 — 相机投影 & EWA Splatting ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    demo_frontal_projection();
    demo_oblique_projection();
    demo_focal_length();
    demo_ascii_splat();
    return 0;
}
