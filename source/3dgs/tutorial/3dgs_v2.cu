/*
 * ============================================================
 *  3DGS Tutorial  V2 —— 球谐函数（Spherical Harmonics）颜色
 * ============================================================
 *
 *  目标：理解 3DGS 如何用球谐函数表示随视角变化的颜色。
 *
 *  ────────────────────────────────────────────────────────
 *  1. 为什么需要球谐函数？
 *  ────────────────────────────────────────────────────────
 *  真实世界中，物体颜色随观察方向（高光、镜面反射）而变化。
 *  如果只存一个固定 RGB 颜色，无法表达这种 view-dependent 效果。
 *
 *  球谐函数是定义在单位球面上的一组正交基函数，
 *  类似"球面上的傅里叶基"：
 *    - 低阶（l=0）：常数，描述漫反射（与方向无关）
 *    - 一阶（l=1）：线性（余弦瓣），描述简单方向性
 *    - 二阶（l=2）以上：捕捉高频的高光/各向异性效果
 *
 *  ────────────────────────────────────────────────────────
 *  2. 球谐基函数（实数形式，Condon-Shortley 相位）
 *  ────────────────────────────────────────────────────────
 *  l=0（1 个基函数）：
 *      Y₀⁰ = 0.2821  （常数）
 *
 *  l=1（3 个基函数，以方向向量 d=(dx,dy,dz) 表示）：
 *      Y₁⁻¹ = 0.4886 · dy
 *      Y₁⁰  = 0.4886 · dz
 *      Y₁¹  = 0.4886 · dx
 *
 *  l=2（5 个基函数）：
 *      Y₂⁻² = 1.0924 · dx·dy
 *      Y₂⁻¹ = 1.0924 · dy·dz
 *      Y₂⁰  = 0.3153 · (3·dz²-1)
 *      Y₂¹  = 1.0924 · dx·dz
 *      Y₂²  = 0.5462 · (dx²-dy²)
 *
 *  l=3（7 个基函数）：
 *      Y₃⁻³ = 0.5900 · dy·(3·dx²-dy²)
 *      Y₃⁻² = 2.8906 · dx·dy·dz
 *      Y₃⁻¹ = 0.4572 · dy·(5dz²-1)
 *      Y₃⁰  = 0.3732 · dz·(5dz²-3)
 *      Y₃¹  = 0.4572 · dx·(5dz²-1)
 *      Y₃²  = 1.4453 · (dx²-dy²)·dz
 *      Y₃³  = 0.5900 · dx·(dx²-3dy²)
 *
 *  ────────────────────────────────────────────────────────
 *  3. 颜色评估
 *  ────────────────────────────────────────────────────────
 *  每个高斯存储 48 个 SH 系数（16 个基 × 3 个 RGB 通道）：
 *
 *      C(d) = Σ_{i=0}^{15}  c_i · Y_i(d)       （每通道独立求和）
 *
 *  最后加上 0.5 偏置（使默认颜色约为 0.5），并 clamp 到 [0,1]。
 *
 *  ────────────────────────────────────────────────────────
 *  4. 本文件演示
 *  ────────────────────────────────────────────────────────
 *  [Demo 1]  仅用 l=0（常数）：颜色与方向无关，输出固定颜色
 *
 *  [Demo 2]  加入 l=1（线性）：颜色随视角方向线性变化
 *
 *  [Demo 3]  完整 0~3 阶：从不同方向观察同一个高斯，
 *            打印颜色随仰角/方位角变化的表格
 *
 *  [Demo 4]  生成一张"环境光球"图像（64×32），
 *            颜色 = SH 评估值，输出 PPM 文件
 *
 * ============================================================
 */

#include "utils.cuh"
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────
//  SH 系数常数
// ─────────────────────────────────────────────────────────────
#define SH_C0   0.28209479177387814f
#define SH_C1   0.4886025119029199f
#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 1.0925484305920792f
#define SH_C2_4 0.5462742152960396f
#define SH_C3_0 0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 0.4570457994644658f
#define SH_C3_5 1.4453057213202770f
#define SH_C3_6 0.5900435899266435f

// SH 阶数对应系数数目：1, 4, 9, 16
#define SH_DEGREE 3
#define SH_COEFFS 16   // (SH_DEGREE+1)²

// ─────────────────────────────────────────────────────────────
//  计算 SH 基函数值（给定单位方向向量 d）
//  输出数组 Y[0..15]
// ─────────────────────────────────────────────────────────────
__host__ __device__ inline void eval_sh_basis(float3 d, float Y[SH_COEFFS])
{
    float x = d.x, y = d.y, z = d.z;

    // l=0
    Y[0]  = SH_C0;

    // l=1
    Y[1]  = SH_C1 * y;
    Y[2]  = SH_C1 * z;
    Y[3]  = SH_C1 * x;

    // l=2
    float xy = x*y, yz = y*z, xz = x*z;
    float x2 = x*x, y2 = y*y, z2 = z*z;

    Y[4]  = SH_C2_0 * xy;
    Y[5]  = SH_C2_1 * yz;
    Y[6]  = SH_C2_2 * (3.f*z2 - 1.f);
    Y[7]  = SH_C2_3 * xz;
    Y[8]  = SH_C2_4 * (x2 - y2);

    // l=3
    Y[9]  = SH_C3_0 * y * (3.f*x2 - y2);
    Y[10] = SH_C3_1 * xy * z;
    Y[11] = SH_C3_2 * y * (5.f*z2 - 1.f);
    Y[12] = SH_C3_3 * z * (5.f*z2 - 3.f);
    Y[13] = SH_C3_4 * x * (5.f*z2 - 1.f);
    Y[14] = SH_C3_5 * (x2 - y2) * z;
    Y[15] = SH_C3_6 * x * (x2 - 3.f*y2);
}

// ─────────────────────────────────────────────────────────────
//  从 SH 系数 + 视角方向 d 计算 RGB 颜色
//  coeffs: 48 个 float，排布为 [R0,R1,...,R15, G0,...,G15, B0,...,B15]
// ─────────────────────────────────────────────────────────────
__host__ __device__ inline float3 sh_to_color(const float* coeffs, float3 d)
{
    d = normalize3(d);
    float Y[SH_COEFFS];
    eval_sh_basis(d, Y);

    float r = 0.f, g = 0.f, b = 0.f;
    for (int i = 0; i < SH_COEFFS; ++i) {
        r += coeffs[i            ] * Y[i];
        g += coeffs[i + SH_COEFFS] * Y[i];
        b += coeffs[i + 2*SH_COEFFS] * Y[i];
    }

    // 加偏置 0.5（使 0 系数时颜色 ≈ 0.5 灰），clamp 到 [0,1]
    return clamp3({r + 0.5f, g + 0.5f, b + 0.5f}, 0.f, 1.f);
}

// ─────────────────────────────────────────────────────────────
//  Demo 1 —— 只用 l=0（常数项），颜色与方向无关
// ─────────────────────────────────────────────────────────────
void demo_dc_only()
{
    printf("\n==============================\n");
    printf("  Demo 1: 仅 l=0（漫反射常数项）\n");
    printf("==============================\n");

    float coeffs[3 * SH_COEFFS] = {};

    // c00_R = 0.3/SH_C0，使 r ≈ 0.3 + 0.5 = 0.8（偏红色）
    coeffs[0]             = 0.3f / SH_C0;  // R DC
    coeffs[SH_COEFFS]     = -0.3f / SH_C0; // G DC → 0.5-0.3=0.2
    coeffs[2 * SH_COEFFS] = -0.4f / SH_C0; // B DC → 0.5-0.4=0.1

    printf("  SH 系数：只设置 l=0 DC 项\n");
    printf("  期望颜色（与方向无关）≈ (0.80, 0.20, 0.10)\n\n");

    float3 dirs[] = {
        {1,0,0}, {-1,0,0}, {0,1,0}, {0,0,1}, {0.577f,0.577f,0.577f}
    };
    const char* dir_names[] = {
        "+X", "-X", "+Y", "+Z", "diagonal"
    };

    printf("  %-10s  RGB\n", "方向");
    for (int i = 0; i < 5; ++i) {
        float3 c = sh_to_color(coeffs, dirs[i]);
        printf("  %-10s  (%.3f, %.3f, %.3f)\n", dir_names[i], c.x, c.y, c.z);
    }
    printf("\n  → 所有方向颜色相同（√）\n");
}

// ─────────────────────────────────────────────────────────────
//  Demo 2 —— l=1（线性），颜色随方向变化
// ─────────────────────────────────────────────────────────────
void demo_linear_sh()
{
    printf("\n==============================\n");
    printf("  Demo 2: l=0 + l=1，颜色随方向线性变化\n");
    printf("==============================\n");

    float coeffs[3 * SH_COEFFS] = {};

    // DC: 灰色基础
    coeffs[0]             = 0.f;   // R DC → 0.5
    coeffs[SH_COEFFS]     = 0.f;   // G DC → 0.5
    coeffs[2*SH_COEFFS]   = 0.f;   // B DC → 0.5

    // R 通道：沿 +X 方向变亮（Y₁¹ = SH_C1 · x）
    coeffs[3]                = 0.5f / SH_C1;   // R, Y₁¹ 系数
    // B 通道：沿 +Z 方向变亮（Y₁⁰ = SH_C1 · z）
    coeffs[2*SH_COEFFS + 2] = 0.5f / SH_C1;   // B, Y₁⁰ 系数

    printf("  设置：R 通道沿 +X 亮，B 通道沿 +Z 亮\n\n");
    printf("  %-10s  RGB（期望：+X 偏红，-X 偏青；+Z 偏蓝，-Z 偏黄）\n\n", "方向");

    struct { float3 d; const char* name; } dirs[] = {
        {{1,0,0},   "+X "}, {{-1,0,0},  "-X "},
        {{0,0,1},   "+Z "}, {{0,0,-1},  "-Z "},
        {{0,1,0},   "+Y "}, {{0,-1,0},  "-Y "}
    };
    for (auto& e : dirs) {
        float3 c = sh_to_color(coeffs, e.d);
        printf("  %-6s  (%.3f, %.3f, %.3f)\n", e.name, c.x, c.y, c.z);
    }
}

// ─────────────────────────────────────────────────────────────
//  Demo 3 —— 完整 0~3 阶：随方位角变化的颜色表格
// ─────────────────────────────────────────────────────────────
void demo_full_sh_table()
{
    printf("\n==============================\n");
    printf("  Demo 3: 完整 0~3 阶 SH 随方位角变化\n");
    printf("  水平环绕（θ=90°，φ 从0~360°）\n");
    printf("==============================\n");

    float coeffs[3 * SH_COEFFS] = {};

    // 设置各阶系数，构造一个有趣的颜色变化
    // DC
    coeffs[0]           = 0.f;   // R → 0.5
    coeffs[SH_COEFFS]   = 0.f;   // G → 0.5
    coeffs[2*SH_COEFFS] = 0.f;   // B → 0.5
    // l=1
    coeffs[3]              = 0.4f / SH_C1;  // R, Y1_x
    coeffs[SH_COEFFS + 1]  = 0.4f / SH_C1;  // G, Y1_y
    coeffs[2*SH_COEFFS + 2]= 0.4f / SH_C1;  // B, Y1_z
    // l=2
    coeffs[4]              = 0.2f / SH_C2_0;  // R, Y2_xy（高光）
    coeffs[SH_COEFFS + 8]  = 0.2f / SH_C2_4;  // G, Y2_x2-y2

    printf("\n  φ(°)  R      G      B\n");
    printf("  ----  -----  -----  -----\n");
    float pi = 3.14159265f;
    for (int i = 0; i <= 12; ++i) {
        float phi = (float)i / 12.f * 2.f * pi;
        float3 d  = { cosf(phi), sinf(phi), 0.f };
        float3 c  = sh_to_color(coeffs, d);
        printf("  %4.0f  %.3f  %.3f  %.3f\n",
               phi * 180.f / pi, c.x, c.y, c.z);
    }
}

// ─────────────────────────────────────────────────────────────
//  Demo 4 —— 输出"环境光球" PPM 图像（经纬图）
//
//  将 0~3 阶 SH 系数在整个单位球面上可视化，
//  每像素 = 以该像素对应的球面方向评估 SH 颜色
// ─────────────────────────────────────────────────────────────
void demo_envmap_ppm()
{
    printf("\n==============================\n");
    printf("  Demo 4: 输出 SH 环境光球 PPM\n");
    printf("==============================\n");

    const int W = 128, H = 64;
    float img[W * H * 3];

    float coeffs[3 * SH_COEFFS] = {};
    // 构造彩色 SH
    coeffs[0]              =  0.1f / SH_C0;   // R DC
    coeffs[SH_COEFFS]      = -0.1f / SH_C0;   // G DC
    coeffs[2*SH_COEFFS]    = -0.2f / SH_C0;   // B DC
    coeffs[3]              =  0.5f / SH_C1;   // R, Y1_x
    coeffs[SH_COEFFS + 1]  =  0.5f / SH_C1;   // G, Y1_y
    coeffs[2*SH_COEFFS+2]  =  0.5f / SH_C1;   // B, Y1_z
    coeffs[4]              =  0.3f / SH_C2_0; // R, Y2_xy
    coeffs[SH_COEFFS + 6]  =  0.3f / SH_C2_2; // G, Y2_z2
    coeffs[2*SH_COEFFS+8]  =  0.3f / SH_C2_4; // B, Y2_x2-y2

    float pi = 3.14159265f;
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float phi   = ((float)px + 0.5f) / W * 2.f * pi - pi;
            float theta = ((float)py + 0.5f) / H * pi;
            float3 d = {
                sinf(theta) * cosf(phi),
                sinf(theta) * sinf(phi),
                cosf(theta)
            };
            float3 c = sh_to_color(coeffs, d);
            img[(py*W+px)*3+0] = c.x;
            img[(py*W+px)*3+1] = c.y;
            img[(py*W+px)*3+2] = c.z;
        }
    }

    save_ppm("3dgs_v2_envmap.ppm", img, W, H);
    printf("用图像查看器打开 3dgs_v2_envmap.ppm 查看颜色随方向的变化\n");
    printf("\n[完成] 继续阅读 3dgs_v3.cu — CPU Alpha 合成渲染器\n");
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────
int main()
{
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  3DGS Tutorial V2 — 球谐函数颜色             ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    demo_dc_only();
    demo_linear_sh();
    demo_full_sh_table();
    demo_envmap_ppm();
    return 0;
}
