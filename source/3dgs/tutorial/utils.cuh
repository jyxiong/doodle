#pragma once
/*
 * ============================================================
 *  3DGS Tutorial — utils.cuh
 *  公共数学工具库（供所有 v0~v4 文件 #include）
 * ============================================================
 *
 *  提供：
 *    - Vec3 / Vec4 的算术运算（基于 CUDA 内置 float3/float4）
 *    - Mat3（3×3 行主序矩阵）& Mat2（2×2）
 *    - 四元数 Quat → 旋转矩阵
 *    - 协方差矩阵构建 & 求逆
 *    - 简单 CPU 计时器
 *    - CUDA_CHECK 宏
 * ============================================================
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// ─────────────────────────────────────────────────────────────
//  基本 float3 运算（host & device）
// ─────────────────────────────────────────────────────────────
__host__ __device__ inline float3 make_f3(float x, float y, float z)
{ return {x, y, z}; }

__host__ __device__ inline float3 operator+(float3 a, float3 b)
{ return {a.x+b.x, a.y+b.y, a.z+b.z}; }

__host__ __device__ inline float3 operator-(float3 a, float3 b)
{ return {a.x-b.x, a.y-b.y, a.z-b.z}; }

__host__ __device__ inline float3 operator*(float s, float3 v)
{ return {s*v.x, s*v.y, s*v.z}; }

__host__ __device__ inline float3 operator*(float3 v, float s)
{ return {s*v.x, s*v.y, s*v.z}; }

__host__ __device__ inline float3 operator/(float3 v, float s)
{ return {v.x/s, v.y/s, v.z/s}; }

__host__ __device__ inline float dot3(float3 a, float3 b)
{ return a.x*b.x + a.y*b.y + a.z*b.z; }

__host__ __device__ inline float len3(float3 v)
{ return sqrtf(dot3(v, v)); }

__host__ __device__ inline float3 normalize3(float3 v)
{ return v / len3(v); }

__host__ __device__ inline float3 clamp3(float3 v, float lo, float hi)
{
    return { fminf(fmaxf(v.x, lo), hi),
             fminf(fmaxf(v.y, lo), hi),
             fminf(fmaxf(v.z, lo), hi) };
}

// ─────────────────────────────────────────────────────────────
//  float2 运算
// ─────────────────────────────────────────────────────────────
__host__ __device__ inline float2 operator+(float2 a, float2 b)
{ return {a.x+b.x, a.y+b.y}; }

__host__ __device__ inline float2 operator-(float2 a, float2 b)
{ return {a.x-b.x, a.y-b.y}; }

__host__ __device__ inline float2 operator*(float s, float2 v)
{ return {s*v.x, s*v.y}; }

// ─────────────────────────────────────────────────────────────
//  Mat3 — 3×3 行主序矩阵
// ─────────────────────────────────────────────────────────────
struct Mat3 {
    float m[3][3];

    __host__ __device__ float* operator[](int i)             { return m[i]; }
    __host__ __device__ const float* operator[](int i) const { return m[i]; }

    // 单位矩阵
    __host__ __device__ static Mat3 identity() {
        Mat3 I{};
        I[0][0] = I[1][1] = I[2][2] = 1.f;
        return I;
    }

    // 零矩阵
    __host__ __device__ static Mat3 zero() { return Mat3{}; }
};

// Mat3 × Vec3
__host__ __device__ inline float3 operator*(const Mat3& M, float3 v) {
    return {
        M[0][0]*v.x + M[0][1]*v.y + M[0][2]*v.z,
        M[1][0]*v.x + M[1][1]*v.y + M[1][2]*v.z,
        M[2][0]*v.x + M[2][1]*v.y + M[2][2]*v.z
    };
}

// Mat3 × Mat3
__host__ __device__ inline Mat3 operator*(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// 转置
__host__ __device__ inline Mat3 transpose(const Mat3& M) {
    Mat3 T{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            T[i][j] = M[j][i];
    return T;
}

// 逆（对称正定 3×3，代数余子式法，适用于协方差矩阵）
__host__ __device__ inline Mat3 inverse(const Mat3& M) {
    float a = M[0][0], b = M[0][1], c = M[0][2];
    float d = M[1][1], e = M[1][2];
    float f = M[2][2];

    float det = a*(d*f - e*e) - b*(b*f - e*c) + c*(b*e - d*c);

    Mat3 inv{};
    float s = 1.f / det;
    inv[0][0] =  (d*f - e*e) * s;
    inv[0][1] =  (c*e - b*f) * s;
    inv[0][2] =  (b*e - d*c) * s;
    inv[1][0] =  inv[0][1];
    inv[1][1] =  (a*f - c*c) * s;
    inv[1][2] =  (b*c - a*e) * s;
    inv[2][0] =  inv[0][2];
    inv[2][1] =  inv[1][2];
    inv[2][2] =  (a*d - b*b) * s;
    return inv;
}

// ─────────────────────────────────────────────────────────────
//  Mat2 — 2×2 对称矩阵（用于 2D 协方差）
// ─────────────────────────────────────────────────────────────
struct Mat2 {
    float a, b, c;   // [ a  b ]
                     // [ b  c ]

    // 行列式
    __host__ __device__ float det() const { return a*c - b*b; }

    // 逆（对称 2×2）
    __host__ __device__ Mat2 inv() const {
        float s = 1.f / det();
        return { c*s, -b*s, a*s };
    }

    // 2D 二次型：xᵀ M x
    __host__ __device__ float quad(float2 x) const {
        return a*x.x*x.x + 2.f*b*x.x*x.y + c*x.y*x.y;
    }
};

// ─────────────────────────────────────────────────────────────
//  四元数 → 旋转矩阵
// ─────────────────────────────────────────────────────────────
struct Quat { float w, x, y, z; };

// 对四元数归一化
__host__ __device__ inline Quat normalize(Quat q) {
    float n = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return {q.w/n, q.x/n, q.y/n, q.z/n};
}

// 单位四元数 → SO(3) 旋转矩阵
__host__ __device__ inline Mat3 quat_to_mat3(Quat q) {
    q = normalize(q);
    float w=q.w, x=q.x, y=q.y, z=q.z;
    Mat3 R{};
    R[0][0] = 1 - 2*(y*y + z*z);  R[0][1] = 2*(x*y - w*z);      R[0][2] = 2*(x*z + w*y);
    R[1][0] = 2*(x*y + w*z);      R[1][1] = 1 - 2*(x*x + z*z);  R[1][2] = 2*(y*z - w*x);
    R[2][0] = 2*(x*z - w*y);      R[2][1] = 2*(y*z + w*x);      R[2][2] = 1 - 2*(x*x + y*y);
    return R;
}

// ─────────────────────────────────────────────────────────────
//  3D 协方差矩阵（对称，存储上三角 6 个元素）
// ─────────────────────────────────────────────────────────────
//  Σ = R · S · Sᵀ · Rᵀ，其中 S = diag(sx, sy, sz)
//
//  推导：
//    令 M = R · S  → M 的每列是 R 的对应列乘以 scale
//    则 Σ = M · Mᵀ
//
//  存储顺序：[s00, s01, s02, s11, s12, s22]
struct Cov3d {
    float s00, s01, s02, s11, s12, s22;

    // 转换为完整 Mat3
    __host__ __device__ Mat3 to_mat3() const {
        Mat3 M{};
        M[0][0]=s00; M[0][1]=s01; M[0][2]=s02;
        M[1][0]=s01; M[1][1]=s11; M[1][2]=s12;
        M[2][0]=s02; M[2][1]=s12; M[2][2]=s22;
        return M;
    }
};

__host__ __device__ inline Cov3d build_cov3d(Quat q, float3 scale) {
    Mat3 R = quat_to_mat3(q);
    // M = R · S：每列乘以对应 scale
    Mat3 M{};
    for (int i = 0; i < 3; ++i) {
        M[i][0] = R[i][0] * scale.x;
        M[i][1] = R[i][1] * scale.y;
        M[i][2] = R[i][2] * scale.z;
    }
    // Σ = M · Mᵀ
    Mat3 cov = M * transpose(M);
    return { cov[0][0], cov[0][1], cov[0][2],
             cov[1][1], cov[1][2], cov[2][2] };
}

// ─────────────────────────────────────────────────────────────
//  相机结构体
// ─────────────────────────────────────────────────────────────
struct Camera {
    float fx, fy;   // 焦距（像素）
    float cx, cy;   // 主点（像素）
    int   width, height;

    // 外参：世界 → 相机
    // p_cam = R_cw · p_world + t_cw
    Mat3  R_cw;
    float3 t_cw;

    // 将世界坐标系点变换到相机坐标系
    __host__ __device__ float3 world_to_cam(float3 p) const {
        return R_cw * p + t_cw;
    }

    // 相机坐标系点投影到像素坐标
    // 返回 (u, v) 像素，z 为深度
    __host__ __device__ float2 project(float3 p_cam) const {
        return {
            fx * (p_cam.x / p_cam.z) + cx,
            fy * (p_cam.y / p_cam.z) + cy
        };
    }
};

// ─────────────────────────────────────────────────────────────
//  EWA Splatting：3D 协方差 → 2D 协方差
// ─────────────────────────────────────────────────────────────
//
//  Σ' = J · W · Σ · Wᵀ · Jᵀ  （取 2×2 左上块）
//
//  W = R_cw（相机旋转部分）
//
//  J 是透视投影在 p_cam 处的 Jacobian（2×3）：
//     J = [ fx/z    0    -fx·x/z² ]
//         [  0    fy/z   -fy·y/z² ]
//
//  注：直接用 3×3 J（第 3 行为 0），最后取 2×2 左上角
__host__ __device__ inline Mat2 compute_cov2d(
    const Cov3d& cov3d, const Camera& cam, float3 p_cam)
{
    float x = p_cam.x, y = p_cam.y, z = p_cam.z;
    float fx = cam.fx, fy = cam.fy;

    // Jacobian J（3×3，第 3 行为 0 → 以 Mat3 表示）
    Mat3 J{};
    J[0][0] =  fx / z;        J[0][1] = 0.f;           J[0][2] = -fx * x / (z*z);
    J[1][0] = 0.f;            J[1][1] =  fy / z;        J[1][2] = -fy * y / (z*z);
    // 第 3 行全 0

    // W = R_cw
    const Mat3& W = cam.R_cw;

    // T = J · W
    Mat3 T = J * W;

    // 3D 协方差全矩阵
    Mat3 cov_mat = cov3d.to_mat3();

    // Σ'3x3 = T · Σ · Tᵀ
    Mat3 cov2d_full = T * cov_mat * transpose(T);

    // 取 2×2 左上角
    // 加一点低通滤波（论文中为避免混叠在对角上加 0.3）
    return {
        cov2d_full[0][0] + 0.3f,
        cov2d_full[0][1],
        cov2d_full[1][1] + 0.3f
    };
}

// ─────────────────────────────────────────────────────────────
//  PPM 图像输出
// ─────────────────────────────────────────────────────────────
inline void save_ppm(const char* filename, const float* rgb,
                     int width, int height)
{
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "无法写入 %s\n", filename); return; }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = (unsigned char)(fminf(rgb[3*i  ], 1.f) * 255.f);
        unsigned char g = (unsigned char)(fminf(rgb[3*i+1], 1.f) * 255.f);
        unsigned char b = (unsigned char)(fminf(rgb[3*i+2], 1.f) * 255.f);
        fwrite(&r, 1, 1, f);
        fwrite(&g, 1, 1, f);
        fwrite(&b, 1, 1, f);
    }
    fclose(f);
    printf("已写入 %s（%d×%d）\n", filename, width, height);
}

// ─────────────────────────────────────────────────────────────
//  通用工具
// ─────────────────────────────────────────────────────────────

// CPU 计时器
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// CUDA 错误检查宏
#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(_e));       \
        exit(1);                                                   \
    }                                                              \
} while (0)
