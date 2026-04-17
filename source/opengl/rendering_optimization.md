# OpenGL 渲染优化思路总结

## 一、CPU 端优化（减少 CPU 负载与 Draw Call）

### 1.1 减少 Draw Call
- **合批（Batching）**：将多个使用相同状态（着色器、纹理、混合模式）的几何体合并为一次 Draw Call。
- **实例化渲染（Instanced Rendering）**：使用 `glDrawArraysInstanced` / `glDrawElementsInstanced`，一次绘制大量相同网格的不同实例（如草地、粒子），通过 `gl_InstanceID` 或 Instance Buffer 传递每个实例的变换数据。
- **间接绘制（Indirect Draw）**：使用 `glMultiDrawIndirect`，将绘制命令写入 GPU Buffer（`GL_DRAW_INDIRECT_BUFFER`），由 GPU 自主发起绘制，彻底消除 CPU 每帧提交命令的开销。

### 1.2 减少状态切换
- 按材质/着色器排序绘制命令，将相同状态的对象连续绘制，减少 `glBindTexture`、`glUseProgram` 等状态切换开销。
- 使用 **Texture Array**（`GL_TEXTURE_2D_ARRAY`）或 **Bindless Texture**（`ARB_bindless_texture`），减少纹理绑定切换。

### 1.3 剔除（Culling）
- **视锥剔除（Frustum Culling）**：在 CPU 端用 AABB/BVH 剔除不在视锥内的对象，避免无效提交。
- **遮挡剔除（Occlusion Culling）**：使用 `glBeginQuery(GL_SAMPLES_PASSED)` 查询对象是否被遮挡，配合 Z-Prepass 实现 GPU 端遮挡剔除。
- **细节层次（LOD）**：根据对象与相机距离切换不同精度的网格。

---

## 二、GPU 顶点阶段优化

### 2.1 顶点数据布局
- 使用交错（Interleaved）顶点格式 `[pos|normal|uv]` 提升缓存局部性；或分离（Separate）格式按需绑定。
- 使用 **VAO（Vertex Array Object）** 缓存顶点属性绑定状态，避免每帧重复设置。
- 对静态几何使用 `GL_STATIC_DRAW`，动态几何使用 `GL_DYNAMIC_DRAW` / `GL_STREAM_DRAW` 提示驱动分配合适内存。

### 2.2 索引与顶点缓存优化
- 对网格顶点进行 **Forsyth 顺序优化**（或类似算法），提升 Post-Transform 顶点缓存命中率。
- 使用 `glPrimitiveRestartIndex` 合并多个 Triangle Strip，减少 DrawCall 数量。

### 2.3 Geometry Shader 谨慎使用
- Geometry Shader 在许多硬件上性能差且难以并行，尽量用计算着色器或实例化替代。

---

## 三、GPU 光栅化阶段优化

### 3.1 Early-Z 与 Z-Prepass
- **Early-Z**：驱动自动启用，前提是片元着色器不写 `gl_FragDepth` 也不调用 `discard`。
- **Z-Prepass（Depth Prepass）**：先用极简着色器（仅输出深度）渲染所有不透明物体，填充深度缓冲；正式渲染时开启深度测试 `GL_EQUAL`，使通不过深度测试的片元在着色器执行前被丢弃，避免 Overdraw。

### 3.2 背面剔除
- 确保开启 `glEnable(GL_CULL_FACE)`，正确设置 `glFrontFace(GL_CCW)`，剔除背面三角形，减少约一半光栅化工作。

### 3.3 Scissor / Stencil 裁剪
- 使用 Scissor Test 限制渲染区域（如 UI、分屏），避免对屏幕外像素进行计算。

---

## 四、片元（Fragment）着色器优化

### 4.1 减少 ALU 计算
- 将逐顶点（per-vertex）计算移出片元着色器（如光照方向归一化、矩阵变换）。
- 将常量、预计算结果通过 Uniform 或 UBO 传入，避免片元着色器中重复计算。
- 使用低精度类型：`mediump`（GLSL ES）/ `half`（HLSL）在精度要求不高的地方节省带宽和寄存器。

### 4.2 纹理采样优化
- 为纹理开启 **Mipmap**，减少纹理缓存缺失（`glGenerateMipmap`）。
- 使用合适的过滤器：`GL_LINEAR_MIPMAP_LINEAR`（高质量）或 `GL_NEAREST_MIPMAP_NEAREST`（性能优先）。
- 使用 **纹理压缩格式**（BC/DXT、ASTC、ETC2），减少显存占用和带宽。
- 避免在着色器中依赖纹理坐标的动态计算（dependent texture reads），会破坏预取机制。

### 4.3 控制流
- 避免着色器中的动态分支（`if`/`for` 依赖纹理值），GPU SIMD 架构下两分支均会执行。
- 必要时可将分支展开为数学公式（`mix`、`step`、`clamp` 等）。

### 4.4 减少 Overdraw
- 不透明物体**从前到后**排序，配合 Early-Z 减少无效片元着色。
- 透明物体**从后到前**排序（或使用 Order-Independent Transparency）。

---

## 五、内存与带宽优化

### 5.1 Uniform Buffer Object（UBO）与 Shader Storage Buffer（SSBO）
- 使用 **UBO**（`GL_UNIFORM_BUFFER`）将频繁更新的 Uniform 数据（相机矩阵、灯光参数）集中传输，利用绑定点（Binding Point）在多个着色器间共享，减少 `glUniform*` 调用。
- 使用 **SSBO**（`GL_SHADER_STORAGE_BUFFER`）传递大规模可写数据（蒙皮矩阵、粒子状态）。

### 5.2 Framebuffer 优化
- 使用 **Renderbuffer**（而非 Texture）存储不需要采样的附件（深度/模板），驱动可使用更高效的存储格式。
- **Tile-Based 架构**（移动端 GPU）：使用 `glInvalidateFramebuffer` 标记附件为不需要写回主存，减少带宽。
- 避免 CPU 读取 Framebuffer（`glReadPixels`），这会强制 GPU-CPU 同步，产生严重气泡（stall）。

### 5.3 纹理格式与精度
- 使用合适的纹理内部格式：`GL_R11F_G11F_B10F` 替代 `GL_RGB16F`，`GL_RGB10_A2` 替代 `GL_RGBA16F`。
- 使用 `GL_SRGB8_ALPHA8` 在硬件完成 Gamma 转换，避免手动计算。

### 5.4 Buffer 上传策略
- 使用 **Persistent Mapped Buffer**（`GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT`）+ 三重缓冲减少动态数据上传的同步开销。
- 使用 `glBufferSubData` 局部更新而非整块重新上传。

---

## 六、多线程与异步

### 6.1 异步资源上传
- 使用 **Pixel Buffer Object（PBO）**（`GL_PIXEL_UNPACK_BUFFER`）异步上传纹理（`glTexSubImage2D`），避免阻塞渲染线程。
- 利用多 OpenGL Context（共享资源）在后台线程加载纹理/网格。

### 6.2 GPU 查询与 Timer
- 使用 `GL_TIME_ELAPSED` Query 精确测量 GPU 端各阶段耗时，定位瓶颈。
- 避免 `glFinish`（全同步），使用 `glFenceSync` / `glClientWaitSync` 进行轻量同步。

---

## 七、Compute Shader 加速

- 将 CPU 端耗时的并行计算（粒子模拟、蒙皮、视锥剔除、光照剔除）迁移至 **Compute Shader**，直接在显存中完成，避免 PCIe 数据回传。
- 使用 **Atomic 操作** 和 **Shared Memory** 优化 Compute Shader 中的归约（Reduction）、直方图等算法。
- 合理配置 Work Group 大小（通常 64 或 256 threads），充分占用 SM/CU。

---

## 八、延迟渲染（Deferred Rendering）

- **G-Buffer Pass**：将几何信息（位置/法线/反照率/金属度粗糙度）写入多个 RenderTarget（MRT）。
- **Lighting Pass**：对 G-Buffer 中的每个像素执行一次光照计算，不再受几何复杂度影响，适合大量动态光源场景。
- 缺点：显存带宽大幅增加（需压缩法线/存储布局优化），不便处理透明物体和 MSAA。
- **Tiled/Clustered Deferred**：将屏幕分为 Tile（如 16×16），为每个 Tile 建立光源列表，只对影响该 Tile 的光源计算，进一步减少无效光照计算。

---

## 九、性能分析工具

| 工具 | 用途 |
|------|------|
| **RenderDoc** | 逐帧捕捉、查看资源、着色器调试 |
| **NVIDIA Nsight Graphics** | GPU 管线瓶颈分析、着色器 occupancy |
| **AMD Radeon GPU Profiler** | AMD GPU 性能分析 |
| **Intel GPA** | Intel 集显 / Arc 性能分析 |
| **apitrace** | OpenGL API 调用追踪与回放 |
| **GL_TIME_ELAPSED Query** | 轻量级 GPU 计时，内嵌于代码中 |

---

## 十、常见优化检查清单

- [ ] 开启背面剔除 `glEnable(GL_CULL_FACE)`
- [ ] 不透明物体从前到后排序，透明物体从后到前排序
- [ ] 使用 VAO 缓存顶点状态
- [ ] 使用 UBO 批量传递 Uniform 数据
- [ ] 为所有纹理生成 Mipmap 并使用纹理压缩
- [ ] 静态网格使用 `GL_STATIC_DRAW`
- [ ] 片元着色器避免动态分支和 `discard`（保留 Early-Z）
- [ ] 使用 Z-Prepass 减少 Overdraw
- [ ] 动态数据使用 PBO / Persistent Mapped Buffer
- [ ] 使用 Compute Shader 替代 CPU 端大规模并行计算
- [ ] 定期使用 RenderDoc / Nsight 分析帧瓶颈
