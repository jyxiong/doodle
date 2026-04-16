cmake_minimum_required(VERSION 3.24)

# ── 检测 CUDA 是否可用 ──────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "[reduce] CUDA 编译器未找到，跳过 reduce tutorial 构建")
    return()
endif()

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)


# ── 收集所有 reduce_v*.cu 文件并分别编译为可执行文件 ──────────
file(GLOB REDUCE_SOURCES tutorial/reduce_v*.cu)

foreach(SOURCE IN LISTS REDUCE_SOURCES)
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)
    add_executable(${TARGET_NAME} ${SOURCE})
    target_include_directories(${TARGET_NAME} PRIVATE tutorial)
    # 开启 O3 优化以体现各版本的真实性能差距
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
    message(STATUS "[reduce] 注册 target: ${TARGET_NAME}")
endforeach()
