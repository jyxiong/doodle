set(TARGET_NAME doodle)
file(GLOB_RECURSE TARGET_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE TARGET_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(${TARGET_NAME} ${TARGET_SOURCES} ${TARGET_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

# target_compile_definitions(${TARGET_NAME})

# target_compile_options(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC spdlog::spdlog Vulkan::Headers volk)

# target_link_options(${TARGET_NAME})
