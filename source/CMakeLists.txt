set(TARGET_NAME doodle)
file(GLOB_RECURSE TARGET_SOURCES "doodle/*.cpp")
file(GLOB_RECURSE TARGET_HEADERS "doodle/*.h")

add_library(${TARGET_NAME} ${TARGET_SOURCES} ${TARGET_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# target_compile_definitions(${TARGET_NAME})

# target_compile_options(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC EnTT glm spdlog::spdlog Vulkan::Headers volk)

# target_link_options(${TARGET_NAME})
