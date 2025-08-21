set(TARGET_NAME doodle)
file(GLOB_RECURSE TARGET_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE TARGET_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(${TARGET_NAME} ${TARGET_SOURCES} ${TARGET_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "d")

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

# target_compile_definitions(${TARGET_NAME})

# target_compile_options(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC spdlog::spdlog)

# target_link_options(${TARGET_NAME})
