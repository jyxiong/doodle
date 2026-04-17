file(GLOB SOURCES *.cpp)

foreach(SOURCE IN LISTS SOURCES)
    get_filename_component(NAME ${SOURCE} NAME_WE)
    set(TARGET_NAME "container_${NAME}")
    add_executable(${TARGET_NAME} ${SOURCE})
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
    message(STATUS "[container] 注册 target: ${TARGET_NAME}")
endforeach()
