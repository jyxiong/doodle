file(GLOB SOURCES *.cpp)

foreach(SOURCE IN LISTS SOURCES)
    get_filename_component(NAME ${SOURCE} NAME_WE)
    set(TARGET_NAME "algo_${NAME}")
    add_executable(${TARGET_NAME} ${SOURCE})
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
    message(STATUS "[algorithm] 注册 target: ${TARGET_NAME}")
endforeach()
