file(GLOB SOURCES *.cpp)

foreach(SOURCE IN LISTS SOURCES)
    get_filename_component(NAME ${SOURCE} NAME_WE)
    add_executable(${NAME} ${SOURCE})
endforeach()