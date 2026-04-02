set(EXTERNAL_DIR "External")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(NOT TARGET EnTT)
    add_subdirectory(entt)
    set_target_properties(EnTT PROPERTIES FOLDER ${EXTERNAL_DIR}/entt)
endif()

if(NOT TARGET glm)
    option(GLM_ENABLE_CXX_20 "" ON)
    add_subdirectory(glm)
    set_target_properties(glm PROPERTIES FOLDER ${EXTERNAL_DIR}/glm)
    target_compile_definitions(glm INTERFACE GLM_ENABLE_EXPERIMENTAL)
endif()

if(NOT TARGET spdlog)
    option(SPDLOG_BUILD_EXAMPLE "" OFF)
    option(SPDLOG_INSTALL "" OFF)
    option(SPDLOG_USE_STD_FORMAT "" ON)
    add_subdirectory(spdlog)
    set_target_properties(spdlog PROPERTIES FOLDER ${EXTERNAL_DIR}/spdlog)
endif()

if (NOT TARGET volk)
    set(VULKAN_HEADERS_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers)
    add_subdirectory(volk)
    set_target_properties(volk PROPERTIES FOLDER ${EXTERNAL_DIR}/volk)
endif ()

if (NOT TARGET Vulkan-Headers)
    add_subdirectory(Vulkan-Headers)
    set_target_properties(Vulkan-Headers PROPERTIES FOLDER ${EXTERNAL_DIR}/Vulkan-Headers)
endif()
