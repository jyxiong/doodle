set(EXTERNAL_FOLDER "ThirdParty")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(NOT TARGET spdlog)
    option(SPDLOG_BUILD_EXAMPLE "" OFF)
    option(SPDLOG_INSTALL "" OFF)
    option(SPDLOG_USE_STD_FORMAT "" ON)
    add_subdirectory(spdlog)
    set_target_properties(spdlog PROPERTIES FOLDER ${EXTERNAL_FOLDER}/spdlog)
endif()


