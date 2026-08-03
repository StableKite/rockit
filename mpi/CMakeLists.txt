target_include_directories(rockit_shared INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sdk/include>
)
target_include_directories(rockit_static INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sdk/include>
)

add_subdirectory(sdk)
if(ROCKIT_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
