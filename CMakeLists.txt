cmake_minimum_required(VERSION 3.16)
project(rockit VERSION 0.36.0 LANGUAGES C CXX)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_ROCKIT_MPI "Build and install the Rockit MPI interface" ON)
option(USE_ROCKIT_TGI "Install the Rockit TGI interface and configuration" ON)
option(ROCKIT_BUILD_EXAMPLES "Build MPI examples" ON)

function(rockit_enable_strict_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>
    )
endfunction()

function(rockit_enable_example_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Werror>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Werror>
    )
endfunction()

set(ROCKIT_RECONSTRUCTION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/reconstruction")
set(ROCKIT_CORE_SOURCES
    reconstruction/src/rockit_core.cpp
    reconstruction/src/local_fault_injection.cpp
    reconstruction/src/generated_mpi_stubs.cpp
    reconstruction/src/runtime_thread_pool.cpp
    reconstruction/src/runtime_executor.cpp
    reconstruction/src/runtime_scheduler.cpp
    reconstruction/src/runtime_task_graph.cpp
    reconstruction/src/runtime_task_node.cpp
    reconstruction/src/video_channel_lifecycle.cpp
    reconstruction/src/mpp_video_provider.cpp
    reconstruction/src/local_video_io.cpp
    reconstruction/src/local_frame_cpu.cpp
    reconstruction/src/local_image_modules.cpp
    reconstruction/src/vendor_cal_contracts.cpp
    reconstruction/src/local_audio.cpp
)

add_library(rockit_objects OBJECT ${ROCKIT_CORE_SOURCES})
set_target_properties(rockit_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(rockit_objects PRIVATE
    "${ROCKIT_RECONSTRUCTION_DIR}/include"
    "${ROCKIT_RECONSTRUCTION_DIR}/src"
)
rockit_enable_strict_warnings(rockit_objects)

add_library(rockit_shared SHARED $<TARGET_OBJECTS:rockit_objects>)
add_library(Rockit::rockit ALIAS rockit_shared)
set_target_properties(rockit_shared PROPERTIES OUTPUT_NAME rockit EXPORT_NAME rockit)
target_include_directories(rockit_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mpi/sdk/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tgi/sdk/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rockit_shared PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(UNIX AND NOT APPLE)
    target_link_options(rockit_shared PRIVATE
        "LINKER:--version-script=${ROCKIT_RECONSTRUCTION_DIR}/abi/librockit.map"
    )
endif()

add_library(rockit_static STATIC $<TARGET_OBJECTS:rockit_objects>)
add_library(Rockit::rockit_static ALIAS rockit_static)
set_target_properties(rockit_static PROPERTIES OUTPUT_NAME rockit EXPORT_NAME rockit_static)
target_include_directories(rockit_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mpi/sdk/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tgi/sdk/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rockit_static PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

function(rockit_add_feature_library target output_name primary_source shim_sources map_file)
    add_library(${target} SHARED ${primary_source} ${shim_sources})
    target_include_directories(${target} PRIVATE
        "${ROCKIT_RECONSTRUCTION_DIR}/feature/include"
        "${ROCKIT_RECONSTRUCTION_DIR}/feature/src"
    )
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${output_name}")
    rockit_enable_strict_warnings(${target})
    if(UNIX AND NOT APPLE)
        target_link_options(${target} PRIVATE
            "LINKER:--version-script=${ROCKIT_RECONSTRUCTION_DIR}/feature/abi/${map_file}"
        )
    endif()
endfunction()

rockit_add_feature_library(rockit_swcac RkSwCac
    reconstruction/feature/src/feature_swcac.c
    "reconstruction/feature/src/shims_RkSwCac.c;reconstruction/feature/src/feature_special_crt_exports.c"
    libRkSwCac.so.map)
target_link_options(rockit_swcac PRIVATE -nostartfiles)

rockit_add_feature_library(rockit_graphic_lsf graphic_lsf
    reconstruction/feature/src/feature_graphic_lsf.c
    "reconstruction/feature/src/shims_graphic_lsf.c;reconstruction/feature/src/feature_special_crt_exports.c"
    libgraphic_lsf.so.map)
target_link_options(rockit_graphic_lsf PRIVATE -nostartfiles)

rockit_add_feature_library(rockit_pano_stitch panoStitchApp
    reconstruction/feature/src/feature_pano.c
    reconstruction/feature/src/shims_panoStitchApp.c
    libpanoStitchApp.so.map)
rockit_add_feature_library(rockit_dis rkAlgoDis
    reconstruction/feature/src/feature_dis.c
    reconstruction/feature/src/shims_rkAlgoDis.c
    librkAlgoDis.so.map)
rockit_add_feature_library(rockit_gdc rk_gdc_core
    reconstruction/feature/src/feature_gdc.c
    reconstruction/feature/src/shims_rk_gdc_core.c
    librk_gdc_core.so.map)
rockit_add_feature_library(rockit_avs rkgfx_avs
    reconstruction/feature/src/feature_avs.c
    reconstruction/feature/src/shims_rkgfx_avs.c
    librkgfx_avs.so.map)

set(ROCKIT_LIBRARY_TARGETS
    rockit_shared
    rockit_static
    rockit_swcac
    rockit_graphic_lsf
    rockit_pano_stitch
    rockit_dis
    rockit_gdc
    rockit_avs
)

if(USE_ROCKIT_MPI)
    add_subdirectory(mpi)
endif()
if(USE_ROCKIT_TGI)
    add_subdirectory(tgi)
endif()

install(TARGETS ${ROCKIT_LIBRARY_TARGETS}
    EXPORT RockitTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(EXPORT RockitTargets
    FILE RockitTargets.cmake
    NAMESPACE Rockit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Rockit
)
install(DIRECTORY reconstruction/feature/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
