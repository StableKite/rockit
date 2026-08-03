add_compile_definitions(HAVE_API_MPI_AO HAVE_API_MPI_ADEC)

add_library(rt_test_comm STATIC)
target_sources(rt_test_comm PRIVATE
    common/test_mod_vpss.cpp
    common/test_comm_argparse.cpp
    common/test_comm_utils.cpp
    common/test_comm_bmp.cpp
    common/test_comm_imgproc.cpp
    common/test_comm_sys.cpp
    common/test_comm_vdec.cpp
    common/test_comm_vo.cpp
    common/test_comm_ao.cpp
    common/tmedia/test_comm_tmd.cpp
)
target_include_directories(rt_test_comm PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/tmedia"
)
target_compile_options(rt_test_comm PRIVATE -fno-rtti)
target_compile_definitions(rt_test_comm PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rt_test_comm)
target_link_libraries(rt_test_comm PUBLIC Rockit::rockit Threads::Threads)

add_subdirectory(mod)
