set(RK_MPI_TEST_AO_SRC
    test_mpi_ao.cpp
)

set(RK_MPI_TEST_ADEC_SRC
    test_mpi_adec.cpp
)

# set(RK_MPI_TEST_VPSS_SRC
#     test_mpi_vpss.cpp
# )

# set(RK_MPI_TEST_AVS_SRC
#     test_mpi_avs.cpp
# )

# set(RK_MPI_TEST_VGS_SRC
#     test_mpi_vgs.cpp
# )

set(RK_MPI_TEST_MB_SRC
    test_mpi_mb.cpp
)

set(RK_MPI_TEST_SYS_SRC
    test_mpi_sys.cpp
)

# set(RK_MPI_TEST_SYS_DEV_SRC
#     sys/test_sys_device_priority.cpp
# )

set(RK_MPI_TEST_VDEC_SRC
    test_mpi_vdec.cpp
)

set(RK_MPI_TEST_SIMPLE_VDEC_SRC
    test_mpi_simple_vdec.cpp
)

# set(RK_MPI_TEST_VENC_SRC
#     test_mpi_venc.cpp
# )

# set(RK_MPI_TEST_TDE_SRC
#     test_mpi_tde.cpp
# )

set(RK_MPI_TEST_AI_SRC
    test_mpi_ai.cpp
)

# set(RK_MPI_TEST_RGN_SRC
#     test_mpi_rgn.cpp
# )

set(RK_MPI_TEST_VO_SRC
    test_mpi_vo.cpp
)

set(RK_MPI_TEST_AENC_SRC
    test_mpi_aenc.cpp
)

set(RK_MPI_TEST_AMIX_SRC
    test_mpi_amix.cpp
)

# set(RK_MPI_TEST_VI_SRC
#     test_mpi_vi.cpp
# )

set(RK_MPI_TEST_MMZ_SRC
    test_mpi_mmz.cpp
)

# set(RK_MPI_TEST_GDC_SRC
#     test_mpi_gdc.cpp
# )

# set(RK_MPI_TEST_DIS_SRC
#     test_mpi_dis.cpp
# )

set(RK_MPI_TEST_AVIO_SRC
    sys/test_sys_avio.cpp
)

set(FIREFLY_MPI_TEST_AVS_SRC
    firefly_test_mpi_avs.cpp
)

#--------------------------
# rk_mpi_ao_test
#--------------------------
if (HAVE_API_MPI_AO)
add_executable(rk_mpi_ao_test ${RK_MPI_TEST_AO_SRC} )
target_compile_options(rk_mpi_ao_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_ao_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_ao_test)
target_link_libraries(rk_mpi_ao_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_ao_test RUNTIME DESTINATION "bin/rockit")
endif()

#--------------------------
# rk_mpi_adec_test
#--------------------------
add_executable(rk_mpi_adec_test ${RK_MPI_TEST_ADEC_SRC} )
target_compile_options(rk_mpi_adec_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_adec_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_adec_test)
target_link_libraries(rk_mpi_adec_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_adec_test RUNTIME DESTINATION "bin")

# #--------------------------
# # rk_mpi_vpss_test
# #--------------------------
# add_executable(rk_mpi_vpss_test ${RK_MPI_TEST_VPSS_SRC} )
# target_link_libraries(rk_mpi_vpss_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_vpss_test RUNTIME DESTINATION "bin")

# #--------------------------
# # rk_mpi_avs_test
# #--------------------------
# add_executable(rk_mpi_avs_test ${RK_MPI_TEST_AVS_SRC} )
# target_link_libraries(rk_mpi_avs_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_avs_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_mb_test
#--------------------------
add_executable(rk_mpi_mb_test ${RK_MPI_TEST_MB_SRC} )
target_compile_options(rk_mpi_mb_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_mb_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_mb_test)
target_link_libraries(rk_mpi_mb_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_mb_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_sys_test
#--------------------------
add_executable(rk_mpi_sys_test ${RK_MPI_TEST_SYS_SRC} )
target_compile_options(rk_mpi_sys_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_sys_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_sys_test)
target_link_libraries(rk_mpi_sys_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_sys_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_vdec_test
#--------------------------
add_executable(rk_mpi_vdec_test ${RK_MPI_TEST_VDEC_SRC} )
target_compile_options(rk_mpi_vdec_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_vdec_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_vdec_test)
target_link_libraries(rk_mpi_vdec_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_vdec_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_venc_test
#--------------------------
# add_executable(rk_mpi_venc_test ${RK_MPI_TEST_VENC_SRC} )
# target_link_libraries(rk_mpi_venc_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_venc_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_tde_test
#--------------------------
# add_executable(rk_mpi_tde_test ${RK_MPI_TEST_TDE_SRC} )
# target_link_libraries(rk_mpi_tde_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_tde_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_ai_test
#--------------------------
add_executable(rk_mpi_ai_test ${RK_MPI_TEST_AI_SRC} )
target_compile_options(rk_mpi_ai_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_ai_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_ai_test)
target_link_libraries(rk_mpi_ai_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_ai_test RUNTIME DESTINATION "bin")

# rk_mpi_rgn_test
#--------------------------
# add_executable(rk_mpi_rgn_test ${RK_MPI_TEST_RGN_SRC} )
# target_link_libraries(rk_mpi_rgn_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_rgn_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_vo_test
#--------------------------
add_executable(rk_mpi_vo_test ${RK_MPI_TEST_VO_SRC} )
target_compile_options(rk_mpi_vo_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_vo_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_vo_test)
target_link_libraries(rk_mpi_vo_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_vo_test RUNTIME DESTINATION "bin")

# rk_mpi_aenc_test
#--------------------------
add_executable(rk_mpi_aenc_test ${RK_MPI_TEST_AENC_SRC} )
target_compile_options(rk_mpi_aenc_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_aenc_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_aenc_test)
target_link_libraries(rk_mpi_aenc_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_aenc_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_vi_test
#--------------------------
# add_executable(rk_mpi_vi_test ${RK_MPI_TEST_VI_SRC} )
# target_link_libraries(rk_mpi_vi_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_vi_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_mmz_test
#--------------------------
add_executable(rk_mpi_mmz_test ${RK_MPI_TEST_MMZ_SRC} )
target_compile_options(rk_mpi_mmz_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_mmz_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_mmz_test)
target_link_libraries(rk_mpi_mmz_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_mmz_test RUNTIME DESTINATION "bin")

# #--------------------------
# # rk_mpi_gdc_test
# #--------------------------
# add_executable(rk_mpi_gdc_test ${RK_MPI_TEST_GDC_SRC} )
# target_link_libraries(rk_mpi_gdc_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_gdc_test RUNTIME DESTINATION "bin")

#--------------------------
# rk_mpi_avio_test
#--------------------------
add_executable(rk_mpi_avio_test ${RK_MPI_TEST_AVIO_SRC} )
target_compile_options(rk_mpi_avio_test PRIVATE -fno-rtti)
target_compile_definitions(rk_mpi_avio_test PRIVATE ANDROID_STL=c++_shared)
rockit_enable_example_warnings(rk_mpi_avio_test)
target_link_libraries(rk_mpi_avio_test Rockit::rockit rt_test_comm Threads::Threads)
install(TARGETS rk_mpi_avio_test RUNTIME DESTINATION "bin")

#--------------------------
# firefly_mpi_avs_test
#--------------------------
#add_executable(firefly_mpi_avs_test ${FIREFLY_MPI_TEST_AVS_SRC} )
#target_link_libraries(firefly_mpi_avs_test Rockit::rockit rt_test_comm Threads::Threads)
#install(TARGETS firefly_mpi_avs_test RUNTIME DESTINATION "bin")

#--------------------------
# add 6x_rectlinear data
#--------------------------
install(DIRECTORY "6x_rectlinear/" DESTINATION "share/rockit/avs"
    PATTERN "lib" EXCLUDE)

#--------------------------
# rk_mpi_amix_test
#--------------------------
# add_executable(rk_mpi_amix_test ${RK_MPI_TEST_AMIX_SRC} )
# target_link_libraries(rk_mpi_amix_test Rockit::rockit rt_test_comm Threads::Threads)
# install(TARGETS rk_mpi_amix_test RUNTIME DESTINATION "bin")
