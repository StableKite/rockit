target_include_directories(rockit_shared INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sdk/include>
)
target_include_directories(rockit_static INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sdk/include>
)

set(_rockit_tgi_config_dir "${CMAKE_CURRENT_SOURCE_DIR}/sdk/conf")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_rockit_ai_camera_config "${_rockit_tgi_config_dir}/arch64/aicamera_uvc_zoom.json")
else()
    set(_rockit_ai_camera_config "${_rockit_tgi_config_dir}/aicamera_uvc_zoom.json")
endif()

install(FILES "${_rockit_ai_camera_config}"
    DESTINATION "share/aiserver"
    RENAME aicamera.json)
install(FILES "${_rockit_tgi_config_dir}/aisingle.json"
    DESTINATION "share/aiserver")
install(DIRECTORY sdk/include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
