cmake_minimum_required(VERSION 3.1.0)

# ==== Project Name ====
project(ouster_viz)

# ==== Requirements ====
set(OpenGL_GL_PREFERENCE GLVND)
find_package(Threads)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# ==== Options ====
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall)
find_package(glfw3 REQUIRED)
set(OUSTER_VIZ_ADDITIONAL_LIBS glfw GLEW GL)
add_compile_options(-Wextra -Werror)

# ==== Libraries ====
add_library(ouster_viz STATIC src/point_viz.cpp src/cloud.cpp src/camera.cpp
        src/image.cpp)
target_link_libraries(
        ouster_viz
        PUBLIC ${CMAKE_THREAD_LIBS_INIT}
        PRIVATE ouster_client ${OUSTER_VIZ_ADDITIONAL_LIBS})
target_include_directories(ouster_viz PUBLIC include)
target_include_directories(ouster_viz SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})

#[[# ==== Executables ====
add_executable(simple_viz src/main.cpp)
target_link_libraries(simple_viz ${CMAKE_THREAD_LIBS_INIT}
                      ${OUSTER_VIZ_ADDITIONAL_LIBS} ouster_client ouster_viz)
target_include_directories(simple_viz PRIVATE include)
target_include_directories(simple_viz SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})]]

install(TARGETS ouster_viz
        EXPORT ouster_vizTargets
        LIBRARY DESTINATION lib  # 动态库安装路径
        ARCHIVE DESTINATION lib  # 静态库安装路径
        RUNTIME DESTINATION bin  # 可执行文件安装路径
        PUBLIC_HEADER DESTINATION include  # 头文件安装路径
        )
install(
        FILES
        include/ouster/autoexposure.h
        include/ouster/beam_uniformity.h
        include/ouster/colormaps.h
        include/ouster/lidar_scan_viz.h
        include/ouster/point_viz.h
        DESTINATION include/ouster
)