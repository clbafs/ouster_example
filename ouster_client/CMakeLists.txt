cmake_minimum_required(VERSION 3.1.0)

# ==== Project Name ====
project(ouster_client)

# ==== Requirements ====
set(OUSTER_CLIENT_ADDITIONAL_LIBS "")
set(OUSTER_CLIENT_ADDITIONAL_LIB_DIRS "")
find_package(PkgConfig REQUIRED)
pkg_check_modules(jsoncpp REQUIRED jsoncpp)
set(OUSTER_CLIENT_ADDITIONAL_LIBS jsoncpp)
add_compile_options(-Wextra -Werror)

find_package(Eigen3 REQUIRED)

# ==== Options ====
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall)

# ==== Libraries ====
add_library(ouster_client STATIC src/client.cpp src/types.cpp src/compat.cpp src/lidar_scan.cpp)
set_target_properties(ouster_client PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ouster_client PUBLIC ${OUSTER_CLIENT_ADDITIONAL_LIBS})
target_include_directories(ouster_client
                           PUBLIC include ${OUSTER_CLIENT_ADDITIONAL_INCLUDES})
target_include_directories(
  ouster_client SYSTEM
  PUBLIC ${jsoncpp_INCLUDE_DIRS} SYSTEM
  PRIVATE ${EIGEN3_INCLUDE_DIR})

install(TARGETS ouster_client
        EXPORT ouster_clientTargets
        LIBRARY DESTINATION lib  # 动态库安装路径
        ARCHIVE DESTINATION lib  # 静态库安装路径
        RUNTIME DESTINATION bin  # 可执行文件安装路径
        PUBLIC_HEADER DESTINATION include  # 头文件安装路径
        )
install(
        FILES
        include/ouster/impl/client_impl.h
        include/ouster/impl/packet_impl.h
        include/ouster/client.h
        include/ouster/compat.h
        include/ouster/lidar_scan.h
        include/ouster/packet.h
        include/ouster/types.h
        include/ouster/version.h
        DESTINATION include/ouster

)

# ==== Executables ====
#add_executable(ouster_client_example src/main.cpp)
#target_link_libraries(ouster_client_example ouster_client)
#target_include_directories(ouster_client_example PRIVATE include)

