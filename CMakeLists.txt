cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status")
message(STATUS " Version: ${OpenCV_VERSION}")	
add_executable(launch refineDet.cpp post_process.cpp)
link_directories("/home/qlt/qiulingteng/app/app-example/refinedet")

target_link_libraries(launch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" "/home/qlt/qiulingteng/app/app-example/refinedet/libnms_gpu.so")
set_property(TARGET launch PROPERTY CXX_STANDARD 11)
