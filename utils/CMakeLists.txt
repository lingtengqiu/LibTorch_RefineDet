cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status")
message(STATUS " Version: ${OpenCV_VERSION}")	
#add_executable(launch nms.cpp)
add_library(nms_gpu SHARED nms.cpp)
link_directories("/home/qlt/qiulingteng/app/app-example/refinedet/utils")

target_link_libraries(nms_gpu "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" "/home/qlt/qiulingteng/app/app-example/refinedet/utils/libnms.so")
set_property(TARGET nms_gpu PROPERTY CXX_STANDARD 11)
