project(OBSTACLE_REG)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

find_package(OpenCV REQUIRED)

# Project Executable
add_executable (Obstacle_Reg Obstacle_Reg.cpp )
target_link_libraries(Obstacle_Reg ${OpenCV_LIBS})