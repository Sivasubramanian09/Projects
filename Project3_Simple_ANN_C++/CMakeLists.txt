cmake_minimum_required(VERSION 3.20)

project(Net)

set(CMAKE_PREFIX_PATH /home/siva/cpp/libtorch)

find_package(Torch REQUIRED)
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cpp main.h declarations.cpp pima-indians-diabetes.data.csv
                )

target_compile_definitions(main PRIVATE DELIMITER=",")

target_include_directories(main PRIVATE)

target_link_libraries(main "${TORCH_LIBRARIES}")

set_property(TARGET main  PROPERTY CXX_STANDARD 17)
