cmake_minimum_required(VERSION 3.0)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

include_directories(${PROJECT_SOURCE_DIR}/extern_libs/faiss)
link_directories(/home/yiqian/hyq_SIGMOD/extern_libs/faiss/build/faiss)


add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)
include_directories(${PROJECT_SOURCE_DIR}/extern_libs/faiss)
link_directories(/home/yiqian/efanna_graph/extern_libs/faiss/build/faiss)
add_subdirectory(src)
add_subdirectory(tests)


