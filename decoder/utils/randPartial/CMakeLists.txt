cmake_minimum_required(VERSION 3.4...3.18)
project(randpartial)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(pybind11)
# find_package(pybind11 REQUIRED)
include_directories(/usr/include/python3.6m)
# include_directories(/opt/conda/include/python3.8)
find_package(PCL REQUIRED)

include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(randpartial main.cpp)
# set_target_properties(randpartial PROPERTIES PREFIX "")
# set_target_properties(randpartial PROPERTIES SUFFIX ".so")
target_link_libraries(randpartial PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(randpartial PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})