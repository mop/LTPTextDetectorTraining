project(adaboost)
cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED program_options filesystem system serialization timer iostreams)


## Compiler flags
if(DEBUG)
    set(CMAKE_CXX_FLAGS "-g -fpermissive -std=gnu++11 -march=native -Wall -Wpedantic")        
else()
    set(CMAKE_CXX_FLAGS "-O3  -ffast-math -fpermissive -std=gnu++11 -march=native -Wall -fopenmp -Wpedantic")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${OpenCV_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

IF (${EIGEN3_INCLUDE_DIR} MATCHES "OFF")
    SET(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
ENDIF(${EIGEN3_INCLUDE_DIR} MATCHES "OFF")

get_filename_component(mydir "." ABSOLUTE)
include_directories(${mydir} ${EIGEN3_INCLUDE_DIR})

file(GLOB boost_src "src/*.cpp")
list(REMOVE_ITEM boost_src "src/main.cpp")

add_library(adaboost SHARED ${boost_src})
target_link_libraries(adaboost ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(rf src/main.cpp)

target_link_libraries(rf ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} adaboost)

option(WITH_TESTS "build tests" ON)
option(DEBUG "debugging mode" OFF)
option(EIGEN3_INCLUDE_DIR "Eigen3 Include Directory" "/usr/include/eigen3")

if (WITH_TESTS)
    add_subdirectory(tests)
endif()
