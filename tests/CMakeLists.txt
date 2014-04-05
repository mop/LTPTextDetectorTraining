file(GLOB_RECURSE adaboost_test_source_files *.cpp)
add_executable(adaboost_test ${adaboost_test_source_files})

target_link_libraries(adaboost_test adaboost)

if (DEBUG)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "-g -pg -fpermissive -std=gnu++11 -march=native")        
    endif()
else()
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "-O3  -ffast-math -fpermissive -std=gnu++11 -march=native -Wall -fopenmp -Wpedantic")
    endif()
endif()


find_package(OpenCV)
target_link_libraries(adaboost_test ${OpenCV_LIBS})
