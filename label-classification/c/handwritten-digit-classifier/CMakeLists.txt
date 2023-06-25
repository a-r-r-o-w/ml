cmake_minimum_required(VERSION 3.0)
project(handwritten-digit-classifier DESCRIPTION "Handwritten Digit Classifier in C"
                                     LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic -O3 -lm -fsanitize=address,undefined")

include_directories(include)

add_subdirectory(include)
add_subdirectory(tests)
add_subdirectory(src)
