cmake_minimum_required(VERSION 3.0)
project(same-input-output-nn)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic -lm -fsanitize=address,undefined")

add_executable(main src/main.c)
add_executable(rough src/rough.c)
