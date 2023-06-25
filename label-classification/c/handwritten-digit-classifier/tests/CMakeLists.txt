add_executable(matrix-test matrix_test.c)
add_executable(mnist-test mnist_test.c)

target_link_libraries(matrix-test PUBLIC hdclib)
target_link_libraries(mnist-test PUBLIC hdclib)
