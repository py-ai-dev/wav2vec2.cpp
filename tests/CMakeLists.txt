enable_testing()

add_executable(test_ops  test_ops.cpp)
add_executable(test_gguf test_gguf.cpp)

target_include_directories(test_ops  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_gguf PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_test(NAME ops  COMMAND test_ops)
add_test(NAME gguf COMMAND test_gguf)
