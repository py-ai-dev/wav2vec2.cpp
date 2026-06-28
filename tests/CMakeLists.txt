enable_testing()

find_package(Threads REQUIRED)

add_executable(test_ops  test_ops.cpp)
add_executable(test_gguf test_gguf.cpp)

target_include_directories(test_ops  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_gguf PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_ops  PRIVATE Threads::Threads m)
target_link_libraries(test_gguf PRIVATE Threads::Threads m)

add_test(NAME ops  COMMAND test_ops)
add_test(NAME gguf COMMAND test_gguf)
