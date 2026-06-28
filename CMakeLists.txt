cmake_minimum_required(VERSION 3.14)
project(wav2vec2.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ───────────────────────────────────────────────────────────────────
option(WAV2VEC2_BUILD_EXAMPLES "Build CLI examples" ON)
option(WAV2VEC2_FAST_MATH      "Enable -ffast-math"  ON)
option(WAV2VEC2_USE_BLAS       "Accelerate matmul with BLAS" ON)

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(-O3 -Wall -Wextra)
if(WAV2VEC2_FAST_MATH)
    add_compile_options(-ffast-math)
endif()

# ARM NEON (Cortex-X925 / A725)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-march=armv8.2-a+fp16)
endif()

# x86 AVX2
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-mavx2 -mfma)
endif()

# ── Library ───────────────────────────────────────────────────────────────────
add_library(wav2vec2 STATIC src/wav2vec2.cpp)
target_include_directories(wav2vec2
    PUBLIC  include/
    PRIVATE src/)

# libm for sqrtf / expf / logf
target_link_libraries(wav2vec2 PUBLIC m)

# ── Threading ─────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
target_link_libraries(wav2vec2 PUBLIC Threads::Threads)

# ── Optional BLAS ─────────────────────────────────────────────────────────────
if(WAV2VEC2_USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
        target_link_libraries(wav2vec2 PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(wav2vec2 PRIVATE WV2_USE_BLAS)
    else()
        message(STATUS "BLAS not found — using naive matmul fallback")
    endif()
endif()

# ── CLI ───────────────────────────────────────────────────────────────────────
if(WAV2VEC2_BUILD_EXAMPLES)
    add_executable(wav2vec2-cli examples/main/main.cpp)
    target_include_directories(wav2vec2-cli PRIVATE src/)
    target_link_libraries(wav2vec2-cli PRIVATE wav2vec2)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
option(WAV2VEC2_BUILD_TESTS "Build unit tests" ON)
if(WAV2VEC2_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS wav2vec2 wav2vec2-cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(FILES include/wav2vec2.h DESTINATION include)
