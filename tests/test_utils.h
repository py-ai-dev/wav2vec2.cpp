#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static int _tests_run = 0, _tests_failed = 0;

#define ASSERT(cond) do { \
    _tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, #cond); \
        _tests_failed++; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    _tests_run++; \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL [%s:%d] %s == %s  (%s != %s)\n", \
                __FILE__, __LINE__, #a, #b, std::to_string(a).c_str(), std::to_string(b).c_str()); \
        _tests_failed++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    _tests_run++; \
    float _a = (float)(a), _b = (float)(b); \
    if (fabsf(_a - _b) > (float)(eps)) { \
        fprintf(stderr, "FAIL [%s:%d] |%s - %s| < %s  (%.6f vs %.6f, diff=%.6f)\n", \
                __FILE__, __LINE__, #a, #b, #eps, _a, _b, fabsf(_a-_b)); \
        _tests_failed++; \
    } \
} while(0)

#define TEST_SUITE_RESULT() do { \
    if (_tests_failed == 0) \
        fprintf(stderr, "OK  %d/%d tests passed\n", _tests_run, _tests_run); \
    else \
        fprintf(stderr, "FAIL  %d/%d tests failed\n", _tests_failed, _tests_run); \
    return _tests_failed ? 1 : 0; \
} while(0)
