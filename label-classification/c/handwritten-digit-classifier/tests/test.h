#ifndef HDC_TEST_H
#define HDC_TEST_H

#include <stdio.h>

#if defined(__linux__) || defined(__APPLE__)
#  define PASSED "\033[1;32mPASSED\033[0m"
#  define FAILED "\033[1;31mFAILED\033[0m"
#else
#  define PASSED "PASSED\n"
#  define FAILED "FAILED\n"
#endif

#define TEST(name, test) { if (test) test_success(name, #test); else test_failure(name, #test); }

static int test_count = 0;
static int success_count = 0;

void test_success (const char* name, const char* test) {
  ++test_count;
  ++success_count;
  printf("%s (%d: \"%s\")! [%s]\n", PASSED, test_count, name, test);
}

void test_failure (const char* name, const char* test) {
  ++test_count;
  printf("%s (%d \"%s\")! [%s]\n", FAILED, test_count, name, test);
}

void test_stats () {
  printf("Testing complete! Passed %d of %d tests\n", success_count, test_count);
}

#endif // HDC_TEST_H
