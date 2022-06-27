#ifndef HDC_TEST_H
#define HDC_TEST_H

#include <iostream>
#include <string_view>

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

void test_success (const std::string_view& name, const std::string_view& test) {
  ++test_count;
  ++success_count;
  std::cout << PASSED << " (" << test_count << ": \"" << name << "\")! [" << test << "]\n";
}

void test_failure (const std::string_view& name, const std::string_view& test) {
  ++test_count;
  std::cout << FAILED << " (" << test_count << ": \"" << name << "\")! [" << test << "]\n";
}

void test_stats () {
  std::cout << "Testing complete! Passed " << success_count << " of " << test_count << " tests\n";
}

#endif // HDC_TEST_H
