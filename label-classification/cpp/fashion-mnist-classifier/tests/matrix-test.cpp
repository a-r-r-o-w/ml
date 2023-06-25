#include <iostream>

#include "testing.hpp"
#include "matrix.hpp"

int main () {
  fmc::matrix <int> m (2, 2, {{1, 2}, {3, 4}});
  fmc::matrix <int> n = m;
  fmc::matrix <int> r;
  fmc::matrix <long double> s (5, 5, 0.3);

  TEST("construction with default value", s[4][2] == (long double)0.3);

  n *= 8;
  m *= 16;
  m /= 2;
  TEST("equality after simple operations", n == m);

  m = fmc::matrix <int> (2, 2, {{6, 9}, {4, 2}});
  n = fmc::matrix <int> (2, 2, {{2, -9}, {-4, 6}});
  r = m * n;
  r /= 6 * 2 - 9 * 4;
  TEST("product with inverse matrix", r == fmc::matrix <int> (2, 2, {{1, 0}, {0, 1}}));

  r = fmc::matrix <int> (2, 2, {{8, 0}, {0, 8}});
  TEST("addition of two matrices", m + n == r);

  TEST("subtraction of two matrices", n - r == -m);

  n([] (int x) { return std::max(x, 0) + x; });
  r = fmc::matrix <int> (2, 2, {{4, -9}, {-4, 12}});
  TEST("applying a function on matrix", n == r);

  test_stats();

  return 0;
}
