#include <stdio.h>
#include <stdlib.h>

#include "test.h"
#include "matrix.h"

long double f1 (long double x) { return 16 * x; }
long double f2 (long double x) { return x / 8; }

int main () {
  matrix_t m, n, p, q, r;

  matrix_constructor(&m, 3, 4, -1);
  matrix_constructor(&n, 3, 4, 0);
  TEST("default matrix_constructor value", matrix_get(&m, 2, 2) == -1);
  TEST("default matrix_constructor value", matrix_get(&n, 2, 3) != 1);

  matrix_fill(&n, -1);
  TEST("matrix equality", matrix_isequal(&m, &n));

  matrix_apply(&m, f1);
  matrix_scale(&n, 16.0);
  TEST("matrix equality after apply and scale", matrix_isequal(&m, &n));

  matrix_destructor(&m);
  matrix_destructor(&n);

  long double x[] = {6, 9, 4, 2};
  long double* y[2];
  long double z[] = {1, 0, 0, 1};
  y[0] = (long double*)malloc(2 * sizeof(long double));
  y[1] = (long double*)malloc(2 * sizeof(long double));
  y[0][0] = 2; y[0][1] = -9; y[1][0] = -4; y[1][1] = 6;
  matrix_fromarray_1D(&m, x, 2, 2);
  matrix_fromarray_2D(&n, (const long double**)y, 2, 2);
  free(y[0]);
  free(y[1]);
  matrix_fromarray_1D(&p, z, 2, 2);
  matrix_scale(&p, -24.0);
  matrix_constructor(&r, 2, 2, 0);
  matrix_dot(&r, &m, &n);
  TEST("matrix dot product", matrix_isequal(&p, &r));

  matrix_add(&m, &n);
  matrix_apply(&m, f2);
  matrix_fromarray_1D(&q, z, 2, 2);
  TEST("matrix addition", matrix_isequal(&m, &q));

  matrix_destructor(&r);
  matrix_destructor(&q);
  matrix_destructor(&p);
  matrix_destructor(&n);
  matrix_destructor(&m);

  test_stats();

  return 0;
}
