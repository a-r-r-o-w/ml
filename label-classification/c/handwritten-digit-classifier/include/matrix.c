#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "matrix.h"
#include "error_handler.h"

void matrix_scalar_add (matrix_t* m, long double constant) {
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] += constant;
}

void matrix_add (matrix_t* m, const matrix_t* n) {
#ifdef DEBUG_MODE
  if (m->rows != n->rows || m->cols != n->cols)
    error("incompatible matrices for addition");
#endif
  
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] += n->values[i];
}

void matrix_apply (matrix_t* m, long double (*apply_func) (long double)) {
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] = apply_func(m->values[i]);
}

void matrix_apply_from (const matrix_t* src, matrix_t* dst, long double (*apply_func)(long double)) {
#ifdef DEBUG_MODE
  if (src->rows != dst->rows || src->cols != dst->cols)
    error("incompatible matrices for assignment");
#endif
  for (int i = 0; i < src->rows * src->cols; ++i)
    dst->values[i] = apply_func(src->values[i]);
}

void matrix_assign (const matrix_t* src, matrix_t* dst) {
#ifdef DEBUG_MODE
  if (src->rows != dst->rows || src->cols != dst->cols)
    error("incompatible matrices for assignment");
#endif
  
  for (int i = 0; i < src->rows * src->cols; ++i)
    dst->values[i] = src->values[i];
}

void matrix_constructor (matrix_t* m, int rows, int cols, long double default_value) {
#ifdef DEBUG_MODE
  if (rows < 0 || cols < 0)
    error("matrix dimensions must be positive");
#endif
  
  m->rows = rows;
  m->cols = cols;
  
  int x = m->rows * m->cols;
  m->values = (long double*)malloc(x * sizeof(long double));
  for (int i = 0; i < x; ++i)
    m->values[i] = default_value;
}

void matrix_copy (const matrix_t* src, matrix_t* dst) {
  dst->rows = src->rows;
  dst->cols = src->cols;
  
  int x = dst->rows * dst->cols;
  dst->values = (long double*)malloc(x * sizeof(long double));
  for (int i = 0; i < x; ++i)
    dst->values[i] = src->values[i];
}

void matrix_destructor (matrix_t* m) {
  m->rows = 0;
  m->cols = 0;
  free(m->values);
}

void matrix_dot (matrix_t* r, const matrix_t* m, const matrix_t* n) {
#ifdef DEBUG_MODE
  if (m->cols != n->rows || r->rows != m->rows || r->cols != n->cols)
    error("matrices are incompatible for dot product");
#endif
  
  for (int i = 0; i < m->rows; ++i)
    for (int j = 0; j < n->cols; ++j) {
      int x = matrix_getindex(r, i, j);
      r->values[x] = 0;

      for (int k = 0; k < m->cols; ++k) {
        int y = matrix_getindex(m, i, k);
        int z = matrix_getindex(n, k, j);
        r->values[x] += m->values[y] * n->values[z];
      }
    }
}

void matrix_fill (matrix_t* m, long double value) {
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] = value;
}

long double* matrix_flatten (const matrix_t* m) {
  return m->values;
}

void matrix_fromarray_1D (matrix_t* m, const long double* values, int rows, int cols) {
  matrix_constructor(m, rows, cols, 0);
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] = values[i];
}

void matrix_fromarray_2D (matrix_t* m, const long double** values, int rows, int cols) {
  matrix_constructor(m, rows, cols, 0);
  int index = 0;
  for (int i = 0; i < m->rows; ++i)
    for (int j = 0; j < m->cols; ++j)
      m->values[index++] = values[i][j];
}

long double matrix_get (const matrix_t* m, int row, int col) {
  return m->values[matrix_getindex(m, row, col)];
}

int matrix_getindex (const matrix_t* m, int row, int col) {
  int index = m->cols * row + col;

#ifdef DEBUG_MODE
  if (index > m->rows * m->cols)
    error("out of bounds index");
#endif

  return index;
}

bool matrix_isequal (const matrix_t* m, const matrix_t* n) {
  if (m->rows != n->rows || m->cols != n->cols)
    return false;
  for (int i = 0; i < m->rows * m->cols; ++i)
    if (fabsl(m->values[i] - n->values[i]) > epsilon)
      return false;
  return true;
}

void matrix_print (const matrix_t* m) {
  for (int i = 0; i < m->rows * m->cols; ++i) {
    if (i > 0 && i % m->cols == 0)
      puts("");
    printf("%.10Lf ", m->values[i]);
  }
  puts("");
}

void matrix_scale (matrix_t* m, long double x) {
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] *= x;
}

void matrix_scalar_subtract (matrix_t* m, long double x) {
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] -= x;
}

void matrix_set (matrix_t* m, int row, int col, long double x) {
  m->values[matrix_getindex(m, row, col)] = x;
}

void matrix_subtract (matrix_t* m, const matrix_t* n) {
#ifdef DEBUG_MODE
  if (m->rows != n->rows || m->cols != n->cols)
    error("incompatible matrices for addition");
#endif
  
  for (int i = 0; i < m->rows * m->cols; ++i)
    m->values[i] -= n->values[i];
}

void matrix_transpose (const matrix_t* m, matrix_t* t) {
  matrix_constructor(t, m->cols, m->rows, 0);
  int index = 0;
  for (int i = 0; i < m->rows; ++i)
    for (int j = 0; j < m->cols; ++j)
      t->values[matrix_getindex(t, j, i)] = m->values[index++];
}
