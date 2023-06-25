#ifndef HDC_MATRIX_H
#define HDC_MATRIX_H

#include <stdbool.h>

typedef struct matrix matrix_t;

struct matrix {
  long double* values;
  int rows;
  int cols;
};

static const int epsilon = 1e-9;

void           matrix_scalar_add        (matrix_t*, long double);
void           matrix_add               (matrix_t*, const matrix_t*);
void           matrix_apply             (matrix_t*, long double (*) (long double));
void           matrix_apply_from        (const matrix_t*, matrix_t*, long double (*) (long double));
void           matrix_assign            (const matrix_t*, matrix_t*);
void           matrix_constructor       (matrix_t*, int, int, long double);
void           matrix_copy              (const matrix_t*, matrix_t*);
void           matrix_destructor        (matrix_t*);
void           matrix_dot               (matrix_t*, const matrix_t*, const matrix_t*);
void           matrix_fill              (matrix_t*, long double);
long double*   matrix_flatten           (const matrix_t*);
void           matrix_fromarray_1D      (matrix_t*, const long double*, int, int);
void           matrix_fromarray_2D      (matrix_t*, const long double**, int, int);
long double    matrix_get               (const matrix_t*, int, int);
int            matrix_getindex          (const matrix_t*, int, int);
bool           matrix_isequal           (const matrix_t*, const matrix_t*);
void           matrix_print             (const matrix_t*);
void           matrix_scale             (matrix_t*, long double);
void           matrix_scalar_subtract   (matrix_t*, long double);
void           matrix_set               (matrix_t*, int, int, long double);
void           matrix_subtract          (matrix_t*, const matrix_t*);
void           matrix_transpose         (const matrix_t*, matrix_t*);

#endif // HDC_MATRIX_H
