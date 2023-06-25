#ifndef HDC_UTILS_H
#define HDC_UTILS_H

long double sigmoid            (long double);
long double sigmoid_derivative (long double);

long double square_error            (long double, long double);
long double square_error_derivative (long double, long double);

int argmax (long double*, int);

long double random_value ();

bool file_exists (const char*);

#endif // HDC_UTILS_H
