#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/stat.h>

#include "utils.h"

long double sigmoid (long double x) {
  long double r = 1;
  r /= 1 + exp2l(-x);
  return r;
}

long double sigmoid_derivative (long double x) {
  long double r = sigmoid(x);
  return r * (1 - r);
}

long double square_error (long double x, long double y) {
  long double z = x - y;
  return z * z;
}

long double square_error_derivative (long double x, long double y) {
  return 2 * (x - y);
}

// helper function to find the index of the maximum value in an array
int argmax (long double* array, int size) {
  int max_index = 0;
  for (int i = 0; i < size; ++i)
    if (array[max_index] < array[i])
      max_index = i;
  return max_index;
}

// helper function to return a random value in range [-1, 1]
// note: produces only RAND_MAX different values
long double random_value () {
  long double r = rand();
  r /= RAND_MAX;
  r = 2 * r - 1;
  return r;
}

bool file_exists (const char* filename) {
  struct stat buffer;
  return stat(filename, &buffer) == 0;
}
