#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

const int training_data_size = 4096;
const int testing_data_size = 256;

long double training_data[4096];
long double testing_data[256];

long double relu (long double x) {
  return x > 0 ? x : 0;
}

long double drelu__dx (long double x) {
  return x > 0 ? 1 : 0;
}

long double mean_squared_error (long double x, long double y) {
  long double z = x - y;
  return z * z;
}

long double dmean_squared_error__dx (long double x, long double y) {
  return 2 * (x - y);
}

void generate_data () {
  for (int i = 0; i < training_data_size; ++i)
    training_data[i] = rand() % 2;
  for (int i = 0; i < testing_data_size; ++i)
    testing_data[i] = rand() % 2;
}

int main () {
  srand(time(0));

  generate_data();

  long double a0;
  long double a1;
  long double a2;
  long double z1;
  long double z2;
  long double b1 = rand() % 10;
  long double b2 = rand() % 10;
  long double w1 = rand() % 10;
  long double w2 = rand() % 10;
  long double learning_rate = 0.001;
  int epochs = 100;

  while (epochs--) {
    for (int i = 0; i < training_data_size; ++i) {
      a0 = training_data[i];
      z1 = a0 * w1 + b1;
      a1 = relu(z1);
      z2 = a1 * w2 + b2;
      a2 = relu(z2);

      long double C = mean_squared_error(a2, a0);
      long double dC_da2 = dmean_squared_error__dx(a2, a0);
      long double da2_dz2 = drelu__dx(z2);
      long double dz2_dw2 = a1;
      long double dz2_db2 = 1;
      long double dz2_da1 = w2;
      long double da1_dz1 = drelu__dx(z1);
      long double dz1_dw1 = a0;
      long double dz1_db1 = 1;
      
      long double dC_dw2 = dC_da2 * da2_dz2 * dz2_dw2;
      long double dC_db2 = dC_da2 * da2_dz2 * dz2_db2;
      long double dC_dw1 = dz1_dw1 * da1_dz1 * dz2_da1 * da2_dz2 * dC_da2;
      long double dC_db1 = dz1_db1 * da1_dz1 * dz2_da1 * da2_dz2 * dC_da2;
      // printf("dC_dw2: %Lf, dC_db2: %Lf, dC_dw1: %Lf, dC_db1: %Lf\n", dC_dw2, dC_db2, dC_dw1, dC_db1);

      w2 += learning_rate * (-dC_dw2);
      b2 += learning_rate * (-dC_db2);
      w1 += learning_rate * (-dC_dw1);
      b1 += learning_rate * (-dC_db1);
    }
  }

  printf("w1: %Lf, b1: %Lf, w2: %Lf, b2: %Lf\n", w1, b1, w2, b2);

  const long double epsilon = 1e-9;
  int correct = 0;

  for (int i = 0; i < testing_data_size; ++i) {
    a0 = testing_data[i];
    z1 = a0 * w1 + b1;
    a1 = relu(z1);
    z2 = a1 * w2 + b2;
    a2 = relu(z2);

    // printf("input: %Lf, output: %Lf\n", testing_data[i], roundl(a2));
    if (fabsl(round(a2) - testing_data[i]) < epsilon)
      ++correct;
  }

  printf("accuracy: %Lf\n", (long double)correct / testing_data_size);

  return 0;
}
