#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist.h"

int main () {
  // seed the random number generator
  srand(time(0));

  // initialise the mnist dataset
  mnist_t dataset;
  mnist_constructor(&dataset);
  mnist_load(&dataset);

  puts("5 Random Training Dataset Images\n");

  for (int i = 0; i < 5; ++i) {
    mnist_display_random_train_image(&dataset);
    puts("");
  }

  puts("5 Random Testing Dataset Images\n");

  for (int i = 0; i < 5; ++i) {
    mnist_display_random_test_image(&dataset);
    puts("");
  }

  // cleanup
  mnist_destructor(&dataset);

  return 0;
}
