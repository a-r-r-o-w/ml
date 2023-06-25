#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "mnist.h"
#include "nn.h"
#include "utils.h"

static void init (network_t* network, mnist_t* mnist) {
  // seed the random number generator
  srand(time(0));

  // initialise the mnist dataset
  mnist_constructor(mnist);
  mnist_load(mnist);
  mnist_normalize(mnist);

  // setup our neural network
  // the neural network contains 4 layers - 1 input, 2 hidden, 1 output
  network_constructor(network, 4, 0.01, square_error, square_error_derivative);
  network_add(network, MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT, sigmoid, sigmoid_derivative);
  network_add(network, 16, sigmoid, sigmoid_derivative);
  network_add(network, 16, sigmoid, sigmoid_derivative);
  network_add(network, 10, sigmoid, sigmoid_derivative);
}

static void train () {
  network_t network;
  mnist_t mnist;

  init(&network, &mnist);

  // randomise the weights and biases of the neural network
  network_randomize(&network);

  // train the neural network on the training dataset and save it
  network_train(&network, &mnist, 5);
  network_save(&network, "hdc.model");

  // cleanup
  network_destructor(&network);
  mnist_destructor(&mnist);
}

static void test () {
  network_t network;
  mnist_t mnist;

  init(&network, &mnist);

  // randomise the weights and biases of the neural network
  network_randomize(&network);

  // load the saved neural network
  network_load(&network, "hdc.model");

  int correct_predictions = network_test(&network, &mnist);
  printf("Accuracy: %.2Lf%%\n", (long double)100 * correct_predictions / mnist.test_size);

  // cleanup
  network_destructor(&network);
  mnist_destructor(&mnist);
}

static void usage () {
  puts("Usage: ./handwritten-digit-classifier [train|test]");
  exit(0);
}

int main (int argc, char* argv[]) {
  if (argc != 2)
    usage();

  int n = strlen(argv[1]);

  if (n == 4 && strncmp(argv[1], "test", 4) == 0)
    test();
  else if (n == 5 && strncmp(argv[1], "train", 5) == 0)
    train();
  else
    usage();

  return 0;
}
