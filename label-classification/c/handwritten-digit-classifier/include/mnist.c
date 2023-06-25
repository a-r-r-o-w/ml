#include <stdio.h>
#include <stdlib.h>

#include "error_handler.h"
#include "matrix.h"
#include "mnist.h"
#include "utils.h"

// Construct a mnist object and allocate necessary memory
void mnist_constructor (mnist_t* m) {
  m->train_size = TRAINING_DATASET_SIZE;
  m->test_size = TESTING_DATASET_SIZE;
  m->train_images = (matrix_t*)malloc(m->train_size * sizeof(matrix_t));
  m->test_images = (matrix_t*)malloc(m->test_size * sizeof(matrix_t));
  m->train_labels = (matrix_t*)malloc(m->train_size * sizeof(matrix_t));
  m->test_labels = (matrix_t*)malloc(m->test_size * sizeof(matrix_t));
  
  for (int i = 0; i < m->train_size; ++i)
    matrix_constructor(m->train_images + i, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH, 0);
  for (int i = 0; i < m->test_size; ++i)
    matrix_constructor(m->test_images + i, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH, 0);
  for (int i = 0; i < m->train_size; ++i)
    matrix_constructor(m->train_labels + i, 1, 10, 0);
  for (int i = 0; i < m->test_size; ++i)
    matrix_constructor(m->test_labels + i, 1, 10, 0);
}

// Destruct a mnist object and free up allocated memory
void mnist_destructor (mnist_t* m) {
  for (int i = 0; i < m->test_size; ++i)
    matrix_destructor(m->test_labels + i);
  for (int i = 0; i < m->train_size; ++i)
    matrix_destructor(m->train_labels + i);
  for (int i = 0; i < m->test_size; ++i)
    matrix_destructor(m->test_images + i);
  for (int i = 0; i < m->train_size; ++i)
    matrix_destructor(m->train_images + i);
  
  free(m->test_labels);
  free(m->train_labels);
  free(m->test_images);
  free(m->train_images);
  m->test_labels = NULL;
  m->train_labels = NULL;
  m->test_images = NULL;
  m->train_images = NULL;
  m->test_size = 0;
  m->train_size = 0;
}

// Display a mnist digit image
// Note that depending on the font used in the terminal and other terminal settings,
// the displayed image may vary in size and appearance
// The terminal you use must support ANSI Escape Codes
void mnist_display_image (matrix_t* img, int label) {
  printf("Label: %d\n", label);
  for (int i = 0; i < MNIST_IMAGE_HEIGHT; ++i) {
    for (int j = 0; j < MNIST_IMAGE_WIDTH; ++j) {
      int value = matrix_get(img, i, j);
      printf("\x1b[48;2;%d;%d;%dm  \x1b[0m", value, value, value);
    }
    puts("");
  }
}

// Display a mnist digit image from the testing dataset
void mnist_display_test_image (mnist_t* m, int index) {
#ifdef DEBUG_MODE
  if (index >= m->test_size)
    error("out of bounds access for testing dataset");
#endif

  int label = argmax(m->test_labels[index].values, 10);
  mnist_display_image(&m->test_images[index], label);
}

// Display a mnist digit image from the training dataset
void mnist_display_train_image (mnist_t* m, int index) {
#ifdef DEBUG_MODE
  if (index >= m->train_size)
    error("out of bounds access for training dataset");
#endif
  
  int label = argmax(m->train_labels[index].values, 10);
  mnist_display_image(&m->train_images[index], label);
}

// Display a random mnist digit image from the testing dataset
void mnist_display_random_test_image (mnist_t* m) {
  int r = rand() % m->test_size;
  int label = argmax(m->test_labels[r].values, 10);
  mnist_display_image(&m->test_images[r], label);
}

// Display a random mnist digit image from the training dataset
void mnist_display_random_train_image (mnist_t* m) {
  int r = rand() % m->train_size;
  int label = argmax(m->train_labels[r].values, 10);
  mnist_display_image(&m->train_images[r], label);
}

// Load the mnist dataset
void mnist_load (mnist_t* m) {
  puts("[*] Initialising from MNIST dataset");

  char mnist_train[] = "../res/datasets/mnist_train.csv";
  char mnist_test[] = "../res/datasets/mnist_test.csv";

  if (!file_exists(mnist_train))
    error("missing mnist training dataset in location res/datasets/mnist_train.csv");
  if (!file_exists(mnist_test))
    error("missing mnist training dataset in location res/datasets/mnist_test.csv");
  
  // load the mnist training dataset
  {
    FILE* stream = fopen(mnist_train, "r");
    int value;
    
    // consume the first line with column names
    while (fgetc(stream) != '\n')
      continue;
    
    // read the mnist training dataset
    for (int i = 0; i < m->train_size; ++i) {
      // the first column contains the image label
      fscanf(stream, "%d,", &value);
      m->train_labels[i].values[value] = 1;

      // the remaining columns contain image data
      for (int j = 0; j < MNIST_IMAGE_SIZE - 1; ++j) {
        fscanf(stream, "%d,", &value);
        m->train_images[i].values[j] = value;
      }
      fscanf(stream, "%d", &value);
      m->train_images[i].values[MNIST_IMAGE_SIZE - 1] = value;
    }

    fclose(stream);
  }
  
  puts("[*] Loading MNIST training dataset completed");

  // load the mnist testing dataset
  {
    FILE* stream = fopen(mnist_test, "r");
    int value;
    
    // consume the first line with column names
    while (fgetc(stream) != '\n')
      continue;
    
    // read the mnist testing dataset
    for (int i = 0; i < m->test_size; ++i) {
      // the first column contains the image label
      fscanf(stream, "%d,", &value);
      m->test_labels[i].values[value] = 1;

      // the remaining columns contain image data
      for (int j = 0; j < MNIST_IMAGE_SIZE - 1; ++j) {
        fscanf(stream, "%d,", &value);
        m->test_images[i].values[j] = value;
      }
      fscanf(stream, "%d", &value);
      m->test_images[i].values[MNIST_IMAGE_SIZE - 1] = value;
    }

    fclose(stream);
  }

  puts("[*] Loading MNIST testing dataset completed");
}

void mnist_normalize (mnist_t* m) {
  const long double scaling_factor = 1.0 / 255.0;
  for (int i = 0; i < m->train_size; ++i)
    matrix_scale(m->train_images + i, scaling_factor);
  for (int i = 0; i < m->test_size; ++i)
    matrix_scale(m->test_images + i, scaling_factor);
}
