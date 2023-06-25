#ifndef HDC_MNIST_H
#define HDC_MNIST_H

#include "matrix.h"

#define TRAINING_DATASET_SIZE 60000
#define TESTING_DATASET_SIZE 10000
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_SIZE 784

typedef struct mnist mnist_t;

struct mnist {
  int train_size;
  int test_size;
  matrix_t* train_images;
  matrix_t* test_images;
  matrix_t* train_labels;
  matrix_t* test_labels;
};

void mnist_constructor                  (mnist_t*);
void mnist_destructor                   (mnist_t*);
void mnist_display_image                (matrix_t*, int);
void mnist_display_test_image           (mnist_t*, int);
void mnist_display_train_image          (mnist_t*, int);
void mnist_display_random_test_image    (mnist_t*);
void mnist_display_random_train_image   (mnist_t*);
void mnist_load                         (mnist_t*);
void mnist_normalize                    (mnist_t*);

#endif // HDC_MNIST_H
