#ifndef HDC_NN_H
#define HDC_NN_H

#include "matrix.h"
#include "mnist.h"

typedef struct layer layer_t;
typedef struct network network_t;

struct layer {
  int neuron_count;
  long double cost;
  matrix_t z;
  matrix_t activation;
  matrix_t weight;
  matrix_t bias;
  matrix_t delta;
  long double (*activation_function) (long double);
  long double (*d_activation_function__dx) (long double);
};

struct network {
  int layer_count;
  int layers_used;
  long double learning_rate;
  layer_t* layers;
  long double (*loss_function) (long double, long double);
  long double (*d_loss_function__dx) (long double, long double);
};

void layer_constructor  (layer_t*, layer_t*, int, long double (*) (long double), long double (*) (long double));
void layer_destructor   (layer_t*);
void layer_randomize    (layer_t*);
void layer_set          (layer_t*, matrix_t*);

void network_add                 (network_t*, int, long double (*) (long double), long double (*) (long double));
void network_backward_propagate  (layer_t*, layer_t*, long double);
void network_calculate_costs     (network_t*, layer_t*, const matrix_t*);
void network_constructor         (network_t*, int, long double, long double (*) (long double, long double),
                                  long double (*) (long double, long double));
void network_destructor          (network_t*);
void network_forward_propagate   (layer_t*, layer_t*);
void network_load                (network_t*, const char*);
int  network_predict             (network_t*, matrix_t*);
void network_randomize           (network_t*);
void network_save                (network_t*, const char*);
int  network_test                (network_t*, const mnist_t*);
void network_train               (network_t*, const mnist_t*, int);

#endif // HDC_NN_H
