#include <stdio.h>
#include <stdlib.h>

#include "error_handler.h"
#include "mnist.h"
#include "nn.h"
#include "utils.h"

// Construct a neural network layer
void layer_constructor (layer_t* layer, layer_t* previous_layer, int neuron_count, long double (*activation_function) (long double),
                        long double (*d_activation_function__dx) (long double)) {
  layer->neuron_count = neuron_count;
  layer->cost = 0;
  layer->activation_function = activation_function;
  layer->d_activation_function__dx = d_activation_function__dx;

  if (previous_layer == NULL) {
    // this is most likely an input layer
    // no weights and biases need to be allocated
    matrix_constructor(&layer->z, 1, layer->neuron_count, 0);
    matrix_constructor(&layer->activation, 1, layer->neuron_count, 0);
    matrix_constructor(&layer->weight, 0, 0, 0);
    matrix_constructor(&layer->bias, 0, 0, 0);
    matrix_constructor(&layer->delta, 0, 0, 0);
  }
  else {
    matrix_constructor(&layer->z, 1, layer->neuron_count, 0);
    matrix_constructor(&layer->activation, 1, layer->neuron_count, 0);
    matrix_constructor(&layer->weight, previous_layer->neuron_count, layer->neuron_count, 0);
    matrix_constructor(&layer->bias, 1, layer->neuron_count, 0);
    matrix_constructor(&layer->delta, layer->neuron_count, 1, 0);
  }
}

// Destruct a neural network layer and free up allocated resources
void layer_destructor (layer_t* layer) {
  matrix_destructor(&layer->delta);
  matrix_destructor(&layer->bias);
  matrix_destructor(&layer->weight);
  matrix_destructor(&layer->activation);
  matrix_destructor(&layer->z);
  layer->cost = 0;
  layer->neuron_count = 0;
}

// Set random weights and biases between [-1, 1] in the layer
void layer_randomize (layer_t* layer) {
  for (int i = 0; i < layer->neuron_count; ++i)
    layer->bias.values[i] = random_value();
  for (int i = 0; i < layer->weight.rows; ++i)
    for (int j = 0; j < layer->weight.cols; ++j)
      matrix_set(&layer->weight, i, j, random_value());
}

// Set the activations of a layer
void layer_set (layer_t* layer, matrix_t* m) {
  matrix_assign(m, &layer->activation);
}

// Add a layer to the neural network
void network_add (network_t* network, int neuron_count, long double (*activation_function) (long double),
                  long double (*d_activation_function__dx) (long double)) {
#ifdef DEBUG_MODE
  if (network->layers_used >= network->layer_count)
    error("network already contains maximum number of layers");
#endif
  
  // we need to know the number of neurons in the previous layer in order to
  // create a matrix to represent the `weight` matrix
  layer_t* previous_layer = NULL;
  
  // check if a previous layer exists
  if (network->layers_used > 0)
    previous_layer = &network->layers[network->layers_used - 1];
  
  // construct the current layer
  layer_constructor(network->layers + network->layers_used, previous_layer, neuron_count, activation_function, d_activation_function__dx);
  ++network->layers_used;
}

// Perform backward propagation from one layer to another layer
void network_backward_propagate (layer_t* from, layer_t* to, long double learning_rate) {
  for (int i = 0; i < from->neuron_count; ++i) {
    for (int j = 0; j < to->neuron_count; ++j) {
      // calculate backward propagation errors using cached deltas for each neuron
      long double d_z__d_w = to->activation.values[j];
      long double d_z__d_b = 1;
      long double d_c__d_w = d_z__d_w * from->delta.values[i];
      long double d_c__d_b = d_z__d_b * from->delta.values[i];
      int index = matrix_getindex(&from->weight, j, i);
      
      // multiply with negative of the gradient in order to minimise
      //   gradient is the direction to move in, in order to maximise a function
      from->weight.values[index] += learning_rate * (-d_c__d_w);
      from->bias.values[i] += learning_rate * (-d_c__d_b);
    }
  }
}

// Calculate the loss of the network
void network_calculate_costs (network_t* network, layer_t* layer, const matrix_t* expected) {
  layer->cost = 0;
  for (int i = 0; i < layer->neuron_count; ++i) {
    long double d_activation__d_z = layer->d_activation_function__dx(layer->z.values[i]);
    long double d_cost__d_activation = network->d_loss_function__dx(layer->activation.values[i], expected->values[i]);
    layer->delta.values[i] = d_activation__d_z * d_cost__d_activation;
    layer->cost += network->loss_function(layer->activation.values[i], expected->values[i]);
  }
}

// Calculate the deltas (partial derivatives) and cache it for each neuron in a layer in order to reuse
// instead of recomputing it each time
void network_calculate_deltas (layer_t* current_layer, layer_t* next_layer) {
  matrix_dot(&current_layer->delta, &next_layer->weight, &next_layer->delta);
  for (int i = 0; i < current_layer->neuron_count; ++i) {
    long double d_activation__d_z = current_layer->d_activation_function__dx(current_layer->z.values[i]);
    current_layer->delta.values[i] *= d_activation__d_z;
  }
}

// Construct a neural network
void network_constructor (network_t* network, int layer_count, long double learning_rate,
                          long double (*loss_function) (long double, long double), 
                          long double (*d_loss_function__dx) (long double, long double)) {
  network->layer_count = layer_count;
  network->layers_used = 0;
  network->learning_rate = learning_rate;
  network->layers = (layer_t*)malloc(network->layer_count * sizeof(layer_t));
  network->loss_function = loss_function;
  network->d_loss_function__dx = d_loss_function__dx;
}

// Destruct a neural network and free up allocated resources
void network_destructor (network_t* network) {
  for (int i = 0; i < network->layers_used; ++i)
    layer_destructor(network->layers + i);
  
  free(network->layers);
  network->layers = NULL;
  network->layers_used = 0;
  network->layer_count = 0;
}

// Perform forward propagation from one layer to another layer
void network_forward_propagate (layer_t* from, layer_t* to) {
  // Forward Propagation:
  //   Z(L) = Activation(L - 1) * Weight(L) + Bias(L)
  //   Activation(L) = Activation_Function(Z(L))
  
  matrix_dot(&to->z, &from->activation, &to->weight);
  matrix_add(&to->z, &to->bias);
  matrix_apply_from(&to->z, &to->activation, to->activation_function);
}

void network_load (network_t* network, const char* filename) {
  puts("[*] Loading saved model");

  if (!file_exists(filename))
    error("saved model could not be found with the provided filename");

  FILE* file = fopen(filename, "rb");
  char buffer[256];
  long double value;

  for (int i = 1; i < network->layer_count; ++i) {
    layer_t* layer = network->layers + i;

    fscanf(file, "%[^\n]s", buffer);
    printf("[*] reading %s\n", buffer);

    for (int j = 0; j < layer->neuron_count; ++j) {
      fscanf(file, "%Lf\n", &value);
      layer->bias.values[j] = value;
    }

    fscanf(file, "%[^\n]s", buffer);
    printf("[*] reading %s\n", buffer);

    for (int j = 0; j < layer->weight.rows; ++j)
      for (int k = 0; k < layer->weight.cols; ++k) {
        fscanf(file, "%Lf\n", &value);
        matrix_set(&layer->weight, j, k, value);
      }
  }

  fclose(file);
}

// Predict a label for an image using a trained neural network
int network_predict (network_t* network, matrix_t* img) {
  // holds the flattened 28x28 image data in an array of size 784
  matrix_t flattened;
  flattened.rows = 1;
  flattened.cols = MNIST_IMAGE_SIZE;
  flattened.values = matrix_flatten(img);
  
  // set the input layer of the network with the flattened matrix representation of the image
  layer_set(network->layers + 0, &flattened);

  // forward propagation
  for (int j = 0; j < network->layer_count - 1; ++j)
    network_forward_propagate(network->layers + j, network->layers + j + 1);
  
  int prediction = argmax(network->layers[network->layer_count - 1].activation.values, 10);
  return prediction;
}

// Set random weights and biases in the neural network
void network_randomize (network_t* network) {
  for (int i = 1; i < network->layer_count; ++i)
    layer_randomize(network->layers + i);
}

void network_save (network_t* network, const char* filename) {
  puts("[*] Saving model to file \"hdc.model\"");

  FILE* file = fopen(filename, "w");

  for (int i = 1; i < network->layer_count; ++i) {
    layer_t* layer = network->layers + i;

    fprintf(file, "[layer %d bias]\n", i);
    for (int j = 0; j < layer->neuron_count; ++j)
      fprintf(file, "%.20Lf%c", layer->bias.values[j], " \n"[j == layer->neuron_count - 1]);
    fprintf(file, "[layer %d weights]\n", i);
    for (int j = 0; j < layer->weight.rows; ++j)
      for (int k = 0; k < layer->weight.cols; ++k)
        fprintf(file, "%.20Lf%c", matrix_get(&layer->weight, j, k), " \n"[k == layer->weight.cols - 1]);
  }

  fclose(file);
}

// Test the neural network
int network_test (network_t* network, const mnist_t* mnist) {
  puts("[*] Testing neural network");

  int correct_predictions = 0;

  // test the network on the testing dataset and collect the predictions
  for (int i = 0; i < mnist->test_size; ++i) {
    int prediction = network_predict(network, mnist->test_images + i);
    int label = argmax(mnist->test_labels[i].values, 10);

    if (prediction == label)
      ++correct_predictions;
  }

  return correct_predictions;
}

// Train the neural network
void network_train (network_t* network, const mnist_t* mnist, int epochs) {
  puts("[*] Training neural network");

  // holds the flattened 28x28 image data in an array of size 784
  matrix_t flattened;
  flattened.rows = 1;
  flattened.cols = MNIST_IMAGE_SIZE;

  // train the network on the training dataset for `epochs` number of times
  for (int epoch = 0; epoch < epochs; ++epoch) {
    printf("[*] Epoch: %d\n", epoch + 1);

    // feed every image through the neural network
    for (int i = 0; i < mnist->train_size; ++i) {
      flattened.values = matrix_flatten(mnist->train_images + i);

      // set the input layer of the network with the flattened matrix representation of the image
      layer_set(network->layers + 0, &flattened);

      // forward propagation
      for (int j = 0; j < network->layer_count - 1; ++j)
        network_forward_propagate(network->layers + j, network->layers + j + 1);
      
      // calculate the loss of the network
      network_calculate_costs(network, network->layers + network->layer_count - 1, mnist->train_labels + i);
      
      // calculate the deltas for each layer
      //   the delta value is used to calculate the errors for backward propagation
      //   by caching the derivatives
      for (int j = network->layer_count - 1; j > 1; --j)
        network_calculate_deltas(network->layers + j - 1, network->layers + j);
      
      // backward propagation
      for (int j = network->layer_count - 1; j > 0; --j)
        network_backward_propagate(network->layers + j, network->layers + j - 1, network->learning_rate);
    }
  }
}
