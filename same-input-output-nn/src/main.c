#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct neuron neuron_t;
typedef struct layer layer_t;

const int training_dataset_size = 4096;
const int testing_dataset_size = 256;

long double training_input[4096][1];
long double training_expected[4096][1];
long double testing_input[256][1];
long double testing_expected[256][1];

struct neuron {
  long double z;
  long double activation;
  long double bias;
  long double* weight;
  long double delta;
  int weight_count;
};

struct layer {
  int neuron_count;
  long double cost;
  neuron_t* neurons;
};

// construct a neuron
void neuron_constructor (neuron_t* neuron) {
  neuron->z = 0;
  neuron->activation = 0;
  neuron->bias = 0;
  neuron->weight = NULL;
  neuron->delta = 0;
  neuron->weight_count = 0;
}

// destruct a neuron
void neuron_destructor (neuron_t* neuron) {
  free(neuron->weight);
  neuron->z = 0;
  neuron->activation = 0;
  neuron->bias = 0;
  neuron->weight = NULL;
  neuron->delta = 0;
  neuron->weight_count = 0;
}

// attach a neuron to other neurons
void neuron_attach (neuron_t* neuron, int neuron_count) {
  neuron->bias = rand() % 10;
  neuron->weight_count = neuron_count;
  neuron->weight = (long double*)malloc(sizeof(long double) * neuron_count);
  
  for (int i = 0; i < neuron_count; ++i)
    neuron->weight[i] = rand() % 10;
}

// constuct a layer of neurons
void layer_constructor (layer_t* layer, int neuron_count, layer_t* previous_layer) {
  layer->neuron_count = neuron_count;
  layer->cost = 0;
  layer->neurons = (neuron_t*)malloc(sizeof(neuron_t) * layer->neuron_count);
  
  for (int i = 0; i < layer->neuron_count; ++i)
    neuron_constructor(layer->neurons + i);

  if (previous_layer != NULL) {
    for (int i = 0; i < layer->neuron_count; ++i)
      neuron_attach(layer->neurons + i, previous_layer->neuron_count);
  }
}

// destruct a layer of neurons
void layer_destructor (layer_t* layer) {
  for (int i = 0; i < layer->neuron_count; ++i)
    neuron_destructor(layer->neurons + i);
  free(layer->neurons);
  layer->neuron_count = 0;
  layer->cost = 0;
}

// set the activations of a layer as input values
void layer_input (layer_t* layer, long double* values) {
  for (int i = 0; i < layer->neuron_count; ++i)
    layer->neurons[i].activation = values[i];
}

// pretty print a layer structure
void layer_print (layer_t* layer) {
  printf("<layer_t at %p>: {\n"
         "  \"neuron_count\": %d,\n"
         "  \"neurons\": [\n",
    (void*)layer, layer->neuron_count
  );
  for (int i = 0; i < layer->neuron_count; ++i) {
    neuron_t* neuron = layer->neurons + i;
    printf("    [%d]: {\n"
           "               \"z\": %Lf\n"
           "      \"activation\": %Lf\n"
           "            \"bias\": %Lf\n"
           "           \"delta\": %Lf\n"
           "         \"weights\": [",
      i, neuron->z, neuron->activation, neuron->bias, neuron->delta
    );
    if (neuron->weight_count > 0)
      for (int j = 0; j < neuron->weight_count; ++j)
        printf("%Lf%s", neuron->weight[j], j == neuron->weight_count - 1 ? "]\n" : ", ");
    else
      printf("]\n");
    printf("    }\n"
           "  ]\n"
           "}\n");
  }
}

// generate training and testing input
void generate_input () {
  for (int i = 0; i < training_dataset_size; ++i) {
    training_input[i][0] = rand() % 2;
    training_expected[i][0] = training_input[i][0];
  }
  for (int i = 0; i < testing_dataset_size; ++i) {
    testing_input[i][0] = rand() % 2;
    testing_expected[i][0] = testing_input[i][0];
  }
}

// rectified linear unit function
long double relu (long double x) {
  return x > 0 ? x : 0;
}

// derivative of relu wrt its parameter
long double drelu__dx (long double x) {
  return x > 0 ? 1 : 0;
}

// mean squared error function to calculate cost of network
long double mean_squared_error (long double x, long double y) {
  long double z = x - y;
  return z * z;
}

// derivative of mse function wrt to its paramter
long double dmean_squared_error__dx (long double x, long double y) {
  return 2 * (x - y);
}

void calculate_errors (layer_t* output_layer, long double* expected) {
  output_layer->cost = 0;
  for (int i = 0; i < output_layer->neuron_count; ++i) {
    long double dc_da = dmean_squared_error__dx(output_layer->neurons[i].activation, expected[i]);
    long double da_dz = drelu__dx(output_layer->neurons[i].z);
    output_layer->neurons[i].delta = dc_da * da_dz;
    output_layer->cost += mean_squared_error(output_layer->neurons[i].activation, expected[i]);
  }
  // printf("cost: %Lf\n", output_layer->cost);
}

void calculate_delta (layer_t* current_layer, layer_t* next_layer) {
  for (int i = 0; i < current_layer->neuron_count; ++i) {
    long double da_dz = drelu__dx(current_layer->neurons[i].z);
    long double dZ_da = 0;
    for (int j = 0; j < next_layer->neuron_count; ++j)
      dZ_da += next_layer->neurons[j].weight[i] * next_layer->neurons[j].delta;
    current_layer->neurons[i].delta = da_dz * dZ_da;
  }
}

// forward propagation to calculate activations from one layer to next layer
void forward_propagate (layer_t* from, layer_t* to) {
  for (int i = 0; i < to->neuron_count; ++i) {
    to->neurons[i].z = 0;
    for (int j = 0; j < from->neuron_count; ++j)
      to->neurons[i].z += from->neurons[j].activation * to->neurons[i].weight[j] + to->neurons[i].bias;
    to->neurons[i].activation = relu(to->neurons[i].z);
  }
}

// backward propagation to train network based on cost of forward propagation
void backward_propagate (layer_t* from, layer_t* to, long double learning_rate) {
  for (int i = 0; i < from->neuron_count; ++i) {
    for (int j = 0; j < to->neuron_count; ++j) {
      long double dz_dw = to->neurons[j].activation;
      
      // gradient is direction for maximising value of a function
      long double dc_dw = dz_dw * from->neurons[i].delta;
      long double dc_db = 1 * from->neurons[i].delta;
      
      // negative of gradient is direction for minimizing value of a function
      from->neurons[i].weight[j] += learning_rate * (-dc_dw);
      from->neurons[i].bias += learning_rate * (-dc_db);
    }
  }
}

// train the network
void train (layer_t* input_layer, layer_t* hidden_layer, layer_t* output_layer, int epochs) {
  while (epochs--) {
    for (int i = 0; i < training_dataset_size; ++i) {
      layer_input(input_layer, training_input[i]);
      
      forward_propagate(input_layer, hidden_layer);
      forward_propagate(hidden_layer, output_layer);
      // printf("input: %Lf, output: %Lf\n", training_input[i][0], output_layer->neurons[0].activation);

      calculate_errors(output_layer, training_expected[i]);
      calculate_delta(hidden_layer, output_layer);

      backward_propagate(output_layer, hidden_layer, 0.001);
      backward_propagate(hidden_layer, input_layer, 0.001);
    }
  }
}

int evaluate (layer_t* input_layer, layer_t* hidden_layer, layer_t* output_layer) {
  long double epsilon = 1e-6;
  int correct = 0;
  for (int i = 0; i < testing_dataset_size; ++i) {
    layer_input(input_layer, testing_input[i]);

    forward_propagate(input_layer, hidden_layer);
    forward_propagate(hidden_layer, output_layer);

    if (fabsl(round(output_layer->neurons[0].activation) - testing_expected[i][0]) < epsilon)
      ++correct;
  }
  return correct;
}

int main () {
  srand(time(0));

  generate_input();

  layer_t input_layer;
  layer_constructor(&input_layer, 1, NULL);

  layer_t hidden_layer;
  layer_constructor(&hidden_layer, 1, &input_layer);

  layer_t output_layer;
  layer_constructor(&output_layer, 1, &hidden_layer);

  train(&input_layer, &hidden_layer, &output_layer, 10);
  // layer_print(&output_layer);

  int c = evaluate(&input_layer, &hidden_layer, &output_layer);
  printf("Accuracy: %Lf\n", (long double)c / testing_dataset_size);
  
  layer_destructor(&output_layer);
  layer_destructor(&hidden_layer);
  layer_destructor(&input_layer);

  return 0;
}
