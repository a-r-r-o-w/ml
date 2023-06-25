#include <iostream>

#include "matrix.hpp"
#include "nn.hpp"
#include "utils.hpp"

class dataset {
  public:
    int training_size = 4;
    int testing_size = 4;
    std::vector <nn::matrix <long double>> training_dataset;
    std::vector <nn::matrix <long double>> testing_dataset;
    std::vector <int> training_labels;
    std::vector <int> testing_labels;
    
    dataset ();

    void generate ();
};

dataset::dataset () {
  training_dataset.resize(training_size, nn::matrix <long double> (1, 2));
  testing_dataset.resize(testing_size, nn::matrix <long double> (1, 2));
  training_labels.resize(training_size);
  testing_labels.resize(testing_size);
}

void dataset::generate () {
  for (int index = 0; auto [x, y]: std::vector <std::pair <int, int>> {{0, 0}, {0, 1}, {1, 0}, {1, 1}}) {
    training_dataset[index].set_value(0, 0, x);
    training_dataset[index].set_value(0, 1, y);
    testing_dataset[index].set_value(0, 0, x);
    testing_dataset[index].set_value(0, 1, y);
    training_labels[index] = x ^ y;
    testing_labels[index] = x ^ y;
    ++index;
  }
}

void init_model (nn::network <long double>& model) {
  model
    .add(nn::layer <long double> (2, nn::activation::sigmoid, nn::activation::sigmoid_derivative))
    .add(nn::layer <long double> (32, nn::activation::sigmoid, nn::activation::sigmoid_derivative))
    .add(nn::layer <long double> (2,  nn::activation::sigmoid, nn::activation::sigmoid_derivative))
    .compile();
}

void train (nn::network <long double>& model, const dataset& d) {
  model
    .fit(d.training_dataset, d.training_labels, 10000)
    .save("../model/nn.1.model");
}

void test (nn::network <long double>& model, const dataset& d) {
  model
    .load("../model/nn.1.model")
    .evaluate(d.testing_dataset, d.testing_labels);
}

int main (int argc, char* argv[]) {
  const std::string train_str = "train";
  const std::string test_str = "test";

  if (argc != 2 or (argv[1] != train_str and argv[1] != test_str)) {
    std::cout << "Usage: ./xor-learn [train|test]\n";
    return 0;
  }

  dataset d;
  d.generate();

  nn::network <long double> model (
    0.05,
    nn::error::square_error,
    nn::error::square_error_derivative
  );

  init_model(model);

  if (argv[1] == std::string("train"))
    train(model, d);
  else
    test(model, d);

  return 0;
}