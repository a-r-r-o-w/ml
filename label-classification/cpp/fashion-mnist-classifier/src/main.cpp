#include <iostream>

#include "matrix.hpp"
#include "mnist.hpp"
#include "nn.hpp"
#include "utils.hpp"

void init_model (fmc::network <long double>& model) {
  model
    .add(fmc::layer <long double> (784, fmc::activation::sigmoid, fmc::activation::sigmoid_derivative))
    .add(fmc::layer <long double> (128, fmc::activation::sigmoid, fmc::activation::sigmoid_derivative))
    .add(fmc::layer <long double> (128, fmc::activation::sigmoid, fmc::activation::sigmoid_derivative))
    .add(fmc::layer <long double> (10,  fmc::activation::sigmoid, fmc::activation::sigmoid_derivative))
    .compile();
}

void train (fmc::network <long double>& model, fmc::mnist& mnist) {
  model
    .fit(mnist.training_dataset, mnist.training_labels, 10)
    .save("../model/fmc.1.model");
}

void test (fmc::network <long double>& model, fmc::mnist& mnist) {
  model
    .load("../model/fmc.1.model")
    .evaluate(mnist.testing_dataset, mnist.testing_labels);
}

int main (int argc, char* argv[]) {
  const std::string train_str = "train";
  const std::string test_str = "test";

  if (argc != 2 or (argv[1] != train_str and argv[1] != test_str)) {
    std::cout << "Usage: ./fashion-mnist-classifier [train|test]\n";
    return 0;
  }

  const int training_size = 60000;
  const int testing_size  = 10000;
  
  fmc::mnist mnist (training_size, testing_size);

  mnist
    .load(
      "../res/datasets/fashion-mnist_train.csv",
      "../res/datasets/fashion-mnist_test.csv"
    )
    .normalize();

  fmc::network <long double> model (
    0.005,
    fmc::error::square_error,
    fmc::error::square_error_derivative
  );

  init_model(model);

  if (argv[1] == std::string("train"))
    train(model, mnist);
  else
    test(model, mnist);

  return 0;
}
