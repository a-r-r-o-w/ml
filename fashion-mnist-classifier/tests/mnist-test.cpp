#include <iostream>

#include "mnist.hpp"
#include "utils.hpp"

int main () {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  const int training_size = 60000; // reduce for faster loading
  const int testing_size  = 10000; // reduce for faster loading
  const std::string base_path = "../res/datasets/";
  
  fmc::mnist mnist_fashion (training_size, testing_size);
  mnist_fashion.load(base_path + "fashion-mnist_train.csv", base_path + "fashion-mnist_test.csv");

  std::cout << "5 Random Training Images\n\n";

  for (int i = 0; i < 5; ++i) {
    int index = fmc::random::random(0, training_size - 1);
    std::cout << "Label: " << mnist_fashion.get_named_label(mnist_fashion.training_labels[index]) << '\n';
    mnist_fashion.display_training(index);
    std::cout << '\n';
  }

  std::cout << "5 Random Testing Images\n\n";

  for (int i = 0; i < 5; ++i) {
    int index = fmc::random::random(0, testing_size - 1);
    std::cout << "Label: " << mnist_fashion.get_named_label(mnist_fashion.testing_labels[index]) << '\n';
    mnist_fashion.display_testing(index);
    std::cout << '\n';
  }

  return 0;
}
