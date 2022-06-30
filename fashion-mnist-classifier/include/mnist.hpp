// Arrow

#ifndef FMC_MNIST_HPP
#define FMC_MNIST_HPP

#include <iosfwd>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace fmc {

  class mnist {
    static const int mnist_row_size = 28;
    static const int mnist_col_size = 28;

    public:
      int training_size;
      int testing_size;
      std::vector <fmc::matrix <long double>> training_dataset;
      std::vector <fmc::matrix <long double>> testing_dataset;
      std::vector <int> training_labels;
      std::vector <int> testing_labels;
    
    public:
      mnist (int, int);
      
      void display_training (int) const;
      void display_testing  (int) const;

      std::string get_named_label (int) const;

      mnist& load      (std::string, std::string);
      mnist& normalize ();
    
    private:
      void display (const fmc::matrix <long double>&) const;
  };

  mnist::mnist (int training_size, int testing_size)
    : training_size (training_size),
      testing_size (testing_size) {
    training_dataset.resize(training_size, fmc::matrix <long double> (mnist_row_size, mnist_col_size));
    testing_dataset.resize(testing_size, fmc::matrix <long double> (mnist_row_size, mnist_col_size));
    training_labels.resize(training_size);
    testing_labels.resize(testing_size);
  }

  void mnist::display_training (int index) const {
#ifdef DEBUG_MODE
    if (index < 0 or index >= training_size)
      throw std::runtime_error("out of bounds access will occur with provided index");
#endif

    display(training_dataset[index]);
  }

  void mnist::display_testing (int index) const {
#ifdef DEBUG_MODE
    if (index < 0 or index >= testing_size)
      throw std::runtime_error("out of bounds access will occur with provided index");
#endif

    display(testing_dataset[index]);
  }

  std::string mnist::get_named_label (int label) const {
    static const std::vector <std::string> names = {
      "T-shirt/top",
      "Trouser",
      "Pullover",
      "Dress",
      "Coat",
      "Sandal",
      "Shirt",
      "Sneaker",
      "Bag",
      "Ankle Boot"
    };

    return names[label];
  }

  mnist& mnist::load (std::string training_filepath, std::string testing_filepath) {
    int value;
    int row, col;
    std::string line, pixel, label;

    {
      std::ifstream training_file (training_filepath);
      
      if (!training_file)
        throw std::runtime_error("training dataset does not exist at path: " + training_filepath);
      
      std::getline(training_file, line);
      
      for (int i = 0; i < training_size and training_file.is_open(); ++i) {
        std::getline(training_file, line);
        row = 0, col = 0;

        std::stringstream linestream (line);

        std::getline(linestream, label, ',');
        training_labels[i] = std::stoi(label);

        while (std::getline(linestream, pixel, ',')) {
          value = std::stoi(pixel);
          training_dataset[i].set_value(row, col, value);
          ++col;
          if (col == mnist_col_size)
            ++row, col = 0;
        }
      }

      training_file.close();
    }

    {
      std::ifstream testing_file (testing_filepath);

      if (!testing_file)
        throw std::runtime_error("training dataset does not exist at path: " + testing_filepath);

      std::getline(testing_file, line);
      
      for (int i = 0; i < testing_size and testing_file.is_open(); ++i) {
        std::getline(testing_file, line);
        row = 0, col = 0;

        std::stringstream linestream (line);

        std::getline(linestream, label, ',');
        testing_labels[i] = std::stoi(label);

        while (std::getline(linestream, pixel, ',')) {
          value = std::stoi(pixel);
          testing_dataset[i].set_value(row, col, value);
          ++col;
          if (col == mnist_col_size)
            ++row, col = 0;
        }
      }

      testing_file.close();
    }

    return *this;
  }

  mnist& mnist::normalize () {
    const long double factor = (long double)1.0 / 255.0;

    for (int i = 0; i < training_size; ++i)
      training_dataset[i].scale(factor);
    for (int i = 0; i < testing_size; ++i)
      testing_dataset[i].scale(factor);
    
    return *this;
  }

  void mnist::display (const fmc::matrix <long double>& data) const {
    for (int i = 0; i < data.get_rows(); ++i) {
      for (int j = 0; j < data.get_cols(); ++j) {
        int value = data[i][j];
        std::cout << "\x1b[48;2;" << value << ';' << value << ';' << value << "m  \x1b[0m";
      }
      std::cout << '\n';
    }
  }

} // namespace fmc

#endif // FMC_MNIST_HPP
