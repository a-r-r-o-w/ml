// Arrow

#ifndef FMC_NN_HPP
#define FMC_NN_HPP

#include <algorithm>
#include <iosfwd>
#include <vector>

#include "matrix.hpp"
#include "utils.hpp"

namespace fmc {

  template <typename T>
  class network;

  template <typename T>
  class layer {
    public:
      using ActivationFunc = T (*) (const T&);

    private:
      int neuron_count;
      matrix <T> z;
      matrix <T> activation;
      matrix <T> weight;
      matrix <T> bias;
      matrix <T> delta;

    public:
      ActivationFunc activation_function;
      ActivationFunc activation_function_derivative;
    
    public:
      layer (int, ActivationFunc, ActivationFunc);

      const matrix <T>& get_z            () const;
      const matrix <T>& get_activation   () const;
      const matrix <T>& get_weight       () const;
      const matrix <T>& get_bias         () const;
      const matrix <T>& get_delta        () const;
      int               get_neuron_count () const;

      void backward_propagate (layer&, const T&);
      void calculate_delta    (const layer&);
      void forward_propagate  (layer&);
      void join_layer         (const layer&);
      void randomize          ();
      void set_activation     (const matrix <T>&);
      void set_delta          (const matrix <T>&);
      
      template <typename E>
      friend void network <E>::join_layers ();

      template <typename E>
      friend std::ostream& operator << (std::ostream&, const layer <E>&);
  };

  template <typename T>
  class network {
    public:
      using LossFunction = T (*) (const T&, const T&);

    public:
      int layer_count;
      T cost;
      T learning_rate;
      std::vector <layer <T>> layers;
    
    public:
      LossFunction loss_function;
      LossFunction loss_function_derivative;
    
    public:
      network (const T&, LossFunction, LossFunction);
      
      network& add                (const layer <T>&);
      network& add                (layer <T>&&);
      void     backward_propagate ();
      void     calculate_delta    ();
      void     calculate_loss     (int);
      void     compile            ();
      void     fit                (const std::vector <matrix <T>>&, const std::vector <int>&);
      void     forward_propagate  (const matrix <T>&);
      void     join_layers        ();
      void     randomize          ();
  };

  template <typename T>
  layer <T>::layer (int neuron_count, ActivationFunc activation_function,
                    ActivationFunc activation_function_derivative)
    : neuron_count (neuron_count),
      activation_function (activation_function),
      activation_function_derivative (activation_function_derivative)
  { }

  template <typename T>
  const matrix <T>& layer <T>::get_z () const {
    return z;
  }

  template <typename T>
  const matrix <T>& layer <T>::get_activation () const {
    return activation;
  }

  template <typename T>
  const matrix <T>& layer <T>::get_weight () const {
    return weight;
  }

  template <typename T>
  const matrix <T>& layer <T>::get_bias () const {
    return bias;
  }

  template <typename T>
  const matrix <T>& layer <T>::get_delta () const {
    return delta;
  }

  template <typename T>
  int layer <T>::get_neuron_count () const {
    return neuron_count;
  }

  template <typename T>
  void layer <T>::backward_propagate (layer <T>& layer, const T& learning_rate) {
    weight = -layer.activation.transpose() * delta * learning_rate;
    bias = -delta * learning_rate;
  }

  template <typename T>
  void layer <T>::calculate_delta (const layer <T>& layer) {
    delta = layer.delta * layer.weight.transpose();
  }

  template <typename T>
  void layer <T>::forward_propagate (layer <T>& layer) {
    layer.z = activation * layer.weight + layer.bias;
    layer.activation = layer.z;
    layer.activation(layer.activation_function);
  }

  template <typename T>
  void layer <T>::join_layer (const layer <T>& layer) {
    z          = matrix <T> (1, neuron_count);
    activation = matrix <T> (1, neuron_count);
    weight     = matrix <T> (layer.neuron_count, neuron_count);
    bias       = matrix <T> (1, neuron_count);
    delta      = matrix <T> (1, neuron_count);
  }
  
  template <typename T>
  void layer <T>::randomize () {
    bias([] ([[maybe_unused]] const T& _) { return random::random <T> (0, 1); });
    weight([] ([[maybe_unused]] const T& _) { return random::random <T> (0, 1); });
  }

  template <typename T>
  void layer <T>::set_activation (const matrix <T>& activation_) {
#ifdef DEBUG_MODE
    if (activation.get_rows() != activation_.get_rows() or activation.get_cols() != activation_.get_cols())
      throw std::runtime_error("incompatible matrix for activation assignment");
#endif
    activation = activation_;
  }

  template <typename T>
  std::ostream& operator << (std::ostream& stream, const layer <T>& layer) {
    stream << "<layer object @" << &layer << ">: {\n"
           << "  neuron_count: " << layer.neuron_count << ",\n"
           << "             z:\n" << layer.z << '\n'
           << "    activation:\n" << layer.activation << '\n'
           << "        weight:\n" << layer.weight << '\n'
           << "          bias:\n" << layer.bias << '\n'
           << "         delta:\n" << layer.delta << '\n'
           << "}";
    return stream;
  }

  template <typename T>
  network <T>::network (const T& learning_rate, LossFunction loss_function,
                        LossFunction loss_function_derivative)
    : layer_count (0),
      cost (T()),
      learning_rate (learning_rate),
      loss_function (loss_function),
      loss_function_derivative (loss_function_derivative)
  { }

  template <typename T>
  network <T>& network <T>::add (const layer <T>& layer) {
    ++layer_count;
    layers.push_back(layer);
    return *this;
  }

  template <typename T>
  network <T>& network <T>::add (layer <T>&& layer) {
    ++layer_count;
    layers.emplace_back(std::move(layer));
    return *this;
  }

  template <typename T>
  void network <T>::backward_propagate () {
    for (int i = layer_count - 1; i > 1; --i)
      layers[i].backward_propagate(layers[i - 1], learning_rate);
  }

  template <typename T>
  void network <T>::calculate_delta () {
    for (int i = layer_count - 1; i > 1; --i)
      layers[i - 1].calculate_delta(layers[i]);
  }

  template <typename T>
  void network <T>::calculate_loss (int label) {
    int output_neuron_count = layers.back().get_neuron_count();

#ifdef DEBUG_MODE
    if (label < 0 or label >= output_neuron_count)
      throw std::runtime_error("label does not lie in the range of number of neurons in output layer");
#endif
    
    const matrix <T>& z = layers.back().get_z();
    const matrix <T>& predictions = layers.back().get_activation();
    matrix <T> new_delta (1, output_neuron_count);
    std::vector <T> expected (output_neuron_count, 0);
    
    expected[label] = 1;
    cost = 0;

    for (int i = 0; i < output_neuron_count; ++i) {
      T error = loss_function(predictions[0][i], expected[i]);
      
      T activation_z_derivative    = layers.back().activation_function_derivative(z[0][i]);
      T cost_activation_derivative = loss_function_derivative(predictions[0][i], expected[i]);
      
      cost += error;
      new_delta[0][i] = activation_z_derivative * cost_activation_derivative;
    }

    cost /= output_neuron_count;
  }

  template <typename T>
  void network <T>::compile () {
    join_layers();
    randomize();
  }

  template <typename T>
  void network <T>::fit (const std::vector <matrix <T>>& data, const std::vector <int>& labels) {
#ifdef DEBUG_MODE
    if (data.size() != labels.size())
      throw std::runtime_error("data and labels must have same size");
#endif

    for (int i = 0; i < (int)data.size(); ++i) {
      forward_propagate(data[i]);
      calculate_loss(labels[i]);
      calculate_delta();
      backward_propagate();
    }
  }

  template <typename T>
  void network <T>::forward_propagate (const matrix <T>& data) {
    layers.front().set_activation(data);
    for (int i = 0; i < layer_count - 1; ++i)
      layers[i].forward_propagate(layers[i + 1]);
  }

  template <typename T>
  void network <T>::join_layers () {
    layer <T> dummy (0, activation::sigmoid, activation::sigmoid_derivative);
    layers[0].join_layer(dummy);

    for (int i = 0; i < layer_count - 1; ++i)
      layers[i + 1].join_layer(layers[i]);
  }

  template <typename T>
  void network <T>::randomize () {
    std::for_each(layers.begin(), layers.end(), [] (layer <T>& layer) { layer.randomize(); });
  }

} // namespace fmc

#endif // FMC_NN_HPP
