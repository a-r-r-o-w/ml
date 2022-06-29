// Arrow

#ifndef FMC_UTILS_HPP
#define FMC_UTILS_HPP

#include <cmath>
#include <concepts>
#include <random>
#include <type_traits>

namespace fmc {

  namespace random {
    
    /**
     * @brief returns a random integer in range [x, y]
     * 
     * @tparam T integer type
     * @param x start of range
     * @param y end of range
     * @return T random integer in range [x, y]
     */
    template <typename T>
    T random (const T& x, const T& y) requires std::integral <T> {
      static std::random_device rd;
      static std::mt19937 generator (rd());
      static std::uniform_int_distribution <T> distribution (x, y);
      return distribution(generator);
    }

    /**
     * @brief returns a random floating point value in range [x, y]
     * 
     * @tparam T floating point type
     * @param x start of range
     * @param y end of range
     * @return T random floating point value in range [x, y]
     */
    template <typename T>
    T random (const T& x, const T& y) requires std::floating_point <T> {
      static std::random_device rd;
      static std::mt19937 generator (rd());
      static std::uniform_real_distribution <T> distribution (x, y);
      return distribution(generator);
    }

  } // namespace random

  namespace activation {

    /**
     * @brief ReLU (Rectified Linear Unit)
     * 
     * @tparam T comparable type
     * @param x input
     * @return T ReLU(x)
     */
    template <typename T>
    T relu (const T& x) requires std::totally_ordered <T> {
      return x > 0 ? x : 0;
    }

    /**
     * @brief Derivative of the ReLU function w.r.t. its input
     * 
     * @tparam T comparable type
     * @param x input
     * @return T ReLU_Derivative(x)
     */
    template <typename T>
    T relu_derivative (const T& x) requires std::totally_ordered <T> {
      return x > 0 ? 1 : 0;
    }

    /**
     * @brief Sigmoid
     * 
     * @tparam T integral or floating point type
     * @param x input
     * @return T Sigmoid(x)
     */
    template <typename T>
    T sigmoid (const T& x) requires (std::integral <T> or std::floating_point <T>) {
      T result = T(1) / (T(1) + std::exp(-x));
      return result;
    }

    /**
     * @brief Derivative of the Sigmoid function w.r.t. its input
     * 
     * @tparam T integral or floating point type
     * @param x input
     * @return T Sigmoid_Derivative(x)
     */
    template <typename T>
    T sigmoid_derivative (const T& x) requires (std::integral <T> or std::floating_point <T>) {
      T result = sigmoid(x);
      return result * (1 - result);
    }

  } // namespace activation

  namespace error {

    /**
     * @brief Calculates the square error: (x - y) ** 2
     * 
     * @tparam T type
     * @param lhs x (in above expression)
     * @param rhs y (in above expression)
     * @return T square error
     */
    template <typename T>
    T square_error (const T& lhs, const T& rhs) {
      T difference = lhs - rhs;
      return difference * difference;
    }

    /**
     * @brief Calculates the derivative of the square error w.r.t. x: 2 * (x - y)
     * 
     * @tparam T type
     * @param lhs x (in above expression)
     * @param rhs y (in above expression)
     * @return T square error derivative
     */
    template <typename T>
    T square_error_derivative (const T& lhs, const T& rhs) {
      T difference = lhs - rhs;
      return 2 * difference;
    }

  } // namespace error

} // namespace fmc

#endif // FMC_UTILS_HPP
