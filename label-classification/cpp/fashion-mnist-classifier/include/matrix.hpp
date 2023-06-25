// Arrow

#ifndef FMC_MATRIX_HPP
#define FMC_MATRIX_HPP

#include <iosfwd>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fmc {
  
  /**
   * @brief implementation of a simple interface to use 2d matrices and
            perform operations on them
   * 
   * @tparam T type of the elements that the matrix holds
   */
  template <typename T>
  class matrix {
    public:
      using vec1d = std::vector <T>;
      using vec2d = std::vector <vec1d>;
      using ApplyFuncConstParameter = T (*) (const T&);
      using ApplyFuncNonConstParameter = T (*) (T);
    
    private:
      int rows;
      int cols;
      vec2d values;
    
    public:
      matrix (int = 0, int = 0, const T& = T());
      matrix (int, int, const vec2d&);
      matrix (const matrix&);
      matrix (matrix&&);

      matrix& operator = (const matrix&);
      matrix& operator = (matrix&&);

      matrix& operator += (const matrix&);
      matrix& operator += (const T&);
      matrix& operator -= (const matrix&);
      matrix& operator -= (const T&);
      matrix& operator *= (const matrix&);
      matrix& operator *= (const T&);
      matrix& operator /= (const T&);
      matrix& operator + ();
      matrix& operator - ();

      matrix& operator () (ApplyFuncConstParameter);
      matrix& operator () (ApplyFuncNonConstParameter);

      vec1d&       operator [] (int);
      const vec1d& operator [] (int) const;

      int      get_rows               () const;
      int      get_cols               () const;
      const T& get_value              (int, int) const;
      T        get_value_copy         (int, int) const;
      const T& get_values             () const;
      T&       get_value_reference    (int, int);
      vec2d    get_values_copy        () const;
      vec2d&   get_values_reference   ();
      void     set_value              (int, int, const T&);

      matrix add       (const matrix&) const;
      matrix dot       (const matrix&) const;
      matrix scale     (const T&) const;
      matrix subtract  (const matrix&) const;
      matrix transpose () const;

      template <typename E>
      friend bool operator == (const matrix <E>&, const matrix <E>&);
      
      template <typename E>
      friend bool operator != (const matrix <E>&, const matrix <E>&);

      template <typename E>
      friend std::ostream& operator << (std::ostream&, const matrix <E>&);

      template <typename E>
      friend std::istream& operator >> (std::istream&, matrix <E>&);
  };

  /**
   * @brief Construct a new matrix <T>::matrix object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param rows number of rows in the matrix
   * @param cols number of columns in the matrix
   * @param default_value default value for elements in the matrix
   */
  template <typename T>
  matrix <T>::matrix (int rows, int cols, const T& default_value)
    : rows (rows),
      cols (cols),
      values (matrix <T>::vec2d(rows, matrix <T>::vec1d(cols, default_value)))
  { }

  /**
   * @brief Construct a new matrix <T>::matrix object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param rows number of rows in the matrix
   * @param cols number of columns in the matrix
   * @param values set matrix elements to provided values
   */
  template <typename T>
  matrix <T>::matrix (int rows, int cols, const matrix <T>::vec2d& values)
    : rows (rows),
      cols (cols),
      values (values) {
#ifdef DEBUG_MODE
    int r = values.size();
    int c = values.empty() ? 0 : values.front().size();

    if (rows != r or cols != c)
      throw std::runtime_error("number of rows and cols in vec2d does not match provided row and col size");
#endif
  }

  /**
   * @brief Construct a new matrix <T>::matrix object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix <T> object to create new matrix from by copy
   */
  template <typename T>
  matrix <T>::matrix (const matrix <T>& m)
    : rows (m.rows),
      cols (m.cols),
      values (m.values)
  { }

  /**
   * @brief Construct a new matrix <T>::matrix object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix <T> object to create new matrix from by move
   */
  template <typename T>
  matrix <T>::matrix (matrix <T>&& m)
    : rows (m.rows),
      cols (m.cols),
      values (std::move(m.values))
  { }

  /**
   * @brief Copy a matrix <T>::matrix object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix <T> object to assign new matrix from by copy
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator = (const matrix <T>& m) {
    if (this != &m) {
      rows = m.rows;
      cols = m.cols;
      values = m.values;
    }
    return *this;
  }

  /**
   * @brief Move a matrix <T>::matrix object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix <T> object to assign new matrix from by move
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator = (matrix <T>&& m) {
    if (this != &m) {
      rows = m.rows;
      cols = m.cols;
      values = std::move(m.values);
    }
    return *this;
  }

  /**
   * @brief Operator += overload to carry out addition of two matrix <T> objects
   * 
   * @tparam T type of the elements that the matrix holds
   * @param rhs right-hand-side for the addition operation (left-hand-side is `this`)
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator += (const matrix <T>& rhs) {
#ifdef FMC_DEBUG_MODE
    if (rows != rhs.rows or cols != rhs.cols)
      throw std::runtime_error("incompatible matrices for add operation");
#endif

    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] += rhs.values[i][j];
    return *this;
  }

  /**
   * @brief Operator += overload to carry out addition of a matrix <T> object and a scalar value
   * 
   * @tparam T type of the elements that the matrix holds
   * @param value scalar value to perform addition with
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator += (const T& value) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] += value;
    return *this;
  }

  /**
   * @brief Operator -= overload to carry out subtraction of two matrix <T> objects
   * 
   * @tparam T type of the elements that the matrix holds
   * @param rhs right-hand-side for the subtraction operation (left-hand-side is `this`)
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator -= (const matrix <T>& rhs) {
#ifdef FMC_DEBUG_MODE
    if (rows != rhs.rows or cols != rhs.cols)
      throw std::runtime_error("incompatible matrices for subtract operation");
#endif

    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] -= rhs.values[i][j];
    return *this;
  }

  /**
   * @brief Operator -= overload to carry out subtraction of a matrix <T> object and a scalar value
   * 
   * @tparam T type of the elements that the matrix holds
   * @param value scalar value to perform subtraction with
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator -= (const T& value) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] -= value;
    return *this;
  }

  /**
   * @brief Operator *= overload to carry out dot product of two matrix <T> objects
   * 
   * When performing A *= B, it is assumed that the number of columns in matrix A
   * is equal to the number of rows in matrix B i.e., A.cols == B.rows.
   * If this is not the case, dot product is carried out incorrectly.
   * 
   * A check for the above can be enabled by defining DEBUG_MODE or compiling with
   * the flag -DDEBUG_MODE
   * 
   * @tparam T type of the elements that the matrix holds
   * @param rhs right-hand-side for the dot product operation (left-hand-side is `this`)
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator *= (const matrix <T>& rhs) {
#ifdef DEBUG_MODE
    if (cols != rhs.rows)
      throw std::runtime_error("incompatible matrices for product operation");
#endif

    matrix <T>::vec2d result (rows, matrix <T>::vec1d(rhs.cols, 0));
    
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < rhs.cols; ++j)
        for (int k = 0; k < cols; ++k)
          result[i][j] += values[i][k] * rhs.values[k][j];
    
    rows = rows;
    cols = rhs.cols;
    values = std::move(result);

    return *this;
  }

  /**
   * @brief Operator *= overload to carry out multiplication of a matrix <T> object and a scalar value
   * 
   * @tparam T type of the elements that the matrix holds
   * @param value scalar value to perform multiplication with
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator *= (const T& value) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] *= value;
    return *this;
  }

  /**
   * @brief Operator /= overload to carry out division of a matrix <T> object and a scalar value
   * 
   * @tparam T type of the elements that the matrix holds
   * @param value scalar value to perform division with
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator /= (const T& value) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] /= value;
    return *this;
  }

  /**
   * @brief Does nothing to values of the matrix. Similar to multiplying by 1.
   * 
   * @tparam T type of the elements that the matrix holds
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator + () {
    return *this;
  }

  /**
   * @brief Negates all values of the matrix i.e. positive values become negative and negative values become
   *        positive. Similar to multiplying by -1.
   * 
   * @tparam T type of the elements that the matrix holds
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator - () {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] = -values[i][j];
    return *this;
  }

  /**
   * @brief Operator () overload to apply a function to all elements of the matrix
   * 
   * @tparam T type of the elements that the matrix holds
   * @tparam FunctionType type of the function that is to be applied
   * @param apply_function a function that accepts a parameter of type const T& and returns a value of type T
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator () (ApplyFuncConstParameter apply_function) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] = apply_function(values[i][j]);
    return *this;
  }

  /**
   * @brief Operator () overload to apply a function to all elements of the matrix
   * 
   * @tparam T type of the elements that the matrix holds
   * @tparam FunctionType type of the function that is to be applied
   * @param apply_function a function that accepts a parameter of type T and returns a value of type T
   * @return matrix <T>& reference to self (`this`)
   */
  template <typename T>
  matrix <T>& matrix <T>::operator () (ApplyFuncNonConstParameter apply_function) {
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        values[i][j] = apply_function(values[i][j]);
    return *this;
  }

  /**
   * @brief Operator [] overload to access matrix <T>::matrix rows directly
   * 
   * @tparam T type of the elements that the matrix holds
   * @param index row index that is to be accessed
   * @return matrix <T>::vec1d& reference to matrix row
   */
  template <typename T>
  typename matrix <T>::vec1d& matrix <T>::operator [] (int index) {
#ifdef DEBUG_MODE
    if (index < 0 or index >= rows)
      throw std::runtime_error("out of bounds access will occur with the provided index");
#endif
    return values[index];
  }

  /**
   * @brief Operator [] overload to access matrix <T>::matrix rows directly
   * 
   * @tparam T type of the elements that the matrix holds
   * @param index row index that is to be accessed
   * @return matrix <T>::vec1d& const reference to matrix row
   */
  template <typename T>
  const typename matrix <T>::vec1d& matrix <T>::operator [] (int index) const {
#ifdef DEBUG_MODE
    if (index < 0 or index >= rows)
      throw std::runtime_error("out of bounds access will occur with the provided index");
#endif
    return values[index];
  }

  /**
   * @brief Getter function for matrix <T>::rows
   * 
   * @tparam T type of the elements that the matrix holds
   * @return int number of rows in the matrix
   */
  template <typename T>
  int matrix <T>::get_rows () const
  { return rows; }

  /**
   * @brief Getter function for matrix <T>::cols
   * 
   * @tparam T type of the elements that the matrix holds
   * @return int number of cols in the matrix 
   */
  template <typename T>
  int matrix <T>::get_cols () const
  { return cols; }

  /**
   * @brief Getter function to return a matrix <T>::matrix element by const reference
   * 
   * @tparam T type of the elements that the matrix holds
   * @param i row (0-based indexing)
   * @param j column (0-based indexing)
   * @return const T& const reference to matrix <T>::matrix element at row `i` and col `j`
   */
  template <typename T>
  const T& matrix <T>::get_value (int i, int j) const {
#ifdef DEBUG_MODE
    if (i < 0 or i >= rows or j < 0 or j >= cols)
      throw std::runtime_error("out of bounds access will occur with the provided row and col values");
#endif
    return values[i][j];    
  }

  /**
   * @brief Getter function to return a matrix <T>::matrix element by copy
   * 
   * A check out of bounds access can be enabled by defining DEBUG_MODE or
   * compiling with the flag -DDEBUG_MODE
   * 
   * @tparam T type of the elements that the matrix holds
   * @param i row (0-based indexing)
   * @param j column (0-based indexing)
   * @return T copy of matrix <T>::matrix element at row `i` and col `j`
   */
  template <typename T>
  T matrix <T>::get_value_copy (int i, int j) const {
#ifdef DEBUG_MODE
    if (i < 0 or i >= rows or j < 0 or j >= cols)
      throw std::runtime_error("out of bounds access will occur with the provided row and col values");
#endif
    return values[i][j];
  }
  
  /**
   * @brief Getter function to return a matrix <T>::matrix element by reference
   * 
   * A check out of bounds access can be enabled by defining DEBUG_MODE or
   * compiling with the flag -DDEBUG_MODE
   * 
   * @tparam T type of the elements that the matrix holds
   * @param i row (0-based indexing)
   * @param j column (0-based indexing)
   * @return T& reference of matrix <T>::matrix element at row `i` and col `j`
   */
  template <typename T>
  T& matrix <T>::get_value_reference (int i, int j) {
#ifdef DEBUG_MODE
    if (i < 0 or i >= rows or j < 0 or j >= cols)
      throw std::runtime_error("out of bounds access will occur with the provided row and col values");
#endif
    return values[i][j];
  }

  /**
   * @brief Getter function to return all matrix <T>::matrix elements by copy
   * 
   * @tparam T type of the elements that the matrix holds
   * @return matrix <T>::vec2d copy of all matrix <T>::matrix elements
   */
  template <typename T>
  typename matrix <T>::vec2d matrix <T>::get_values_copy () const
  { return values; }

  /**
   * @brief Getter function to return all matrix <T>::matrix elements by reference
   * 
   * @tparam T type of the elements that the matrix holds
   * @return matrix <T>::vec2d& reference to all matrix <T>::matrix elements
   */
  template <typename T>
  typename matrix <T>::vec2d& matrix <T>::get_values_reference ()
  { return values; }

  /**
   * @brief Setter function to set the value of a particular matrix <T>::matrix element
   * 
   * A check out of bounds access can be enabled by defining DEBUG_MODE or
   * compiling with the flag -DDEBUG_MODE
   * 
   * @tparam T type of the elements that the matrix holds
   * @param i row (0-based indexing)
   * @param j column (0-based indexing)
   * @param value value to set for matrix <T>::matrix element at row `i` and col `j`
   */
  template <typename T>
  void matrix <T>::set_value (int i, int j, const T& value) {
#ifdef DEBUG_MODE
    if (i < 0 or i >= rows or j < 0 or j >= cols)
      throw std::runtime_error("out of bounds access will occur with the provided row and col values");
#endif
    values[i][j] = value;
  }

  /**
   * @brief Add two matrices
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix to be added to `this`
   * @return matrix <T> sum of the two matrices
   */
  template <typename T>
  matrix <T> matrix <T>::add (const matrix <T>& m) const {
    return *this + m;
  }

  /**
   * @brief Dot product of two matrices
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix to be multiplied to `this`
   * @return matrix <T> dot product of the two matrices
   */
  template <typename T>
  matrix <T> matrix <T>::dot (const matrix <T>& m) const {
    return *this * m;
  }

  /**
   * @brief Subtract two matrices
   * 
   * @tparam T type of the elements that the matrix holds
   * @param m matrix to be subtracted from `this`
   * @return matrix <T> difference of the two matrices
   */
  template <typename T>
  matrix <T> matrix <T>::subtract (const matrix <T>& m) const {
    return *this - m;
  }

  /**
   * @brief Scale all values of a matrix by some scaling_factor
   * 
   * @tparam T type of the elements that the matrix holds
   * @param scaling_factor factor to be scale the matrix by
   * @return matrix <T> scaled matrix
   */
  template <typename T>
  matrix <T> matrix <T>::scale (const T& scaling_factor) const {
    return *this * scaling_factor;
  }

  /**
   * @brief Transpose of a matrix
   * 
   * @tparam T type of the elements that the matrix holds
   * @return matrix <T> transposed matrix
   */
  template <typename T>
  matrix <T> matrix <T>::transpose () const {
    matrix <T> t (cols, rows);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        t[j][i] = values[i][j];
    return t;
  }

  /**
   * @brief Operator == overload to check equality of two matrix <T>::matrix objects
   * 
   * @tparam T type of the elements that the matrix holds
   * @param lhs left-hand-side matrix
   * @param rhs right-hand-side matrix
   * @return true if lhs matrix is equal to rhs matrix
   * @return false if lhs matrix is not equal to rhs matrix
   */
  template <typename T>
  bool operator == (const matrix <T>& lhs, const matrix <T>& rhs) {
    return lhs.rows == rhs.rows and lhs.cols == rhs.cols and lhs.values == rhs.values;
  }

  /**
   * @brief Operator != overload to check inequality of two matrix <T>::matrix objects
   * 
   * @tparam T type of the elements that the matrix holds
   * @param lhs left-hand-side matrix
   * @param rhs right-hand-side-matrix
   * @return true if lhs matrix is not equal to rhs matrix
   * @return false if lhs matrix is equal to rhs matrix
   */
  template <typename T>
  bool operator != (const matrix <T>& lhs, const matrix <T>& rhs) {
    return not(lhs == rhs);
  }

  /**
   * @brief Operator << overload to insert a matrix <T>::matrix object representation into
   *        a std::ostream object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param stream std::ostream object to insert into
   * @param m matrix <T> object to be inserted into stream
   * @return std::ostream& reference to std::ostream object to allow chaining
   */
  template <typename T>
  std::ostream& operator << (std::ostream& stream, const matrix <T>& m) {
    for (int i = 0; i < m.rows; ++i) {
      for (int j = 0; j < m.cols; ++j) {
        stream << m.values[i][j];
        if (j != m.cols - 1)
          stream << ' ';
      }
      if (i != m.rows - 1)
        stream << '\n';
    }
    return stream;
  }

  /**
   * @brief Operator >> overload to extract a matrix <T>::matrix object representation from
   *        a std::istream object
   * 
   * @tparam T type of the elements that the matrix holds
   * @param stream std::istream object to extract from
   * @param m matrix <T> object to be extracted from stream
   * @return std::istream& reference to std::istream object to allow chaining
   */
  template <typename T>
  std::istream& operator >> (std::istream& stream, matrix <T>& m) {
    for (int i = 0; i < m.rows; ++i) {
      for (int j = 0; j < m.cols; ++j)
        stream >> m.values[i][j];
    }
    return stream;
  }

  // Same as operator += overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator + (matrix <T> lhs, const matrix <T>& rhs) {
    lhs += rhs;
    return lhs;
  }

  // Same as operator += overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator + (matrix <T> lhs, const T& rhs) {
    lhs += rhs;
    return lhs;
  }

  // Same as operator -= overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator - (matrix <T> lhs, const matrix <T>& rhs) {
    lhs -= rhs;
    return lhs;
  }

  // Same as operator -= overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator - (matrix <T> lhs, const T& rhs) {
    lhs -= rhs;
    return lhs;
  }

  // Same as operator *= overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator * (matrix <T> lhs, const matrix <T>& rhs) {
    lhs *= rhs;
    return lhs;
  }

  // Same as operator *= overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator * (matrix <T> lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }

  // Same as operator /= overload except that this helper function returns a copy
  template <typename T>
  matrix <T> operator / (matrix <T> lhs, const T& rhs) {
    lhs /= rhs;
    return lhs;
  }

} // namespace fmc

#endif // FMC_MATRIX_HPP
