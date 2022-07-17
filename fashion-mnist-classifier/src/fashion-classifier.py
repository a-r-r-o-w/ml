import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

class FashionClassifier:
  """FashionClassifier Implementation in Python"""

  def __init__ (self):
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    self._x_train, self._y_train = train
    self._x_test, self._y_test = test
    self._x_train = tf.keras.utils.normalize(self._x_train, axis = 1)
    self._x_test = tf.keras.utils.normalize(self._x_test, axis = 1)
    self._model = None
  
  def _load_model (self):
    try:
      self._model = tf.keras.models.load_model('fashion-classifier.model')
    except OSError:
      print('Model could not be found. Please "train" the model first')
      exit()
  
  def train (self) -> None:
    """Train the classifier model on the Fashion MNIST training dataset and save it"""

    self._model = tf.keras.models.Sequential()
    self._model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
    self._model.add(tf.keras.layers.Dense(128, activation = 'sigmoid'))
    self._model.add(tf.keras.layers.Dense(128, activation = 'sigmoid'))
    self._model.add(tf.keras.layers.Dense(10, activation = 'sigmoid'))
    self._model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    self._model.fit(self._x_train, self._y_train, epochs = 5)
    self._model.save('fashion-classifier.model')

  def test (self) -> None:
    """Test the classifier model on the Fashion MNIST testing dataset and save it"""

    self._load_model()
    loss, accuracy = self._model.evaluate(self._x_test, self._y_test)
    print('Accuracy:', round(accuracy * 100, 2))
  
  def custom (self, imagepath: str) -> None:
    """Test the classifier model on a custom image
    :param str imagepath: (required) path to image
    """

    self._load_model()
    img = cv2.imread(imagepath)[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = self._model.predict(img)
    guess = np.argmax(prediction)
    print('Guess:', guess)
    plt.imshow(img[0], cmap = plt.cm.binary)
    plt.show()
  
  def view (self, *, count: int = 5) -> None:
    """View the predictions made by the model on random data from the Fashion MNIST testing dataset
    :param int count: (optional) number of predictions to view
    """

    self._load_model()
    indices = random.sample(range(0, len(self._x_test)), count)

    for index in indices:
      img = self._x_test[index]
      prediction = self._model.predict(np.array([img]))
      guess = np.argmax(prediction)
      print('Guess:', guess)
      plt.imshow(img, cmap = plt.cm.binary)
      plt.show()

if __name__ == '__main__':
  fire.core.Display = lambda lines, out: print(*lines, file = out)
  fire.Fire(FashionClassifier)
