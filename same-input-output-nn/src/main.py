import tensorflow as tf
import random

training_size = 4096
testing_size = 256
training_data = [random.randint(0, 1) for _ in range(training_size)]
testing_data = [random.randint(0, 1) for _ in range(testing_size)]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (1,)))
model.add(tf.keras.layers.Dense(1, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'relu'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(training_data, training_data, epochs = 10)

loss, accuracy = model.evaluate(testing_data, testing_data)
print('Loss:', loss)
print('Accuracy:', accuracy)
