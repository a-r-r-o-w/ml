# same-input-output-nn

This is an implementation of a very simple neural network that I wanted to try and implement after watching Prof. Patrick Winston's lectures (can be found on MIT OpenCourseWare). [This](https://www.youtube.com/watch?v=uXt8qF2Zzfo) and [this](https://www.youtube.com/watch?v=VrMHA3yX_QI) are the videos I mention above.

The neural network is very basic and consists of an input layer, hidden layer and output layer. Each layer only contains a single neuron. The goal of the network is to learn to predict the value of the input as the output. I make use of [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) as the activation functions for the neurons, [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) for calculating losses and [gradient descent backpropagation](https://en.wikipedia.org/wiki/Backpropagation) to update the network to perform better. The network takes in as input either a 0 or a 1 and then tries to predict the value of the input. It starts out with a random weight and bias assigned to each neuron and should, in theory, gradually change into some values (using gradient descent backpropagation) that are able to predict the right output.

I've tried making the network to successfully learn and carry out the task of predicting the output to be same as the input but, to the best of my understanding of neural nets so far, haven't been able to figure out why it doesn't work. Most of the time when the program is run, there seems to be an accuracy of 50% (which is because the data is randomly generated 0s and 1s, and even if one would guess only one of the two values all the time, they would be correct approximately with 50% accuracy). On some executions, however, it does seem to be able to find the right system of weights and biases and is able to predict with 100% accuracy.

### Build

```
git clone https://github.com/a-r-r-o-w/ai-ml
cd same-input-output-nn
mkdir build
cd build
cmake ..
make
```

### Results

**C**

Sometimes, the neural network does seem to work as expected but not so at other times...

```bash
┌──(arrow) 💀 [~/…/github/ai-ml/same-input-output-nn/build] <master>
└─$ ./main
Accuracy: 0.515625

┌──(arrow) 💀 [~/…/github/ai-ml/same-input-output-nn/build] <master>
└─$ ./main
Accuracy: 1.000000

┌──(arrow) 💀 [~/…/github/ai-ml/same-input-output-nn/build] <master>
└─$ ./main
Accuracy: 0.515625
```

**Python**

It is sometimes able to find the right fit of weights and biases:

```bash
$ python3 main.py
2022-06-17 19:57:50.335047: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-06-17 19:57:50.424395: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
2022-06-17 19:57:50.424463: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-06-17 19:57:50.425058: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
128/128 [==============================] - 0s 673us/step - loss: 1.2036e-04 - accuracy: 1.0000
Epoch 2/10
128/128 [==============================] - 0s 640us/step - loss: 2.1967e-11 - accuracy: 1.0000
Epoch 3/10
128/128 [==============================] - 0s 635us/step - loss: 3.4495e-15 - accuracy: 1.0000
Epoch 4/10
128/128 [==============================] - 0s 663us/step - loss: 3.4989e-15 - accuracy: 1.0000
Epoch 5/10
128/128 [==============================] - 0s 644us/step - loss: 3.5692e-15 - accuracy: 1.0000
Epoch 6/10
128/128 [==============================] - 0s 646us/step - loss: 3.5354e-15 - accuracy: 1.0000
Epoch 7/10
128/128 [==============================] - 0s 628us/step - loss: 3.4391e-15 - accuracy: 1.0000
Epoch 8/10
128/128 [==============================] - 0s 632us/step - loss: 3.5510e-15 - accuracy: 1.0000
Epoch 9/10
128/128 [==============================] - 0s 630us/step - loss: 3.5067e-15 - accuracy: 1.0000
Epoch 10/10
128/128 [==============================] - 0s 631us/step - loss: 3.5172e-15 - accuracy: 1.0000
8/8 [==============================] - 0s 767us/step - loss: 1.7764e-15 - accuracy: 1.0000
Loss: 1.7763568394002505e-15
Accuracy: 1.0
```

but not so at other times:

```bash
$ python3 main.py
2022-06-17 19:57:59.830278: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-06-17 19:57:59.836549: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
2022-06-17 19:57:59.836574: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-06-17 19:57:59.836815: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
128/128 [==============================] - 0s 669us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 2/10
128/128 [==============================] - 0s 641us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 3/10
128/128 [==============================] - 0s 644us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 4/10
128/128 [==============================] - 0s 637us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 5/10
128/128 [==============================] - 0s 631us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 6/10
128/128 [==============================] - 0s 633us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 7/10
128/128 [==============================] - 0s 642us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 8/10
128/128 [==============================] - 0s 681us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 9/10
128/128 [==============================] - 0s 647us/step - loss: 0.5081 - accuracy: 0.4919
Epoch 10/10
128/128 [==============================] - 0s 623us/step - loss: 0.5081 - accuracy: 0.4919
8/8 [==============================] - 0s 767us/step - loss: 0.4805 - accuracy: 0.5195
Loss: 0.48046875
Accuracy: 0.51953125
```
