mnist-ml
========

Pre-requisites
--------------

- [Python >= 3.5](https://www.python.org/downloads/release/python-352/)
- [TensorFlow 1.0](https://www.tensorflow.org/install/)

If you have Python installed, you can simply download and install TensorFlow
with pip:

`pip install --upgrade tensorflow`

How to run
----------

If you do not want to modify the default settings, simply run training and
evaluation as follows:

```
python3 main.py
```

You can obtain the full command synopsis by running `python3 main.py --help`:

```
usage: main.py [-h] [--model-type MODEL_TYPE] [--batch-size BATCH_SIZE]
               [--batches BATCHES] [--max-cpu-cores MAX_CPU_CORES] [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --model-type MODEL_TYPE
                        which model (basic_cnn, highway_cnn)
  --batch-size BATCH_SIZE
                        batch size
  --batches BATCHES     total number of batches
  --max-cpu-cores MAX_CPU_CORES
                        how many cores to use at most
  --quiet               be quiet
```

It is recommended to set a suitable number of CPU cores to speed up training. 

Note: the current version of TensorFlow reports a number of errors at start-up in Windows.
This is a [known bug](http://stackoverflow.com/q/42217532/1467943).

Data sets
---------

The script uses the TensorFlow wrapper for MNIST data. When first executed, the
script will automatically download the data and store it in the current
directory.

Models
------

There are two basic model types which are described below.

### Basic CNN

This model is implemented based on [TensorFlow tutorial on
MNIST](https://www.tensorflow.org/get_started/mnist/pros). It is a two-layer
convolutional neural network with drop-out, followed by a fully connected layer
of size 1024 and finally by a read-out layer (softmax which predicts the
probability of the 10 possible categories).

This model achieves an accuracy of around 99.1 after 20000 batches (of size 50).
Note that there is some variance in the results due to randomness in parameter
initialization.

### Deep Highway Network

This is an implementation of the following paper:

[Srivastava et al.: Training Very Deep Networks. NIPS 2015](https://arxiv.org/abs/1507.06228)

The authors report accuracy around 99.4 with 16 convolutional filters.
The paper does not contain all the necessary details but the authors made their
implementation available:

<https://github.com/flukeskywalker/highway-networks/blob/master/examples/highways/mnist-10layers/mnist_network.prototxt>

This model is considerably slower to train. After 1000 batches of size 70, its
accuracy is around 97.5.
