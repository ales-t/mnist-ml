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

Get the full command synopsis by running `python3 main.py --help`:

```
usage: main.py [-h] [--batch-size BATCH_SIZE] [--batches BATCHES]
               [--max-cpu-cores MAX_CPU_CORES] [--model-type MODEL_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --batches BATCHES     total number of batches
  --max-cpu-cores MAX_CPU_CORES
                        how many cores to use at most
  --model-type MODEL_TYPE
                        which model (basic_cnn, highway_cnn)
```


<https://github.com/flukeskywalker/highway-networks/blob/master/examples/highways/mnist-10layers/mnist_network.prototxt>
