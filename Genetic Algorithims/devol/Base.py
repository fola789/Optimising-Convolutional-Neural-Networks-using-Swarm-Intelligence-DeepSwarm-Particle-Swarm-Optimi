import tensorflow as tf
from tensorflow import keras

from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K 
from devol import DEvol, GenomeHandler

# **Prepare dataset**
# This problem uses mnist,fashion_mnist and cifar10. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

K.set_image_data_format("channels_last")
gpus = tf.config.experimental.list_physical_devices('GPU')

def gpu_memory_limiter():
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5024)])
        except RuntimeError as e:
            print(e)
    
def train_dataset(dataset, num1, num2, num3):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = x_train.reshape(x_train.shape[0], num1, num2, num3).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], num1, num2, num3).astype('float32') / 255
    y_train = to_categorical(y_train) # Converts a class vector (integers) that represent different categories to binary class matrix which has columns equal to the number of categories in the data. e.g. for use with categorical_crossentropy.
    y_test = to_categorical(y_test)
    dataset = ((x_train, y_train), (x_test, y_test))
    return dataset  # return so it can be used outside of the function

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
def create_genome_handler(x_train):
    genome_handler = GenomeHandler(max_conv_layers=6, 
                               max_dense_layers=2, # includes final dense layer
                               max_filters=256,
                               max_dense_nodes=1024,
                               input_shape= x_train.shape[1:],
                               n_classes=10)
    return genome_handler

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.
# The best model is returned decoded and with `epochs` training done.
def run_genetic_program(genome_handler,dataset):
    devol = DEvol(genome_handler)
    model = devol.run(dataset=dataset,
                  num_generations=10,
                  pop_size=10,
                  epochs=3)
    print(model.summary())