import Base
from devol import genome_handler
import tensorflow as tf
from keras.datasets import cifar10

gpus = tf.config.experimental.list_physical_devices('GPU')
with tf.device('/gpu:0'):    
    Base.gpu_memory_limiter()
    cifar10_dataset = Base.train_dataset(cifar10,32,32,3)
    genome = Base.create_genome_handler(cifar10_dataset[0][0])
    Base.run_genetic_program(genome, cifar10_dataset)
