# -*- coding: utf-8 -*-

import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.e-7

class MNISTDataset(Dataset):
    """Scikit-Learn MNIST dataset."""
    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        # takes a big chunk of the dataset for training
        if mode == 'train':
            self.data = digits.images[:1000].astype(np.float32)
            self.targets = digits.target[:1000]
        # validation dataset
        elif mode == 'val':
            self.data = digits.images[1000:1350].astype(np.float32)
            self.targets = digits.target[1000:1350]
        # test dataset
        else:
            self.data = digits.images[1350:].astype(np.float32)
            self.targets = digits.target[1350:]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return (sample_x, sample_y)

# Search space of solutions where we are trying to find the optimal solution
class SearchSpace:
    def __init__(self):
        # we can have up to 3 layers
        self.number_conv_layers = [1, 2, 3]
        # options we have for the number of filters
        self.filters_per_conv_layer = [8, 16, 32, 64]
        # options for kernel size
        self.kernel_size_per_conv_layer = [3, 5]
        # options for activation functions , noop means no activation function (if not noop activations will be used after each conv layer)
        self.activation_type_per_conv_layer = ['sigmoid', 'relu', 'tanh', 'noop']
        # options for types of pooling layers, noop means no pooling layer (if not noop pooling will be applied after each conv layer)
        self.pooling_type_per_conv_layer = ['max', 'avg', 'noop']
        # options for pooling kernel size
        self.pooling_size_per_conv_layer = [2, 3, 4]
        # options for optimizer type
        self.optimizer = ['sgd', 'adam']
        # options for training learning rate
        self.learning_rate = [0.001, 0.005, 0.01]

# State is a datapoint in the whole search space
class State:
    def __init__(self, 
                list_filter,
                list_kernel,
                list_activation,
                list_pooling_type,
                list_pooling_size,
                optimizer,
                learning_rate
                ):
        self.list_filters = list_filter
        self.list_kernel = list_kernel
        self.list_activation = list_activation
        self.list_pooling_type = list_pooling_type
        self.list_pooling_size = list_pooling_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate

# Given a search space this class can generate a random state and the neighbors of a particular state
class StateGenerator:
    def __init__(self, search_space):
        self.search_space = search_space
    
    def generate_random_state(self):
        # Randomly select number of conv layers
        number_conv_layers = random.choice(self.search_space.number_conv_layers)
        
        # For each conv layer randomly select number of filters, kernel size, activation, pooling layer type and pooling layer kernel size
        list_filters = random.choices(self.search_space.filters_per_conv_layer, k = number_conv_layers)
        list_kernel = random.choices(self.search_space.kernel_size_per_conv_layer, k = number_conv_layers)
        list_activation = random.choices(self.search_space.activation_type_per_conv_layer, k = number_conv_layers)
        list_pooling_type = random.choices(self.search_space.pooling_type_per_conv_layer, k = number_conv_layers)
        list_pooling_size = random.choices(self.search_space.pooling_size_per_conv_layer, k = number_conv_layers)
        
        # Randomly select optimizer type and learning rate
        optimizer = random.choice(self.search_space.optimizer)
        learning_rate = random.choice(self.search_space.learning_rate)

        state = State(list_filters, list_kernel, list_activation, list_pooling_type, list_pooling_size, optimizer, learning_rate)
        return state
    
    def generate_neighbors(self, start_state):
        """
          Definition of neighbor state:
            Has one less conv layer (remove last layer)
            Has one more conv layer (randomly select specs for additional layer)
            Same number of layers but change one of the specs of any layer
        """
        list_neighbors = []
        # Decrease number of layers
        if len(start_state.list_filters) > self.search_space.number_conv_layers[0]:
            state = State(start_state.list_filters[:-1], 
                          start_state.list_kernel[:-1], 
                          start_state.list_activation[:-1], 
                          start_state.list_pooling_type[:-1], 
                          start_state.list_pooling_size[:-1], 
                          start_state.optimizer, 
                          start_state.learning_rate)
            list_neighbors.append(state)
        
        # Increase number of layers
        if len(start_state.list_filters) < self.search_space.number_conv_layers[-1]:
            state = State(start_state.list_filters + [random.choice(self.search_space.filters_per_conv_layer)], 
                          start_state.list_kernel + [random.choice(self.search_space.kernel_size_per_conv_layer)], 
                          start_state.list_activation + [random.choice(self.search_space.activation_type_per_conv_layer)], 
                          start_state.list_pooling_type + [random.choice(self.search_space.pooling_type_per_conv_layer)], 
                          start_state.list_pooling_size + [random.choice(self.search_space.pooling_size_per_conv_layer)], 
                          start_state.optimizer, 
                          start_state.learning_rate)
            list_neighbors.append(state)
        
        for layer_id in range(len(start_state.list_filters)):
            # Change number of filters of the current layer
            for value in self.search_space.filters_per_conv_layer:
                if value == start_state.list_filters[layer_id]:
                    continue
                state = State(start_state.list_filters[0:layer_id] + [value] + start_state.list_filters[layer_id + 1 :], 
                          start_state.list_kernel, 
                          start_state.list_activation, 
                          start_state.list_pooling_type, 
                          start_state.list_pooling_size, 
                          start_state.optimizer, 
                          start_state.learning_rate)
                list_neighbors.append(state)

            # Change kernel size of the current layer
            for value in self.search_space.kernel_size_per_conv_layer:
                if value == start_state.list_kernel[layer_id]:
                    continue
                state = State(start_state.list_filters, 
                          start_state.list_kernel[0:layer_id] + [value] + start_state.list_kernel[layer_id + 1 :], 
                          start_state.list_activation, 
                          start_state.list_pooling_type, 
                          start_state.list_pooling_size, 
                          start_state.optimizer, 
                          start_state.learning_rate)
                list_neighbors.append(state)
            
            # Change activation type for the current layer
            for value in self.search_space.activation_type_per_conv_layer:
                if value == start_state.list_activation[layer_id]:
                    continue
                state = State(start_state.list_filters, 
                          start_state.list_kernel, 
                          start_state.list_activation[0:layer_id] + [value] + start_state.list_activation[layer_id + 1 :], 
                          start_state.list_pooling_type, 
                          start_state.list_pooling_size, 
                          start_state.optimizer, 
                          start_state.learning_rate)
                list_neighbors.append(state)
            
            # Change pooling type of the current layer
            for value in self.search_space.pooling_type_per_conv_layer:
                if value == start_state.list_pooling_type[layer_id]:
                    continue
                state = State(start_state.list_filters, 
                          start_state.list_kernel, 
                          start_state.list_activation, 
                          start_state.list_pooling_type[0:layer_id] + [value] + start_state.list_pooling_type[layer_id + 1 :], 
                          start_state.list_pooling_size, 
                          start_state.optimizer, 
                          start_state.learning_rate)
                list_neighbors.append(state)
            
            # Change pooling kernel size of the current layer
            for value in self.search_space.pooling_size_per_conv_layer:
                if value == start_state.list_pooling_size[layer_id]:
                    continue
                state = State(start_state.list_filters, 
                          start_state.list_kernel, 
                          start_state.list_activation, 
                          start_state.list_pooling_type, 
                          start_state.list_pooling_size[0:layer_id] + [value] + start_state.list_filters[layer_id + 1 :], 
                          start_state.optimizer, 
                          start_state.learning_rate)
                list_neighbors.append(state)

        # Change optimizer type
        for value in self.search_space.optimizer:
            if value == start_state.optimizer:
                continue
            state = State(start_state.list_filters, 
                        start_state.list_kernel, 
                        start_state.list_activation, 
                        start_state.list_pooling_type, 
                        start_state.list_pooling_size, 
                        value, 
                        start_state.learning_rate)
            list_neighbors.append(state)
        
        # Change learning rate
        for value in self.search_space.learning_rate:
            if value == start_state.learning_rate:
                continue
            state = State(start_state.list_filters, 
                        start_state.list_kernel, 
                        start_state.list_activation, 
                        start_state.list_pooling_type, 
                        start_state.list_pooling_size, 
                        start_state.optimizer, 
                        value)
            list_neighbors.append(state)
        
        return list_neighbors

class NeuralNet(nn.Module):
    def __init__(self, state):
        super().__init__()

        # Keep track of input channels, output width and height
        # We use padding = same and stride = 1 so conv layer does not affect the width and height
        # Pooling does change width and height which are re calculated
        # NB : not all architectures in the search space are possible because the images in MNIST are small
        # Thus it is possible that after two pooling layers the output is empty

        input_channels = 1 # Change this if we need to change the input image depth (MNIST is gray scale so only 1 input channel)
        width = 8 # Change this if we need to change the input image size (for example other dataset)
        height = 8 # Change this if we need to change the input image size (for example other dataset)

        self.layers = []
        for i in range(len(state.list_filters)):
            # Convolutional layer
            self.layers.append(
                nn.Conv2d(
                    in_channels = input_channels,
                    out_channels = state.list_filters[i],
                    kernel_size = state.list_kernel[i],
                    stride = 1,
                    padding = 'same'
                )
            )
 
            input_channels = state.list_filters[i]
            width = width
            height = height

            # Activation function
            if state.list_activation[i] == 'sigmoid':
                self.layers.append(nn.Sigmoid())

            if state.list_activation[i] == 'relu':
                self.layers.append(nn.ReLU())
            
            if state.list_activation[i] == 'tanh':
                self.layers.append(nn.Tanh())
            
            # Pooling layer
            if state.list_pooling_type[i] == 'max':
                self.layers.append(nn.MaxPool2d(state.list_pooling_size[i]))
                width = math.floor((width - state.list_pooling_size[i]) / state.list_pooling_size[i] + 1)
                height = math.floor((height - state.list_pooling_size[i]) / state.list_pooling_size[i] + 1) 
            
            if state.list_pooling_type[i] == 'avg':
                self.layers.append(nn.AvgPool2d(state.list_pooling_size[i]))
                width = math.floor((width - state.list_pooling_size[i]) / state.list_pooling_size[i] + 1)
                height = math.floor((height - state.list_pooling_size[i]) / state.list_pooling_size[i] + 1)
            
        
        flatten_output_shape = input_channels * width * height
        self.fc = nn.Linear(flatten_output_shape, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class NeuralNetTrainer:
    """
      This class takes as input the state, and training and validation data loaders
      train : trains the neural network and returns a list of validation losses and accuracy errors
      evaluation : returns the current model's validation loss and accuracy error (1.0 - accuracy)
      get_score : trains the model (if possible) and returns the accuracy of the model on the validation dataset
    """
    def __init__(self, state, train_loader, validation_loader):
        self.neural_net = NeuralNet(state)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        if state.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.neural_net.parameters(), lr = state.learning_rate)
        if state.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.neural_net.parameters(), lr = state.learning_rate)
        
        self.num_epochs = 50
        self.train_loader = train_loader
        self.validation_loader = validation_loader
    
    def train(self, verbose = False):
        nll_val = []
        error_val = []
  
        # Main training loop
        for epoch in range(self.num_epochs):
            # load batches
            for indx_batch, (batch, targets) in enumerate(self.train_loader):
                # calculate the forward pass (loss function for given images and labels)
                outputs = self.neural_net(batch)                    
                loss = self.criterion(outputs, targets)

                # remember we need to zero gradients! Just in case!
                self.optimizer.zero_grad()
                # calculate backward pass
                loss.backward()
                # run the optimizer
                self.optimizer.step()
            
            # Validation: Evaluate the model on the validation data
            loss_e, error_e = self.evaluation()
            
            if verbose:
                print("Epoch {0} : Validation Loss {1} - Validation Error {2}".format(epoch, loss_e, error_e))
            
            nll_val.append(loss_e)  # save for plotting
            error_val.append(error_e)  # save for plotting

        # Return nll and classification error.
        nll_val = np.asarray(nll_val)
        error_val = np.asarray(error_val)

        return nll_val, error_val
    
    def evaluation(self):
        
        loss_test = 0.
        loss_error = 0.
        N = 0.
        
        # start evaluation
        with torch.no_grad():
            for indx_batch, (test_batch, test_targets) in enumerate(self.validation_loader):
                # loss (nll)
                outputs = self.neural_net.forward(test_batch)
                loss_test_batch = self.criterion(outputs, test_targets)
                loss_test = loss_test + loss_test_batch.item()
    
                # classification error
                _, y_pred = torch.max(outputs, 1)
                e = 1. * (y_pred == test_targets)
                loss_error = loss_error + (1. - e).sum().item()
                # the number of examples
                N = N + test_batch.shape[0]
                # divide by the number of examples
            loss_test = loss_test / N
            loss_error = loss_error / N
            
        return loss_test, loss_error
    
    def get_score(self):
        try:
            self.train()
            _, acc_error = self.evaluation()
            return 1.0 - acc_error
        
        except Exception as e:
            # This is how we handle architectures that cannot be trained
            # errors are : Calculated output size: (Nx0x0). Output size is too small
            print("Cannot train this architecture : ", e)
            return 0.0

def simulated_annealing(state_generator, train_loader, val_loader, ):
    """Peforms simulated annealing to find a solution"""
    
    alpha = 0.8
    current_temp = 90

    # Start by initializing the current state and best state with a randomly generated state
    current_state = state_generator.generate_random_state()
    current_state_score = NeuralNetTrainer(current_state, train_loader, val_loader).get_score()

    solution = current_state
    solution_score = current_state_score

    accuracy_steps = []
    for iteration in range(50):
        
        # Get the list of neighbors of the current state
        list_neighbors = state_generator.generate_neighbors(current_state)
        
        # Select a random neighbor and compute the validation accuracy of the model
        neighbor = random.choice(list_neighbors)
        neighbor_score = NeuralNetTrainer(neighbor, train_loader, val_loader).get_score()

        # If the neighbor is better than the best so far, save it as the best
        if neighbor_score > solution_score:
            solution = neighbor
            solution_score = neighbor_score
        
        cost_diff = neighbor_score - current_state_score

        # Check if neighbor is better than the current state
        if cost_diff > 0:
            current_state = neighbor
            current_state_score = neighbor_score
            
        # if the neighbor is not better than the current, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                current_state = neighbor
                current_state_score = neighbor_score
        
        # decrement the temperature
        current_temp *= alpha
        # keep track of best solution score per iteration
        accuracy_steps.append(solution_score)
        print("Iteration : ", iteration , " Best accuracy so far : ", solution_score)

    return solution, solution_score, accuracy_steps

image_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((4.88,), (6.01,))]
)

print("Preparing Data")
train_data = MNISTDataset(mode='train', transforms = image_transform)
val_data = MNISTDataset(mode='val', transforms = image_transform)
test_data = MNISTDataset(mode='test', transforms = image_transform)

# Initialize data loaders.
training_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_data, batch_size = 64, shuffle = False)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

state_generator = StateGenerator(SearchSpace())
best_network, accuracy, sa_history = simulated_annealing(state_generator, training_loader, val_loader)

plt.plot(range(len(sa_history)), sa_history)
plt.title('SA plot')
plt.grid()

