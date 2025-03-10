# Introduction-to-Deep-Learning-DA6401-

# Assignment 1: Neural Network Implementation for MNIST and Fashion-MNIST

In this project I have implemented a configurable feedforward neural network from scratch using NumPy, designed to train on MNIST and Fashion-MNIST datasets. It supports various optimizers, activation functions, loss functions, and network architectures.

## Features

- Support for both MNIST and Fashion-MNIST datasets
- Flexible network architecture with configurable hidden layers and neurons
- Multiple activation functions: Sigmoid, Tanh, ReLU, Linear (Identity)
- Various optimizers: SGD, Momentum, Nesterov Accelerated Gradient, RMSProp, Adam, NAdam
- Loss functions: Cross-Entropy and Mean Squared Error
- Weight initialization techniques: Random and Xavier
- L2 regularization (weight decay)
- Batch training
- Experiment tracking with Weights & Biases
- Confusion matrix visualization

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install numpy argparse wandb tensorflow
```

## Usage

Run the script with desired parameters:

```bash
python train.py --dataset fashion_mnist --epochs 10 --batch_size 32 --loss cross_entropy --optimizer adam --learning_rate 0.001 --num_layers 2 --hidden_size 128 --activation relu
```

### Command Line Arguments

#### Weights & Biases Configuration
- `-wp`, `--wandb_project`: Project name for Weights & Biases (default: 'myprojectname')
- `-we`, `--wandb_entity`: Entity name for Weights & Biases (default: 'myname')

#### Dataset and Training
- `-d`, `--dataset`: Dataset to use ('mnist' or 'fashion_mnist', default: 'fashion_mnist')
- `-e`, `--epochs`: Number of epochs to train (default: 1)
- `-b`, `--batch_size`: Batch size for training (default: 4)

#### Loss Function
- `-l`, `--loss`: Loss function ('mean_squared_error' or 'cross_entropy', default: 'cross_entropy')

#### Optimizer and Learning Parameters
- `-o`, `--optimizer`: Optimizer to use ('sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam', default: 'sgd')
- `-lr`, `--learning_rate`: Learning rate (default: 0.1)
- `-m`, `--momentum`: Momentum parameter for momentum and NAG optimizers (default: 0.5)
- `-beta`, `--beta`: Beta parameter for RMSProp (default: 0.5)
- `-beta1`, `--beta1`: Beta1 parameter for Adam and NAdam (default: 0.5)
- `-beta2`, `--beta2`: Beta2 parameter for Adam and NAdam (default: 0.5)
- `-eps`, `--epsilon`: Epsilon parameter for optimizers (default: 0.000001)

#### Regularization and Weight Initialization
- `-w_d`, `--weight_decay`: L2 regularization parameter (default: 0.0)
- `-w_i`, `--weight_init`: Weight initialization method ('random' or 'Xavier', default: 'random')

#### Network Architecture
- `-nhl`, `--num_layers`: Number of hidden layers (default: 1)
- `-sz`, `--hidden_size`: Number of neurons per hidden layer (default: 4)
- `-a`, `--activation`: Activation function ('identity', 'sigmoid', 'tanh', 'ReLU', default: 'sigmoid')

## Code Structure

- **Activation Functions**: Implementation of sigmoid, tanh, ReLU, and linear activation functions with their derivatives
- **Optimizer Functions**: Implementation of SGD, Momentum, Nesterov, RMSProp, Adam, and NAdam optimizers
- **Weight Initialization**: Random and Xavier initialization methods
- **Forward Pass**: Implementation of feed-forward propagation
- **Backward Pass**: Implementation of backpropagation with support for different activation functions
- **Loss Computation**: Cross-entropy and mean squared error with L2 regularization
- **Training Loop**: Batch training with validation and tracking of best model
- **Metrics Tracking**: Integration with Weights & Biases for experiment tracking and visualization

## Visualization

The implementation logs various metrics to Weights & Biases:
- Training and validation loss
- Training and validation accuracy
- Confusion matrices for training and test sets
- Final test accuracy

## Example

```bash
python train.py --dataset fashion_mnist --epochs 20 --batch_size 64 --loss cross_entropy --optimizer adam --learning_rate 0.001 --num_layers 2 --hidden_size 128 --activation relu --weight_init Xavier --weight_decay 0.0001 --wandb_project "my_fashion_mnist" --wandb_entity "my_username"
```

This command trains a neural network with 2 hidden layers of 128 neurons each, using ReLU activation, Adam optimizer, and Xavier initialization on the Fashion-MNIST dataset for 20 epochs.