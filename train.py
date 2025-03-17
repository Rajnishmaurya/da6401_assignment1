import numpy as np
import argparse
import wandb
from keras.datasets import mnist, fashion_mnist

# wandb.init(project="assignment 1", entity="da24m015-iitm",name="Question-1")
# class_names = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
# 			   'Ankle boot']

# (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
# trainX=trainX / 255.0
# testX=testX / 255.0

# def log_images():
# 	set_images=[]
# 	set_labels=[]
# 	count=0
# 	for d in range(len(trainy)):
# 		if trainy[d]==count:
# 				set_images.append(trainX[d])
# 				set_labels.append(class_names[trainy[d]])
# 				count=count+1
# 		else:
# 				pass
# 		if count==10:
# 			break

# 	wandb.log({"Plot": [wandb.Image(img, caption=caption) for img, caption in zip(set_images, set_labels)]})
# log_images()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST or Fashion-MNIST dataset')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset and training parameters
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size used to train neural network')
    
    # Loss function
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    
    # Optimizer and related parameters
    parser.add_argument('-o', '--optimizer', default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers')
    
    # Weight parameters
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'Xavier'], help='Weight initialization method')
    
    # Network architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', default='sigmoid', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function')
    
    args = parser.parse_args()
    
    # Convert 'Xavier' to 'xavier' and 'ReLU' to 'relu' for consistency
    if args.weight_init == 'Xavier':
        args.weight_init = 'xavier'
    if args.activation == 'ReLU':
        args.activation = 'relu'
    elif args.activation == 'identity':
        args.activation = 'linear'  # Using 'linear' as our internal name for identity activation
    
    # Map 'nag' to 'nesterov' for consistency with your implementation
    if args.optimizer == 'nag':
        args.optimizer = 'nesterov'
    
    # Load the specified dataset
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        class_names = [str(i) for i in range(10)]  # 0-9 for MNIST
    else:  # fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Preprocess the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    num_classes = 10
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "beta": args.beta,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.epsilon,
            "weight_decay": args.weight_decay,
            "weight_init": args.weight_init,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "activation": args.activation
        }
    )
    
    # One-hot encode the labels
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    y_train = one_hot_encode(y_train, num_classes)
    y_test = one_hot_encode(y_test, num_classes)
    
    # Split training data into train and validation sets
    split_idx = int(0.9 * len(x_train))
    x_train, x_val = x_train[:split_idx], x_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]
    y_train_original, y_val_original = y_train_original[:split_idx], y_train_original[split_idx:]
    
    # Create network architecture
    architecture = [x_train.shape[1]] + [args.hidden_size] * args.num_layers + [num_classes]
    
    # Train the network
    print(f"Training network with {args.num_layers} hidden layers, {args.hidden_size} neurons, {args.activation} activation, {args.optimizer} optimizer...")
    best_weights, best_biases = train(
        x_train, y_train, x_val, y_val, y_train_original, y_val_original,
        layers=architecture,
        learning_rate=args.learning_rate,
        activation=args.activation,
        optimizer=args.optimizer,
        weight_init=args.weight_init,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        momentum_param=args.momentum,
        loss_function=args.loss
    )
    
    # Log confusion matrices
    log_confusion_matrices(best_weights, best_biases, args.activation, x_train, y_train_original, x_test, y_test_original, class_names)
    
    # Evaluate on test set
    test_predictions = predict(x_test, best_weights, best_biases, args.activation)
    test_accuracy = np.mean(test_predictions == y_test_original)
    wandb.log({"test_accuracy": test_accuracy})
    
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print("Training complete. Results logged to Weights & Biases.")

# One-hot encode the labels
def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

# Activation Functions and their derivartives
def linear(Z):
    return Z

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    # Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

    
def sigmoid_derivative(self, A):
    """Derivative of sigmoid function"""
    return A * (1 - A)
    
def tanh_derivative(self, A):
    """Derivative of tanh function"""
    return 1 - A**2
        
def relu_derivative(self, A):
    """Derivative of ReLU function"""
    return np.where(A > 0, 1, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# Optimizer functions - UPDATED
#sgd optimizer
def sgd(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads_W[i]
        biases[i] -= learning_rate * grads_b[i]
    return weights, biases, [], [], [], []


#momentum optimizer
def momentum(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    momentum_param = kwargs.get('momentum_param', 0.9)
    if not v_W:
        v_W = [np.zeros_like(W) for W in weights]
        v_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        v_W[i] = momentum_param * v_W[i] - learning_rate * grads_W[i]
        v_b[i] = momentum_param * v_b[i] - learning_rate * grads_b[i]
        
        weights[i] += v_W[i]
        biases[i] += v_b[i]
    return weights, biases, v_W, v_b, [], []

#nesterov optimizer
def nesterov(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    momentum_param = kwargs.get('momentum_param', 0.9)
    if not v_W:
        v_W = [np.zeros_like(W) for W in weights]
        v_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        # Compute lookahead position
        lookahead_W = weights[i] + momentum_param * v_W[i]
        lookahead_b = biases[i] + momentum_param * v_b[i]
        
        # Update velocity
        v_W[i] = momentum_param * v_W[i] - learning_rate * grads_W[i]
        v_b[i] = momentum_param * v_b[i] - learning_rate * grads_b[i]
        
        # Update weights and biases with corrected lookahead step
        weights[i] = lookahead_W + v_W[i]
        biases[i] = lookahead_b + v_b[i]
    return weights, biases, v_W, v_b, [], []

#rmsprop optimizer
def rmsprop(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    beta = kwargs.get('beta', 0.9)
    epsilon = kwargs.get('epsilon', 1e-6)
    
    if not v_W:
        v_W = [np.zeros_like(W) for W in weights]
        v_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        # Update velocity for weights and biases separately
        v_W[i] = beta * v_W[i] + (1 - beta) * (grads_W[i] ** 2)
        v_b[i] = beta * v_b[i] + (1 - beta) * (grads_b[i] ** 2)
        
        # Update weights
        weights[i] -= learning_rate * grads_W[i] / (np.sqrt(v_W[i]) + epsilon)
        
        # Update biases
        biases[i] -= learning_rate * grads_b[i] / (np.sqrt(v_b[i]) + epsilon)
    return weights, biases, v_W, v_b, [], []

#adam optimizer
def adam(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    beta1 = kwargs.get('beta1', 0.9)
    beta2 = kwargs.get('beta2', 0.999)
    epsilon = kwargs.get('epsilon', 1e-6)
    t = kwargs.get('t', 1)
    
    if not v_W:
        v_W = [np.zeros_like(W) for W in weights]
        v_b = [np.zeros_like(b) for b in biases]
        moment2_W = [np.zeros_like(W) for W in weights]
        moment2_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        # First moment estimate
        v_W[i] = beta1 * v_W[i] + (1 - beta1) * grads_W[i]
        v_b[i] = beta1 * v_b[i] + (1 - beta1) * grads_b[i]
        
        # Second moment estimate
        moment2_W[i] = beta2 * moment2_W[i] + (1 - beta2) * (grads_W[i] ** 2)
        moment2_b[i] = beta2 * moment2_b[i] + (1 - beta2) * (grads_b[i] ** 2)
        
        # Bias correction
        v_W_corrected = v_W[i] / (1 - beta1 ** t)
        v_b_corrected = v_b[i] / (1 - beta1 ** t)
        
        moment2_W_corrected = moment2_W[i] / (1 - beta2 ** t)
        moment2_b_corrected = moment2_b[i] / (1 - beta2 ** t)
        
        # Check and correct shape mismatch
        if moment2_b_corrected.shape != biases[i].shape:
            moment2_b_corrected = np.reshape(moment2_b_corrected, biases[i].shape)
        
        # Parameter update
        weights[i] -= learning_rate * v_W_corrected / (np.sqrt(moment2_W_corrected) + epsilon)
        biases[i] -= learning_rate * v_b_corrected / (np.sqrt(moment2_b_corrected) + epsilon)
    
    return weights, biases, v_W, v_b, moment2_W, moment2_b

#nadam optimizer
def nadam(weights, biases, grads_W, grads_b, learning_rate, v_W=None, v_b=None, moment2_W=None, moment2_b=None, **kwargs):
    beta1 = kwargs.get('beta1', 0.9)
    beta2 = kwargs.get('beta2', 0.999)
    epsilon = kwargs.get('epsilon', 1e-6)
    t = kwargs.get('t', 1)
    
    if not v_W:
        v_W = [np.zeros_like(W) for W in weights]
        v_b = [np.zeros_like(b) for b in biases]
        moment2_W = [np.zeros_like(W) for W in weights]
        moment2_b = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        # First moment estimate
        v_W[i] = beta1 * v_W[i] + (1 - beta1) * grads_W[i]
        v_b[i] = beta1 * v_b[i] + (1 - beta1) * grads_b[i]
        
        # Second moment estimate
        moment2_W[i] = beta2 * moment2_W[i] + (1 - beta2) * (grads_W[i] ** 2)
        moment2_b[i] = beta2 * moment2_b[i] + (1 - beta2) * (grads_b[i] ** 2)
        
        # Bias correction
        v_W_corrected = (beta1 * v_W[i] + (1 - beta1) * grads_W[i]) / (1 - beta1 ** t)
        v_b_corrected = (beta1 * v_b[i] + (1 - beta1) * grads_b[i]) / (1 - beta1 ** t)
        
        moment2_W_corrected = moment2_W[i] / (1 - beta2 ** t)
        moment2_b_corrected = moment2_b[i] / (1 - beta2 ** t)
        
        # Parameter update
        weights[i] -= learning_rate * v_W_corrected / (np.sqrt(moment2_W_corrected) + epsilon)
        biases[i] -= learning_rate * v_b_corrected / (np.sqrt(moment2_b_corrected) + epsilon)
    
    return weights, biases, v_W, v_b, moment2_W, moment2_b

# Initialize network weights and biases
def init_weights(layers, method="random"):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        if method.lower() == "xavier":
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
        else:  # Default to "random"
            limit = 0.1
        W = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
        weights.append(W)
        biases.append(np.zeros((1, layers[i+1])))
    return weights, biases

# Map activation function names to functions
activation_functions = {
    "linear": linear,
    "relu": relu, 
    "sigmoid": sigmoid, 
    "tanh": tanh
}

# Map optimizer names to functions
optimizer_functions = {
    "sgd": sgd,
    "momentum": momentum,
    "nesterov": nesterov,
    "rmsprop": rmsprop,
    "adam": adam,
    "nadam": nadam
}

# Forward pass
def forward(X, weights, biases, activation):
    A = [X]
    for i in range(len(weights) - 1):
        Z = A[-1] @ weights[i] + biases[i]
        A.append(activation_functions[activation](Z))
    Z = A[-1] @ weights[-1] + biases[-1]
    A.append(softmax(Z))
    return A

# Compute loss
def compute_loss(y_true, y_pred, weights, weight_decay, loss_function="cross_entropy"):
    if loss_function == "cross_entropy":  
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    else:  # mean_squared_error
        loss = np.mean(np.sum((y_true - y_pred) ** 2, axis=1)) / 2
    
    # Add L2 regularization
    if weight_decay > 0:
        loss += (weight_decay / 2) * sum(np.sum(W**2) for W in weights)
    
    return loss

# Backward pass
def backward(X, y, A, weights, weight_decay, activation, loss_function="cross_entropy"):
    grads_W, grads_b = [], []
    
    # For MSE loss with softmax, we need to adjust the gradient calculation
    if loss_function == "mean_squared_error":
        dA = 2*(A[-1] - y)
    else:  # cross_entropy (with softmax, the gradient is simplified)
        dA = A[-1] - y
    
    for i in reversed(range(len(weights))):
        dW = A[i].T @ dA / X.shape[0]
        db = np.sum(dA, axis=0, keepdims=True) / X.shape[0]
        
        # Add L2 regularization gradient
        if weight_decay > 0:
            dW += weight_decay * weights[i]
        
        grads_W.append(dW)
        grads_b.append(db)
        
        if i > 0:
            if activation == "relu":
                dA = (dA @ weights[i].T) * (A[i] > 0)
            elif activation == "sigmoid":
                dA = (dA @ weights[i].T) * (A[i] * (1 - A[i]))
            elif activation == "tanh":
                dA = (dA @ weights[i].T) * (1 - A[i]**2)
            elif activation == "linear":
                dA = dA @ weights[i].T
    
    return grads_W[::-1], grads_b[::-1]

# Get predictions from the model
def predict(X, weights, biases, activation):
    A = forward(X, weights, biases, activation)
    return np.argmax(A[-1], axis=1)

# Train function
def train(X_train, y_train, X_val, y_val, y_train_original, y_val_original, layers, learning_rate, activation, optimizer, 
          weight_init, weight_decay, epochs, batch_size, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-6, momentum_param=0.9, loss_function="cross_entropy"):
    
    # Initialize weights and biases
    weights, biases = init_weights(layers, weight_init)
    
    # Initialize optimizer-specific parameters
    v_W = []
    v_b = []
    moment2_W = []
    moment2_b = []
    t = 1  # Timestep for Adam/Nadam
    
    num_samples = X_train.shape[0]
    
    # Keep track of best validation accuracy
    best_val_acc = 0
    best_weights, best_biases = None, None
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(num_samples)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        y_train_original_shuffled = y_train_original[indices]
        
        total_loss, total_acc = 0, 0
        num_batches = num_samples // batch_size
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # Forward Pass
            A = forward(X_batch, weights, biases, activation)
            y_pred = A[-1]
            
            # Compute Loss & Accuracy
            loss = compute_loss(y_batch, y_pred, weights, weight_decay, loss_function)
            acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            
            total_loss += loss * len(X_batch)
            total_acc += acc * len(X_batch)
            
            # Backward Pass
            grads_W, grads_b = backward(X_batch, y_batch, A, weights, weight_decay, activation, loss_function)
            
            # Update Weights using the selected optimizer
            weights, biases, v_W, v_b, moment2_W, moment2_b = optimizer_functions[optimizer](
                weights, biases, grads_W, grads_b, learning_rate, 
                v_W, v_b, moment2_W, moment2_b,
                beta=beta, beta1=beta1, beta2=beta2, epsilon=epsilon,
                momentum_param=momentum_param, t=t
            )
            
            # Increment timestep for Adam/Nadam
            if optimizer in ["adam", "nadam"]:
                t += 1
        
        # Compute average loss and accuracy for the epoch
        avg_loss = total_loss / num_samples
        avg_acc = total_acc / num_samples
        
        # Validation Metrics
        val_A = forward(X_val, weights, biases, activation)
        val_pred = val_A[-1]
        val_loss = compute_loss(y_val, val_pred, weights, weight_decay, loss_function)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
        
        # Log to Weights & Biases
        wandb.log({
            "epoch": epoch + 1, 
            "loss": avg_loss, 
            "accuracy": avg_acc,
            "val_loss": val_loss, 
            "val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
    
    # Return the best model weights and biases
    return best_weights, best_biases

# Function to create and log confusion matrices
def log_confusion_matrices(weights, biases, activation, x_train, y_train_original, x_test, y_test_original, class_names):
    # Get predictions for training data
    train_predictions = predict(x_train, weights, biases, activation)
    
    # Get predictions for test data
    test_predictions = predict(x_test, weights, biases, activation)
    
    # Log the training confusion matrix
    wandb.log({
        "train_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_train_original, 
            preds=train_predictions,
            class_names=class_names
        )
    })
    
    # Log the test confusion matrix
    wandb.log({
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test_original, 
            preds=test_predictions,
            class_names=class_names
        )
    })

if __name__ == "__main__":
    main()    