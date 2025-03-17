import numpy as np
from keras.datasets import fashion_mnist
import wandb

# from keras.datasets import fashion_mnist
# import wandb

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


# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-4]},
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10]},
        'hidden_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'activation': {'values': ['relu', 'sigmoid', 'tanh']},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'weight_init': {'values': ['random', 'xavier']},
        'weight_decay': {"values": [0, 0.0005, 0.5]}
    },
    "run_cap": 5
}

# Load and preprocess the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
num_classes = 10

# Define class names for the Fashion-MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# One-hot encode the labels
def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

# Keep original labels for confusion matrix
y_train_original = y_train.copy()
y_test_original = y_test.copy()

y_train, y_test = one_hot_encode(y_train, num_classes), one_hot_encode(y_test, num_classes)

# Split training data into train and validation sets
split_idx = int(0.9 * len(x_train))
x_train, x_val = x_train[:split_idx], x_train[split_idx:]
y_train, y_val = y_train[:split_idx], y_train[split_idx:]
y_train_original, y_val_original = y_train_original[:split_idx], y_train_original[split_idx:]

# Activation Functions
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

activation_functions = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh}

# Optimizer functions
def sgd(weights, biases, grads_W, grads_b, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads_W[i]
        biases[i] -= learning_rate * grads_b[i]
    return weights, biases

def momentum(weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, momentum=0.9):
    for i in range(len(weights)):
        velocity_W[i] = momentum * velocity_W[i] - learning_rate * grads_W[i]
        velocity_b[i] = momentum * velocity_b[i] - learning_rate * grads_b[i]

        weights[i] += velocity_W[i]
        biases[i] += velocity_b[i]
    return weights, biases, velocity_W, velocity_b

def nesterov(weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, momentum=0.9):
    for i in range(len(weights)):
        # Compute lookahead position
        lookahead_W = weights[i] + momentum * velocity_W[i]
        lookahead_b = biases[i] + momentum * velocity_b[i]

        # Update velocity
        velocity_W[i] = momentum * velocity_W[i] - learning_rate * grads_W[i]
        velocity_b[i] = momentum * velocity_b[i] - learning_rate * grads_b[i]

        # Update weights and biases with corrected lookahead step
        weights[i] = lookahead_W + velocity_W[i]
        biases[i] = lookahead_b + velocity_b[i]
    return weights, biases, velocity_W, velocity_b

def rmsprop(weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, beta=0.9, epsilon=1e-6):
    for i in range(len(weights)):
        # Update velocity for weights and biases separately
        velocity_W[i] = beta * velocity_W[i] + (1 - beta) * (grads_W[i] ** 2)
        velocity_b[i] = beta * velocity_b[i] + (1 - beta) * (grads_b[i] ** 2)

        # Update weights
        weights[i] -= learning_rate * grads_W[i] / (np.sqrt(velocity_W[i]) + epsilon)

        # Update biases
        biases[i] -= learning_rate * grads_b[i] / (np.sqrt(velocity_b[i]) + epsilon)
    return weights, biases, velocity_W, velocity_b

def adam(weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, moment2_W, moment2_b, t, beta1=0.9, beta2=0.999, epsilon=1e-6):
    for i in range(len(weights)):
        # First moment estimate
        velocity_W[i] = beta1 * velocity_W[i] + (1 - beta1) * grads_W[i]
        velocity_b[i] = beta1 * velocity_b[i] + (1 - beta1) * grads_b[i]

        # Second moment estimate
        moment2_W[i] = beta2 * moment2_W[i] + (1 - beta2) * (grads_W[i] ** 2)
        moment2_b[i] = beta2 * moment2_b[i] + (1 - beta2) * (grads_b[i] ** 2)

        # Bias correction
        velocity_W_corrected = velocity_W[i] / (1 - beta1 ** t)
        velocity_b_corrected = velocity_b[i] / (1 - beta1 ** t)

        moment2_W_corrected = moment2_W[i] / (1 - beta2 ** t)
        moment2_b_corrected = moment2_b[i] / (1 - beta2 ** t)

        # Check and correct shape mismatch
        if moment2_b_corrected.shape != biases[i].shape:
            print(f"Shape mismatch at layer {i}: {moment2_b_corrected.shape} vs {biases[i].shape}")
            moment2_b_corrected = np.reshape(moment2_b_corrected, biases[i].shape)

        # Parameter update
        weights[i] -= learning_rate * velocity_W_corrected / (np.sqrt(moment2_W_corrected) + epsilon)
        biases[i] -= learning_rate * velocity_b_corrected / (np.sqrt(moment2_b_corrected) + epsilon)
    
    return weights, biases, velocity_W, velocity_b, moment2_W, moment2_b

def nadam(weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, moment2_W, moment2_b, t, beta1=0.9, beta2=0.999, epsilon=1e-6):
    for i in range(len(weights)):
        # First moment estimate
        velocity_W[i] = beta1 * velocity_W[i] + (1 - beta1) * grads_W[i]
        velocity_b[i] = beta1 * velocity_b[i] + (1 - beta1) * grads_b[i]

        # Second moment estimate
        moment2_W[i] = beta2 * moment2_W[i] + (1 - beta2) * (grads_W[i] ** 2)
        moment2_b[i] = beta2 * moment2_b[i] + (1 - beta2) * (grads_b[i] ** 2)

        # Bias correction
        velocity_W_corrected = (beta1 * velocity_W[i] + (1 - beta1) * grads_W[i]) / (1 - beta1 ** t)
        velocity_b_corrected = (beta1 * velocity_b[i] + (1 - beta1) * grads_b[i]) / (1 - beta1 ** t)

        moment2_W_corrected = moment2_W[i] / (1 - beta2 ** t)
        moment2_b_corrected = moment2_b[i] / (1 - beta2 ** t)

        # Parameter update
        weights[i] -= learning_rate * velocity_W_corrected / (np.sqrt(moment2_W_corrected) + epsilon)
        biases[i] -= learning_rate * velocity_b_corrected / (np.sqrt(moment2_b_corrected) + epsilon)
    
    return weights, biases, velocity_W, velocity_b, moment2_W, moment2_b

# Initialize network weights and biases
def init_weights(layers, method="random"):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        if method == "xavier":
            limit = np.sqrt(6 / (layers[i] + layers[i+1]))
        else:  # Default to "random"
            limit = 0.1
        W = np.random.uniform(-limit, limit, (layers[i], layers[i+1]))
        weights.append(W)
        biases.append(np.zeros((1, layers[i+1])))
    return weights, biases

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
def compute_loss(y_true, y_pred, weights, weight_decay):
    loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    loss += (weight_decay / 2) * sum(np.sum(W**2) for W in weights)
    return loss

# Backward pass
def backward(X, y, A, weights, weight_decay, activation):
    grads_W, grads_b = [], []
    dA = A[-1] - y
    
    for i in reversed(range(len(weights))):
        dW = A[i].T @ dA / X.shape[0]
        db = np.sum(dA, axis=0, keepdims=True) / X.shape[0]
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
    
    return grads_W[::-1], grads_b[::-1]

# Get predictions from the model
def predict(X, weights, biases, activation):
    A = forward(X, weights, biases, activation)
    return np.argmax(A[-1], axis=1)

# Train function
def train(X_train, y_train, X_val, y_val, y_train_original, y_val_original, layers, learning_rate, activation, optimizer, 
          weight_init, weight_decay, epochs, batch_size, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-6):
    
    # Initialize weights and biases
    weights, biases = init_weights(layers, weight_init)
    
    # Initialize optimizer-specific parameters
    velocity_W = [np.zeros_like(W) for W in weights]
    velocity_b = [np.zeros_like(b) for b in biases]
    moment2_W = [np.zeros_like(W) for W in weights]
    moment2_b = [np.zeros_like(b) for b in biases]
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
            loss = compute_loss(y_batch, y_pred, weights, weight_decay)
            acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            
            total_loss += loss * len(X_batch)
            total_acc += acc * len(X_batch)
            
            # Backward Pass
            grads_W, grads_b = backward(X_batch, y_batch, A, weights, weight_decay, activation)
            
            # Update Weights using the selected optimizer
            if optimizer == "sgd":
                weights, biases = sgd(weights, biases, grads_W, grads_b, learning_rate)
            elif optimizer == "momentum":
                weights, biases, velocity_W, velocity_b = momentum(
                    weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b)
            elif optimizer == "nesterov":
                weights, biases, velocity_W, velocity_b = nesterov(
                    weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b)
            elif optimizer == "rmsprop":
                weights, biases, velocity_W, velocity_b = rmsprop(
                    weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, beta, epsilon)
            elif optimizer == "adam":
                weights, biases, velocity_W, velocity_b, moment2_W, moment2_b = adam(
                    weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, 
                    moment2_W, moment2_b, t, beta1, beta2, epsilon)
                t += 1
            elif optimizer == "nadam":
                weights, biases, velocity_W, velocity_b, moment2_W, moment2_b = nadam(
                    weights, biases, grads_W, grads_b, learning_rate, velocity_W, velocity_b, 
                    moment2_W, moment2_b, t, beta1, beta2, epsilon)
                t += 1
        
        # Compute average loss and accuracy for the epoch
        avg_loss = total_loss / num_samples
        avg_acc = total_acc / num_samples
        
        # Validation Metrics
        val_A = forward(X_val, weights, biases, activation)
        val_pred = val_A[-1]
        val_loss = compute_loss(y_val, val_pred, weights, weight_decay)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
        
        # Log to Weights & Biases
        wandb.log({
            "epoch": epoch + 1, 
            "loss": avg_loss, 
            "accuracy": avg_acc,
            "val_loss": val_loss, 
            "val_accuracy": val_acc
        })
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
    
    # Return the best model weights and biases
    return best_weights, best_biases

# Function to create and log confusion matrices
def log_confusion_matrices(weights, biases, activation):
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

# Function to run a single training job with wandb logging
def train_model():
    # Initialize wandb for this run
    wandb.init(project="assignment_1", entity="da24m015-iitm")
    config = wandb.config
    wandb.run.name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}"
    
    # Create network architecture based on config
    architecture = [784] + [config.hidden_size] * config.hidden_layers + [10]
    
    # Train the network and get the best model
    best_weights, best_biases = train(
        x_train, y_train, x_val, y_val, y_train_original, y_val_original,
        layers=architecture,
        learning_rate=config.learning_rate,
        activation=config.activation,
        optimizer=config.optimizer,
        weight_init=config.weight_init,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        batch_size=config.batch_size
    )
    # print(f"Best_weights:{best_weights},Best_biases:{best_biases}")
    # Log confusion matrices for the best model
    log_confusion_matrices(best_weights, best_biases, config.activation)

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="assignment_1")

# Run the sweep
wandb.agent(sweep_id, function=train_model)