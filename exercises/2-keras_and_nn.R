#########################################
# Intro to Keras and Neural Nets        #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2022-04-15                            #
#########################################

# 1. Setup / Required Packages =================================================
require(keras)
require(ggplot2)


# 2. What is Keras? =================================================

# Python API for deep learning. Can run on top of a number of deep learning
# frameworks, most notably Tensorflow. Tensorflow is a framework for
# performing operations on *tensors*, represented with *computational graphs*.
# This framework is most famous/popular for implementing neural networks.

# The R keras package provides an interface to the Python keras package.
# When you run keras in R, it requires starting up a Python environment in
# the background, where actual computations are done.

# define a simple tensor in Keras
set.seed(3125)
a <- k_constant(matrix(sample(1:10, 6), 2, 3)); a
set.seed(3125)
a <- k_constant(sample(1:10, 6), shape = c(2,3)); a #equivalent

b <- k_constant(sample(1:10, 6), shape = c(3,2)); b

# perform some basic tensor operations
k_dot(a, b)
k_flatten(a)
k_mean(a)
k_min(a)
k_max(a)

# generate random values
c <- k_random_normal(c(3,3)); c

# quick look at some activation functions (see next section)
k_relu(c)
k_tanh(c)
k_sigmoid(c)


# 3. Neural Networks =================================================

# Artificial neural networks (ANNs) are computational models conceptually 
# similar and analogized to a brain. Basic building blocks are nodes ("neurons"),
# connected with links ("synapses"). A basic neural network consists of 
# an *input layer* of n_x nodes, some number (L) of *hidden layers* with n_h^[l]
# nodes in the lth layer, and an *output layer* with n_y nodes. Commonly,
# n_y can be 1 node (regression), 2 nodes (binary classification), or multiple
# nodes (multi-class classification). There are more complex ANN architectures
# that can output text, images, and sounds.
#
# In a basic feed-forward neural network, each hidden layer has a matrix
# of *weights* (W^[l]) of dimension (n_h^[l], n_h^[l-1]) and *biases* (b^[l])
# of dimension (n_h^[l], 1). In the first hidden layer, the second dimension
# is n_x, matching the input vector. At each layer, we compute:
#
# z^[l] = W[l]*a^[l-1] + b[l],
#
# where a^[l-1] is the output vector from the previous layer, or in the case
# of a^[0], the input x. 
#
# The layer's *activation function*, typically denoted as sigma, decides whether
# the neuron "fires". Common activation functions include linear, logistic,
# tanh, ReLU and RBF.
#
# Neural networks are trained by passing labeled training data through the 
# net and comparing the output prediction y^ to the true y* by computing a 
# cost function J. Then, moving backwards through the net adjusting the weights
# biases using gradient descent, a process called *backpropagation*. 

# Other topics/parameters not covered: connectedness, pruning, dropout, 
# normalization, learning rate, momentum, pooling, many more!

# Some complex NN architectures: Convolutional NNs (images), Recurrent NNs
# (additional connections back to previous layers instead of just forward), 
# long-short term memory (LSTM) models (natural language processing) 


### Implementing a basic feed-forward neural network for regression in R with Keras

housing <- dataset_boston_housing()

# Split into train and test data
x_train <- housing$train$x
x_test  <- housing$test$x
y_train <- housing$train$y
y_test  <- housing$test$y

n_x = dim(x_train)[2] # whatever dimension input data is

# Input layer
basic_input <- layer_input(shape = n_x)

# Hidden layers
basic_hidden <- basic_input |>
  layer_dense(units      = 50, 
              activation = "relu", 
              name       = "hidden0") |>
  layer_dense(units      = 50, 
              activation = "relu", 
              name       = "hidden1")|>
  layer_dense(units      = 10, 
              activation = "relu", 
              name       = "hidden2") 

# Output layer
basic_out   <- layer_dense(object     = basic_hidden,
                         units      = 1, 
                         activation = "linear",
                         name       = "output")

# put together (input layer and output layer)
basic_model <- keras_model(basic_input, basic_out)

summary(basic_model)

# Compile model
compile(basic_model,
        optimizer = optimizer_adam(),
        loss      = "mean_squared_error",
        metrics   = c("mean_absolute_error", "cosine_proximity")
)

# Train model

basic_history <- fit(object          = basic_model, # the model
                     x               = x_train, # the training input 
                     y               = y_train,   # the training output
                     batch_size      = 50,      # the batch size
                     validation_data = list(x_test, y_test), # LIST of validation x, y
                     epochs          = 10,       # transits of training data through algorithm
                     shuffle         = TRUE,      # shuffle data before each epoch
                     view_metrics    = TRUE,      # plot how it's doing
                     verbose         = 1)         # progress bar

print(basic_history)

y_pred <- predict(basic_model, x_test)
cor(y_pred, y_test)

ggplot(data.frame(y_pred, y_test),
       aes(y = y_pred, x = y_test)) + 
  geom_point() +
  theme_minimal()

# pull out wall-of-text and put in rmd
# compare boston housing regression model to a basic LM
# add a quick classification example - throw in some different layers, activations




# 4. Transfer Learning =================================================


# 5. Autoencoders =================================================


