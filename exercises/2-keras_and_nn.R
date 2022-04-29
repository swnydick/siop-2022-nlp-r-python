#########################################
# Intro to Keras and Neural Nets        #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2022-04-15                            #
#########################################

# Some links
# - https://keras.rstudio.com/articles/guide_keras.html
# - https://ruder.io/optimizing-gradient-descent/
# - https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
# - https://towardsdatascience.com/the-unreasonable-progress-of-deep-neural-networks-in-natural-language-processing-nlp-374443b21b00

# 1. Setup / Required Packages =================================================

# environment name for SIOP workshop (same as before)
siop_env <- "r-siop-nlp"

# install tensorflow
tensorflow::install_tensorflow(method  = "conda",
                               envname = siop_env)

require(keras)
require(ggplot2)
require(mlbench)
require(caret)

# setting to the appropriate environment
reticulate::use_condaenv(condaenv = siop_env,
                         required = TRUE)

# 2. What is Keras? =================================================

# define a simple tensor in Keras
set.seed(3125)
a <- k_constant(matrix(sample(1:10, 6), 2, 3, byrow = TRUE)); a # this will take a few seconds as Keras/tf starts

set.seed(3125)
a <- k_constant(sample(1:10, 6), shape = c(2, 3)); a             # equivalent

b <- k_constant(sample(1:10, 6), shape = c(3, 2)); b

# perform some basic tensor operations
k_dot(a, b)
k_flatten(a)
k_mean(a)
k_min(a)
k_max(a)

# generate random values
c <- k_random_normal(c(3, 3)); c

# quick look at some activation functions (see next section)
k_relu(c)     # less than 0 set to 0
k_tanh(c)     # [e^(2x) - 1]/[e^(2x) + 1]
k_sigmoid(c)  # e^x / (1 + e^x)


# 3. Neural Networks =================================================

### Implementing a basic feed-forward neural network for regression in R with Keras

# look at the data and fixing a column
data(BostonHousing)
dim(BostonHousing)
head(BostonHousing)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))

# Split into train and test data
bh_partition <- createDataPartition(BostonHousing$medv,
                                    p    = 0.8,
                                    list = FALSE)

bh_train     <- BostonHousing[ bh_partition, ]
bh_test      <- BostonHousing[-bh_partition, ]

# First let's try a simple, OLS linear regression:

lm_model  <- train(medv ~ .,
                   data   = bh_train,
                   method = "lm")
y_pred_lm <- predict(lm_model, bh_test[ , 1:13])

cor(y_pred_lm, bh_test[ , 14])

# note RMSE, MAE
lm_model

ggplot(data.frame(y_pred_lm, y_test = bh_test[ , 14]),
       aes(y = y_pred_lm, x = y_test)) + 
  geom_point() +
  theme_minimal()



# Now solve regression problem with neural net:

# convert train and test sets into matrices for keras
bh_x_train <- as.matrix(bh_train[ , 1:13])
bh_x_test  <- as.matrix(bh_test[ , 1:13])
bh_y_train <- as.matrix(bh_train[ , 14])
bh_y_test  <- as.matrix(bh_test[ , 14])

n_x = dim(bh_x_train)[2] # whatever dimension input data is

# Input layer
nn_reg_input <- layer_input(shape = n_x)

# Hidden layers
nn_reg_hidden <- nn_reg_input |>
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
nn_reg_out   <- layer_dense(object     = nn_reg_hidden,
                            units      = 1, 
                            activation = "linear",
                            name       = "output")

# put together (input layer and output layer)
nn_reg_model <- keras_model(nn_reg_input, nn_reg_out)

summary(nn_reg_model)

# Compile model
compile(nn_reg_model,
        optimizer = optimizer_adam(),
        loss      = "mean_squared_error",
        metrics   = c("mean_absolute_error", "mean_squared_error")
)

# Train model

nn_reg_history <- fit(object          = nn_reg_model, # the model
                      x               = bh_x_train,   # the training input 
                      y               = bh_y_train,   # the training output
                      batch_size      = 25,           # the batch size
                      validation_data = list(bh_x_test, bh_y_test), # LIST of validation x, y
                      epochs          = 50,        # transits of training data through algorithm
                      shuffle         = TRUE,      # shuffle data before each epoch
                      view_metrics    = TRUE,      # plot how it's doing
                      verbose         = 1)         # progress bar

print(nn_reg_history)

y_pred_nn_reg <- predict(nn_reg_model, bh_x_test)
cor(y_pred_nn_reg, bh_y_test)

ggplot(data.frame(y_pred_nn_reg, bh_y_test),
       aes(y = y_pred_nn_reg, x = bh_y_test)) + 
  geom_point() +
  theme_minimal()

# how did this do compared to the regression model?
cor(y_pred_lm, bh_test[ , 14])

#################################################################################

### Now lets try a classification model

data(PimaIndiansDiabetes)
dim(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)

# Split into train and test data
pi_partition <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                    p    = 0.8,
                                    list = FALSE)

pi_train     <- PimaIndiansDiabetes[ pi_partition, ]
pi_test      <- PimaIndiansDiabetes[-pi_partition, ]

# Start with simple logistic regression model:

logit_model <- train(factor(diabetes) ~ .,
                     data   = pi_train,
                     method = "glm",
                     family = "binomial")
y_pred_logm <- predict(logit_model, pi_test[ , 1:8])

confusionMatrix(y_pred_logm, pi_test[, 9]) # note Accuracy, Kappa


# Now lets try a neural network for binary classfication:

# convert train and test sets into matrices for keras
pi_x_train <- as.matrix(pi_train[ , 1:8])
pi_x_test  <- as.matrix(pi_test[ , 1:8])
pi_y_train <- to_categorical(as.integer(pi_train[ , 9]) - 1, dtype = "int32")
pi_y_test  <- to_categorical(as.integer(pi_test[ , 9]) - 1, dtype = "int32")

n_x = dim(pi_x_train)[2] # whatever dimension input data is

# Input layer
nn_class_input <- layer_input(shape = n_x)

# Hidden layers
nn_class_hidden <- nn_class_input |>
  layer_dense(units      = 32, 
              activation = "tanh", 
              name       = "hidden0") |>
  layer_dense(units      = 8, 
              activation = "relu", 
              name       = "hidden1") 

# Output layer
nn_class_out   <- layer_dense(object     = nn_class_hidden,
                              units      = 2, 
                              activation = "sigmoid",
                              name       = "output")

# put together (input layer and output layer)
nn_class_model <- keras_model(nn_class_input, nn_class_out)

summary(nn_class_model)

# Compile model
compile(nn_class_model,
        optimizer = optimizer_adam(),
        loss      = "binary_crossentropy",
        metrics   = c("accuracy", "binary_crossentropy")
)

# Train model

nn_class_history <- fit(object          = nn_class_model, # the model
                         x               = pi_x_train, # the training input 
                         y               = pi_y_train,   # the training output
                         batch_size      = 32,      # the batch size
                         validation_data = list(pi_x_test, pi_y_test), # LIST of validation x, y
                         epochs          = 50,       # transits of training data through algorithm
                         shuffle         = TRUE,      # shuffle data before each epoch
                         view_metrics    = TRUE,      # plot how it's doing
                         verbose         = 1)         # progress bar

print(nn_class_history)

y_pred_nn_class <- predict(nn_class_model, pi_x_test) %>%
                   k_argmax() %>%
                   as.matrix %>%
                   as.factor
pi_y_test       <- pi_y_test %>%
                   k_argmax() %>%
                   k_cast("int32") %>%
                   as.matrix %>%
                   as.factor

confusionMatrix(y_pred_nn_class, pi_y_test)

# compare to logit (not accuracy/kappa)
confusionMatrix(y_pred_logm, pi_test[, 9])

