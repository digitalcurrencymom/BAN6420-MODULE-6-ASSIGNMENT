## fashion_cnn.R
# Simple R script showing how to use keras in R to train Fashion MNIST
# This is a minimal template — run in an environment with keras and TensorFlow for R installed.

library(keras)

mnist <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

x_train <- array_reshape(x_train, c(nrow(x_train), 28,28,1))/255
x_test <- array_reshape(x_test, c(nrow(x_test), 28,28,1))/255
y_train_cat <- to_categorical(y_train, 10)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% summary()
history <- model %>% fit(x_train, y_train_cat, epochs = 3, batch_size = 128, validation_split = 0.1)

preds <- model %>% predict(x_test[1:2,,,])
labels <- apply(preds, 1, which.max) - 1
cat(paste(labels, collapse='\n'), file = 'predictions_output.txt')
cat('Saved predictions_output.txt\n')
