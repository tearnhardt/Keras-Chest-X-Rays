#Training a model to identify healthy and unhealthy lungs along with determining the type of pneumonia. 
## Goals: 
   #Use Keras to create a mostly accurate model 

setwd("~/Documents/Kaggle")

#Libraries
library(keras)
install_keras()

lung_list <- c("NORMAL","PNEUMONIA")

output_n <- length(lung_list)

img_width <- 20
img_height <- 20 
target_size <- c(img_width,img_height)

channels <- 3

train_image_files_path <- "~/Documents/Kaggle/chest_xray/train/"
valid_image_files_path <- "~/Documents/Kaggle/chest_xray/val/"

train_data_gen = image_data_generator(
  rescale = 1/225
)
valid_data_gen <- image_data_generator(
  rescale= 1/255
)


train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = lung_list,
                                                    seed = 42)
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = lung_list,
                                                    seed = 42)

cat("Number of images per class:")

table(factor(train_image_array_gen$classes))
train_image_array_gen$class_indices

lung_classes_indices <- train_image_array_gen$class_indices



train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 100
epochs <- 10

model <- keras_model_sequential()


model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)


hist <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("~/Documents/GitHub/Keras-Chest-X-Rays/Keras/lungs_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "~/Documents/GitHub/Keras-Chest-X-Rays/Keras")
  )
)
plot(hist)
tensorboard("~/Documents/GitHub/Keras-Chest-X-Rays/Keras")
