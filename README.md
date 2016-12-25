# An example of a fully convolutional neural network for image classification
This is an example of a fully convolutional neural network model for image classification. The model 
accepts input of variable size and yields a `nb_classes x 1` vector 
of class probabilities. This example is implemented in [Keras](https://keras.io/).

# Main ideas

* `1 x 1` convolutions for dimensionality reduction
* Global pooling to produce one value per category at the final layer

# Performance issue
Training the model on images of variable size takes unreasonably long time compared 
to training on inputs of the fixed size. See this [discussion on StackOverflow](http://stackoverflow.com/questions/41322286/training-a-fully-convolutional-neural-network-in-keras-takes-unreasonably-long-t)
for more details.
