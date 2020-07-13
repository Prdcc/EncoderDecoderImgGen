# EncoderDecoderImgGen
Image generation using an encoder decoder model.

## The model
During training, two deep CNNs are appended one after the other: the first takes in a 28 by 28 black and white image and returns a vector, by default of length 10. The second one takes the output from the second one and returns a new 28 by 28 image. The loss is given by the mean square error of the pixel values between the initial and final image. The idea is that the first network should learn to synthesise an image down to 10 numbers and the second one should learn how to recreate said image starting from 10 numbers. The second network should then become really good at recreating an image given only 10 numbers which can easily be controlled to generate new images. Training was done with the MNIST-Fashion dataset and a simple visualisation tool is given in visualisation.ipynb.

## Usage
To retrain the network, possibly with different images, simply call the train function in the train module. The data should be given as a numpy array of shape (n_samples,1,28,28).
