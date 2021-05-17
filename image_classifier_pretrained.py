# Import gluon cv toolkit, a library with pretrained computer vision models and preloaded image datasets.
import gluoncv as gcv
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt

'''Obtain and load image'''
# Download image from internet and store it as a jpg in the ML practice folder
gcv.utils.download(url='https://cdn.mos.cms.futurecdn.net/ntFmJUZ8tw3ULD3tkBaAtf.jpg', path='Test Images/sample_image.jpg')

# Transform jpg into ndarray with HWC format, each value is an unsigned 8 bit value.
image = mx.image.imread('Test Images/sample_image.jpg')
print("Image shape: " + str(image.shape))

'''Preprocess image for prediction'''
# Transform image in four ways to utilize resnet50 neural network:
# 1. Add extra dimension to denote batch size (which would be 1 in this case), for NCHW format (N is for batch size).
# 2. Convert each unsigned 8 bit value into a 32 bit float.
#   - Apparently, floats are quicker to perform calculations with.
# 3. Normalize the data, transforming each value so that the range of each value is centered on 0 and sd of 1.
#   - This makes it so the neural network doesn't have to calculate massive products like (255 * 255 * 255 * ...)
# 4. Crop height and width to 224 x 224 pixels.
transformed_image = gcv.data.transforms.presets.imagenet.transform_eval(image)
print("Transformed Image Shape: " + str(transformed_image.shape))

# Print transformed image from mxnet nd array.
# Convert mxnet nd array into numpy array and flip parameter locations
# from (batch size, channel, height, width) to (height, width, channel).
transformed_image_for_pyplot = np.transpose(transformed_image.asnumpy()[0], (1, 2, 0))
# Generate image using pyplot.
plt.imshow(transformed_image_for_pyplot)
# Show all generated pyplot images.
plt.show()

'''Run image through pretrained model'''
# Load resnet50 machine learning model, pretrained from the imagenet1k dataset,
# which contains 1000 classes of images, with 1 million HD images total.
model = gcv.model_zoo.get_model('ResNet50_v1d', pretrained=True)

# Run forward pass on photo with model, returning a 1 x 1000 nd array ([batch size][number of classes]).
# Each cell in the array contains a raw value (-infinity to infinity)
# that is proportional to the chance of the inputted image belonging to the corresponding class of the index.
# Indexes 0-999 each correspond to a specific class of images (dog, cat, etc.)
predictions = model(transformed_image)
# print("-------------")
# print("Model output:")
# print(predictions)
# print("-------------")

'''Process model output to be human readable'''
# Convert logits (raw values from -infinity to infinity) of each class
# to probabilities (values from 0 to 1 that sum to 1) using softmax.
# Also outputs a 1 x 1000 ndarray.
probabilities = mx.nd.softmax(predictions)[0].asnumpy()
# print("-------------")
# print("Softmax of model")
# print(probabilities)
# print("-------------")

# Find the softmax value with the highest probability of correctly classifying the image,
# and use its index to determine its corresponding class label.
print("Most Probable Image Class is...")
index = np.argmax(probabilities)
print(model.classes[index] + " with a probability of " + str(probabilities[index]))
