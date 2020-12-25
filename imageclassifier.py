# Import gluon cv toolkit, a library with pretrained computer vision models and preloaded image datasets.
import gluoncv as gcv
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt

# Download image from internet and store it as a jpg in the ML practice folder
gcv.utils.download(url='https://cdn.mos.cms.futurecdn.net/ntFmJUZ8tw3ULD3tkBaAtf.jpg', path='sample_image.jpg')

# Transform jpg into ndarray with HWC format, each value is an unsigned 8 bit value.
image = mx.image.imread('sample_image.jpg')
print("Image shape: " + str(image.shape))



# Transform image in three ways to utilize neural network:
# 1. Add extra dimension to denote batch size (which would be 1 in this case), for NCHW format (N stands for batch size).
# 2. Convert each unsigned 8 bit value into a 32 bit float.
# 3. Normalize the data, transforming each value so that the range of each value is centered on 0 and sd of 1.
# 4. Crop height and width to 224 x 224 pixels.
transformed_image = gcv.data.transforms.presets.imagenet.transform_eval(image)
print("Transformed Image Shape: " + str(transformed_image.shape))

# Print transformed image from mxnet nd array.
# Convert mxnet nd array into numpy array and flip parameter locations from (batch size, channel, height, width) to (width, height, channel).
transformed_image_for_pyplot = np.transpose(transformed_image.asnumpy()[0], (1,2,0))
# Generate image using pyplot.
plt.imshow(transformed_image_for_pyplot)
# Show all genereated pyplot images.
plt.show()


# Load pretrained machine learning model, this will also contain all 1000 classes from imagenet1k.
model = gcv.model_zoo.get_model('ResNet50_v1d',pretrained=True)

# Run forward pass on photo with model, returning (batch size, prediction for each class).
predictions = model(transformed_image)
print("Model output")

# Convert logits (raw values from -infinity to infinity) to probabilities (values from 0 to 1 that sum to 1) using softmax.
probabilities = mx.nd.softmax(predictions)[0].asnumpy()
print("Softmax of model")
# print(probabilities)

# Print the highest probability classification.
print("Image Classification")
index = np.argmax(probabilities)
print(model.classes[index] + ": " + str(probabilities[index]))
