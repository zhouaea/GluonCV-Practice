# Import gluon cv toolkit, a library with pretrained computer vision models and preloaded image datasets.
import gluoncv as gcv
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt

'''Obtain and load image'''
# Download image from internet and store it as a jpg in the ML practice folder
gcv.utils.download(url='https://www.hersheypa.com/assets/images/about-hershey/page-hero/hero-conditions-of-use.jpg', path='hershey.jpg')

# Transform jpg into mxnet ndarray with HWC format, each value is an unsigned 8 bit value.
image = mx.image.imread('hershey.jpg')
print("Image shape: " + str(image.shape))

'''Preprocess image for prediction'''
# Transform image in four ways to utilize yolo neural network:
# Return a second resized image in HWC format as a numpy ndarray for later visualization.
# 1. Add extra dimension to denote batch size (which would be 1 in this case), for NCHW format (N is for batch size).
# 2. Convert each unsigned 8 bit value into a 32 bit float.
# 3. Normalize the data, transforming each value so that the range of each value is centered on 0 and sd of 1.
# 4. Crop height and width to a picture that maintains aspect ratio with its shortest side at 512 pixels.
transformed_image, hwc_image = gcv.data.transforms.presets.yolo.transform_test(image, short=512)

'''Display image with pyplot for pedagogical purposes'''
# Print transformed image from mxnet nd array.
# Convert mxnet nd array into numpy array and flip parameter locations
# from (batch size, channel, height, width) to (width, height, channel).
transformed_image_for_pyplot = np.transpose(transformed_image.asnumpy()[0], (1, 2, 0))
# Generate image using pyplot.
# plt.imshow(transformed_image_for_pyplot)
# Show all generated pyplot images.
# plt.show()

'''Run image through pretrained model'''
# Load YOLOv3 machine learning model, pretrained from the COCO (common objects in context) dataset,
# which contains 1000 classes of images, with 1 million HD images total.
model = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

# Run forward pass on photo with model, returning a tuple with 3 tensors.
# First tensor contains object class indices, which tell us which objects are present in the image.
#   *Has a size of (1, 100, 1): 1 image in the batch, 100 potential objects in the photo, 1 index per object.
# Second tensor contains object class probabilities with a size of (1, 100, 1).
#   *Has a size of (1, 100, 1): 1 image in the batch, 100 potential objects in the photo, 1 probability per object.
# Third tensor contains object bounding box coordinates.
#   *Has a size of (1, 100, 4): 1 image in the batch, 100 potential objects, 4 coordinates for a bounding box per object
prediction = model(transformed_image)

'''Unpack prediction'''
# Split prediction into 3 variables.
object_indices, object_probabilities, bounding_boxes = prediction

'''Visualize prediction'''
# Print out descriptors for the top k predictions. The model automatically sorts from greatest to least probability.
k = 10
print(object_indices[0][:k])
print(object_probabilities[0][:k])
print(bounding_boxes[0][:k])

# Print the top 10 objects found and their probabilities
for i in range(10):
    # To extract scalar value from mxnet ndarray, we have to first convert it to a numpy ndarray and then
    # use the item() to get a float scalar. We then convert the float to an int for array indexing the array of labels.
    if int(object_indices[0][:k][i].asnumpy().item(0)) == -1:
        continue
    print(model.classes[int(object_indices[0][:k][i].asnumpy().item(0))] + ": " + str(object_probabilities[0][:k][i].asnumpy().item(0)))

# TODO Graphics not displaying in PyCharm.
gcv.utils.viz.cv_plot_bbox(hwc_image, bounding_boxes[0], scores=object_probabilities[0],
labels=object_indices[0], class_names=model.classes)
plt.show()
