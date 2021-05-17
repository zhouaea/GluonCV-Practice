# https://mxnet.apache.org/versions/1.5.0/tutorials/index.html#python-tutorials
from mxnet import gluon
import numpy as np
from matplotlib import pyplot as plt

'''Load the fashion MNIST dataset from gluon'''
# Array of tuples: first element in the tuple is 28 x 28 x 1 image, second element is the image label.
dataset = gluon.data.vision.datasets.MNIST(train=True)
print(dataset[0][1])

'''Transform image to tensor format'''
# # Store ToTensor function from gluon library
# to_tensor = mxnet.gluon.data.vision.transforms.ToTensor()
# # Transform the image of an image-label pair but not the image label.
# dataset_with_one_tensor = dataset.transform_first(to_tensor)

'''Normalize an image'''
normalize = gluon.data.vision.transforms.Normalize()
dataset_normalized = dataset.transform_first(normalize)

'''Compose a transformation that randomly augments data'''
random_augmentation = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop,
    gluon.data.vision.transforms.RandomFlipLeftRight,
    gluon.data.vision.transforms.RandomBrightness
])

# Transform first only transforms the data but not the label of each element.
augmented_dataset = dataset.transform_first(random_augmentation)

plt.imshow(dataset[0][0].asnumpy())
# TODO After transformation, I cannot access the transformed dataset as a result of various errors.
# plt.imshow(dataset_normalized[0][0].asnumpy())
# plt.imshow(augmented_dataset[0][0].asnumpy())
plt.show()