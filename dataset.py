from mxnet import random, gluon, nd
import matplotlib.pyplot as plt

'''Create a dataset with mxnet ndarrays.'''
# # Create an mxnet ndarray with 10 3d data points.
# x = random.uniform(shape=(10, 3))
# # Create an mxnet ndarray with 10 labels.
# y = random.uniform(shape=(10, 1))
# # We can create a dataset with nd arrays.
# dataset = gluon.data.dataset.ArrayDataset(x,y)
# # Check the 5th data point-label pair.
# print(dataset[4])

'''Load the MNIST dataset'''
# Load the MNIST dataset from a library of gluon's preloaded datasets.
# Each sample is an image (in 3D NDArray) with shape (28, 28, 1)
train_dataset = gluon.data.vision.datasets.MNIST(train=True)
validation_dataset = gluon.data.vision.datasets.MNIST(train=False)

# Access a sample image input from the 6th data point and display it in pyplot.
sample_image = train_dataset[5][0]
print("Label: {}".format(train_dataset[5][1]))
plt.imshow(sample_image[:, :, 0].asnumpy())
plt.show()

'''Create a dataset from images stored in a folder on my local computer'''
# train_dataset = gluon.data.vision.datasets.ImageFolderDataset('pathtofolder')
