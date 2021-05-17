from mxnet import gluon
from matplotlib import pyplot as plt

'''Create a dataset with mxnet ndarrays.'''
dataset = gluon.data.vision.MNIST(train=True)

'''Create a data loader. This can be iterated with through a for loop.'''
# If the dataset is indivisible by the batch size, there are three options for the last batch that will be smaller than the others:
# 1. Keep - A batch with less samples than previous batches is returned.
# 2. Discard - The last batch is discarded if its incomplete.
# 3. Rollover - The remaining samples are rolled over to the next epoch.
data_loader = gluon.data.DataLoader(dataset, batch_size=5, last_batch='keep', shuffle=True)

counter = 1

print(data_loader)

# Each batch in the data loader has a tuple of a batch of data and a batch of labels.
# Data is a (N, H, W, C) ndarray. Labels is a (N) ndarray
for data, labels in data_loader:
    # Print one batch.
    if counter > 1:
        break
    # Iterate through all 5 sample images in the batch.
    for i in range(5):
        print(labels[i])
        plt.imshow(data[i].asnumpy())
        plt.show()
    counter += 1

