from mxnet import gluon, init, autograd, metric
from mxnet.gluon.data.vision import datasets, transforms
from matplotlib import pyplot as plt
from gluoncv import model_zoo
from time import time
# Make sure nothing fishy happens when loading a model.
import warnings

'''Load dataset'''
# The dataset is an array of tuples.
fashion_mnist_train = datasets.FashionMNIST(train=True)
X, y = fashion_mnist_train[0]
print('X shape: ', X.shape, 'X data type', X.dtype, 'y:', y)
plt.imshow(fashion_mnist_train[0][0].asnumpy())
# plt.show()

'''Transform dataset to fit into gluon model'''
transformer = transforms.Compose([
    # 1. Convolutional neural networks iterate by channel, so we want input data to be in CHW format instead of HWC
    # format for better spacial locality.
    # 2.ToTensor also converts uint8 into float32, changing all numbers 0-255 to a decimal between 0 and 1. This
    # is important for mathematical reasons.
    transforms.ToTensor(),
    # Normalizing allows us to converge to a better model faster for mathematical reasons.
    transforms.Normalize(0.13, 0.31)])
# Only transform data and not labels with transform function.
fashion_mnist_train_transformed = fashion_mnist_train.transform_first(transformer)

'''Create dataloader'''
# The data loader is a an array of a tuple of arrays (one tuple has batch of data, one has batch of labels).
batch_size = 256
train_data_loader = gluon.data.DataLoader(
    fashion_mnist_train_transformed, batch_size=batch_size, shuffle=True)

'''Load validation dataset and create validation dataloader'''
mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size)

'''Create neural network'''
model = model_zoo.get_model('ResNet50_v1d', pretrained=False)

'''Initialize a new model'''
# Once data is passed through the model, initialize weights and biases with Xavier initialization.
model.initialize(init=init.Xavier())
# Pass sample data through model so it knows the shape of its inputs and initializes weights and biases.
# We have to do this now so the gluon trainer can get the weights and biases of the model before training starts.
for data_batch, label_batch in train_data_loader:
    model(data_batch)
    break

'''Load a preexisting model'''
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     deserialized_net = gluon.nn.SymbolBlock.imports("trained_fashion_mnist_model.json", ['data'], "trained_fashion_mnist_model-0001.params")

'''Define loss function'''
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

'''Create optimizer'''
trainer = gluon.Trainer(model.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': 0.01})

'''Create a trainer accuracy function'''
train_acc = metric.Accuracy()

'''Perform training'''
batch = 0

# Iterate through batches of data loader (one epoch).
for data_batch, label_batch in train_data_loader:
    tic = time()
    with autograd.record():
        # Calculate model output.
        output = model(data_batch)
        # Calculate loss of model output.
        loss = softmax_cross_entropy(output, label_batch)

    # Calculate gradient of loss function.
    loss.backward()

    # Update model using gradients.
    trainer.step(batch_size)

    # Store training loss.
    # train_loss = loss.mean().asscalar()

    # Determine if the model correctly predicted the output. Add 1 to a total everytime a prediction was correct and divide by number of predictions.
    # train_acc.update(label_batch, output)

    # Note: There are 60,000 images in the training dataset.
    # With 256 images per batch, it would take 235 batches to go through the whole dataset.
    # 235 * 0.035 seconds = 9 seconds to train for one epoch.
    print("Batch[%d] Perf: %.3f seconds for sample completion" %
        (batch, (time() - tic)))
    batch += 1

    # More advanced performance logging that takes longer (goes from > 0.1 seconds per batch to 4 seconds per batch).
    # print("Batch[%d] Loss: %.3f Acc: %s Perf: %.1f seconds for sample completion" %
    #     (batch, train_loss, train_acc.get()[1], (time() - tic)))

'''Save model parameters'''
# Because our preloaded model is a hybrid network, we cannot simply use save_parameters and load_parameters
# and save our parameters into a .params file. We have to store the model architecture in a json file and the parameters
# in a params file.
print("Saving model parameters...")
tic = time()
model.export("Trained Models/trained_fashion_mnist_model", epoch=1)
print("Model parameters saved!")
print("It took %f seconds to save parameters" % (time() - tic))
