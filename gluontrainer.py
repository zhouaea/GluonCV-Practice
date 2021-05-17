# We'll need to manipulate mxnet nd arrays, calculate gradients, and use those gradients to optimize a neural network.
from mxnet import nd, autograd, optimizer, gluon

'''Define dataset where y = 2x + 1.'''
# X has a batch size of 3 with one feature in each batch.
x = nd.array([1, 2, 3])
y = nd.array([3, 5, 7])

'''Initialize a single neuron, capable of <batch_size> weights and one bias.'''
net = gluon.nn.Dense(1)
# Defer initialization of inputs and outputs until the network knows the shape of the data.
net.initialize()

'''Initialize trainer that will update weights and biases of our neural network'''
'''How optimizer works: (current_weight - net.weight.grad() * learning_rate / batch_size).'''
# Store the address space of every weight and bias of the neural network.
parameter_dictionary = net.collect_params()

# METHOD 1: Put optimizer and optimizer params as arguments in trainer. We will use stochastic gradient descent.
# trainer = gluon.Trainer(parameter_dictionary, optimizer='sgd', optimizer_params={'learning_rate': 0.01})

# METHOD 2: Define an optimizer instance and pass it into the trainer along with a reference to the nn parameters.
adam_instance = optimizer.Adam(learning_rate=0.01)
trainer = gluon.Trainer(parameter_dictionary, adam_instance)

# NOTE: To change learning rate
# print(trainer.learning_rate)
# trainer.set_learning_rate(0.1)

'''Perform forward pass, calculating and recording loss function with x passed in the into autograd.'''
# TODO: Where is gradient data stored so that the gluon trainer can access it and update parameters?
# Copy built in loss function (sqrt(summation(predicted - actual)^2)) into a variable.
loss_function = gluon.loss.L2Loss()
for i in range(1000):
    # Record loss function, with x as a set of inputs and y as a set of outputs.
    with autograd.record():
        l = loss_function(net(x), y)

    # Calculate gradients of loss function.
    l.backward()

    # print("Weight and bias before training: ")
    # print(net.weight.data())
    # print(net.bias.data())

    # Provide batch size as argument to trainer and let it update weights and biases.
    trainer.step(3)

    # print("\nWeight and bias after training: ")
    # print(net.weight.data())
    # print(net.bias.data())

'''Test trained neural network on other data'''
# x has a batch size of 3, each with one feature.
x = nd.array([5, 13, 15])
print("Neural network output given 5, 13, and 15")
print(net(x))
