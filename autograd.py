# Automatic Differentiation
# A program records all functions and represents how they connect with one another via a computational graph.
# Repeatedly applies chain rule to computational graph to calculate derivative of final function output.

# As an example, we will differentiate f((g(x)) using automatic differentiation.
from mxnet import nd
from mxnet import autograd

# Define values of x to be inputted into f(x).
x = nd.array([1, 2, 3, 4])

# 1. Let autograd know we are differentiating in respect to x.
# 2. Allocate space for gradients to be computed.
x.attach_grad()

# Define f(x) and g(x)
# f'(x) should equal 4x.
def f(x):
    return 2 * x**2 + 6

# g'(x) should equal 10x
def g(x):
    return 5 * x**2 + 3

# Record the function f(g(x)) in autograd as variable y, where autograd will create a computational map to represent the function.
# The correct dy/dx = f'(g(x)) * g'(x) = 4(5x^2 + 3) * 10x = (20x^2 + 12) * 10x = 200x^3 + 120x.
with autograd.record():
    y = f(g(x))

# Calculate gradients of y with automatic differentiation.
y.backward()

# The gradients of x are stored in the field .grad of the input ndarray.
print(x.grad)
