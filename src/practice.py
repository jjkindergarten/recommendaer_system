from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from scipy.linalg import lstsq

# load data
housing = fetch_california_housing()
m,n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

# transfer the data into tf.format
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
X = scaled_housing_data_plus_bias
y = housing.target.reshape(-1, 1)

# the part includes coef
class Model(object):
    def __init__(self, x):
        self.W = tf.Variable(tf.random.normal((x.shape[1], 1)))

    def model_(self, x):
        return x @ self.W

# the loss function based on model
def loss(model,inputs, targets):
    error = model.model_(inputs) - targets
    return tf.reduce_mean(tf.square(error))

# gradient tape
def grad(model, inputs, outputs):
    with tf.GradientTape() as t:
        current_loss = loss(model, inputs, outputs)
    return t.gradient(current_loss, [model.W])

# add the optimizer
model = Model(X)
optimizer = tf.optimizers.SGD(learning_rate)

# run~ mainly apply_gradients
for i in range(n_epochs):
  grads = grad(model, X, y)
  optimizer.apply_gradients(zip(grads, [model.W]))
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, X, y)))

# compare the SGD result with linear square solution
p, res, rnk, s = lstsq(scaled_housing_data_plus_bias, housing.target.reshape(-1, 1))