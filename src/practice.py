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


#
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []



with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)

    # Add asserts to check the shape of the output.
    tf.debugging.assert_equal(logits.shape, (32, 10))

    loss_value = loss_object(labels, logits)

loss_history.append(loss_value.numpy().mean())
grads = tape.gradient(loss_value, mnist_model.trainable_variables)
optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

