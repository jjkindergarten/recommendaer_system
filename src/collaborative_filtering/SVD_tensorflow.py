import numpy as np
import pandas as pd
from numpy.random import randint, normal, choice
from src.utilits.share import generate_error_matrix
import tensorflow as tf
from src.data.load_data import load_movielens_100k
from src.data.load_data import load_popular_sub_data
from src.utilits.dataio import ShuffleIterator, OneEpochIterator

class inference_svd(object):

    def __init__(self, user_num, item_num, dim):
        self.bias_global = tf.Variable(initial_value = tf.zeros([1]), name = "bias_global")
        self.w_bias_user = tf.Variable(initial_value = tf.zeros([user_num]), name = "embd_bias_user")
        self.w_bias_item = tf.Variable(initial_value = tf.zeros([item_num]), name = "embd_bias_item")
        self.w_user = tf.Variable(initial_value = tf.random.truncated_normal(shape = (user_num, dim), stddev=0.02),
                                  name = "embd_user")
        self.w_item = tf.Variable(initial_value = tf.random.truncated_normal(shape = (item_num, dim), stddev=0.02),
                                  name = "embd_item")
    def infer_reg(self, users, items):
        user_tf = tf.constant(users, dtype=tf.int64)
        item_tf = tf.constant(items,  dtype=tf.int64)
        bias_user = tf.nn.embedding_lookup(self.w_bias_user, user_tf, name="bias_user")
        bias_item = tf.nn.embedding_lookup(self.w_bias_item, item_tf, name="bias_item")
        embd_user = tf.nn.embedding_lookup(self.w_user, user_tf, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(self.w_item, item_tf, name="embedding_item")

        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, self.bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        # regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
        # return infer, regularizer
        return infer


# gradient tape
def grad(infer, outputs):
    with tf.GradientTape() as t:
        outputs_tf = tf.constant(outputs, dtype=tf.float32)
        current_loss = tf.nn.l2_loss(tf.subtract(infer, outputs_tf))
    return t.gradient(current_loss, [model.bias_global, model.w_bias_user, model.w_bias_item, model.w_user, model.w_item])

# load data
data,  train, test = load_movielens_100k()

BATCH_SIZE = 50
DIM = 5
EPOCH_MAX = 100
LEARNING_RATE = 0.001

user_num = len(set(data['user']))
item_num = len(set(data['movie']))
num_batch = data.shape[0]/BATCH_SIZE

iter_train = ShuffleIterator([train["user"],
                              train["movie"],
                              train["rating"]],
                             batch_size=BATCH_SIZE)

iter_test = OneEpochIterator([test["user"],
                              test["movie"],
                              test["rating"]],
                             batch_size=-1)


model = inference_svd(user_num = user_num, item_num = item_num, dim=5)
train_op = tf.optimizers.Adam(learning_rate = LEARNING_RATE)


# run~ mainly apply_gradients
for i in range(EPOCH_MAX * num_batch):
    users, items, rates = iter_train.next()
    infer, regular = model.infer_reg(users, items)
    grads = grad(infer, rates)
    train_op.apply_gradients(zip(grads, [model.bias_global, model.w_bias_user, model.w_bias_item, model.w_user, model.w_item]))
    if i % 20 == 0:


        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, X, y)))
