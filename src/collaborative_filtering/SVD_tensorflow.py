import tensorflow as tf
from src.data.load_data import load_movielens_100k
from src.utilits.dataio import ShuffleIterator, OneEpochIterator
import time
from math import sqrt, inf

def loss_func_svd(rates, bias_global, embd_user, embd_item, bias_user, bias_item, penalty_rate):
    regularizer = tf.reduce_sum(embd_user ** 2, 1) + tf.reduce_sum(embd_item ** 2, 1) + bias_user ** 2 + bias_item ** 2
    penalty = tf.constant(penalty_rate, dtype=tf.float32, shape=[], name="l2")
    cost = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
    cost = sum((rates - (bias_global+ bias_user + bias_item - cost)) ** 2)
    l2_loss = cost + penalty * regularizer
    return l2_loss, cost




# load data
data, train, test = load_movielens_100k(split_ratio = 0.9)

def tf_svd():

    BATCH_SIZE = 200
    DIM = 5
    EPOCH_MAX = 100
    LEARNING_RATE = 0.001
    PENALTY_RATE = 0.05
    PATIENCE = 10

    user_num = len(set(data['user']))
    item_num = len(set(data['movie']))
    num_batch = data.shape[0]/BATCH_SIZE
    mini_vaild_cost = inf
    patience = 1

    iter_train = ShuffleIterator([train["user"],
                                  train["movie"],
                                  train["rating"]],
                                 batch_size=BATCH_SIZE)

    iter_test = OneEpochIterator([test["user"],
                                  test["movie"],
                                  test["rating"]],
                                 batch_size=-1)

    bias_global = tf.Variable(initial_value = tf.zeros([1]), name = "bias_global")
    w_bias_user = tf.Variable(initial_value = tf.zeros([user_num,]), name = "embd_bias_user")
    w_bias_item = tf.Variable(initial_value = tf.zeros([item_num,]), name = "embd_bias_item")
    w_user = tf.Variable(initial_value=tf.random.truncated_normal(shape=(user_num, DIM), stddev=0.02), name="embd_user")
    w_item = tf.Variable(initial_value=tf.random.truncated_normal(shape=(item_num, DIM), stddev=0.02), name="embd_item")

    test_user, test_item, test_rates = next(iter_test)
    train_op = tf.optimizers.Adam(learning_rate = LEARNING_RATE)
    test_user_tf = tf.constant(test_user, dtype=tf.int64, name='test_user_id')
    test_item_tf = tf.constant(test_item, dtype=tf.int64, name='test_item_id')

    print("{} {} {} {}".format("epoch", "train_error",  "vaild_error", "elapsed_time"))
    start = time.time()

    for i in range(int(EPOCH_MAX * num_batch)):
        users, items, rates = iter_train.next()
        user_tf = tf.constant(users, dtype=tf.int64, name='user_id')
        item_tf = tf.constant(items, dtype=tf.int64, name='item_id')

        with tf.GradientTape() as t:
            embd_user = tf.nn.embedding_lookup(w_user, user_tf, name="embedding_user")
            embd_item = tf.nn.embedding_lookup(w_item, item_tf, name="embedding_item")
            bias_user = tf.nn.embedding_lookup(w_bias_user, user_tf, name="bias_user")
            bias_item = tf.nn.embedding_lookup(w_bias_item, item_tf, name="bias_item")

            l2_loss, loss = loss_func_svd(rates, bias_global, embd_user, embd_item, bias_user, bias_item, PENALTY_RATE)

        grads = t.gradient(l2_loss, [bias_global, w_bias_user, w_bias_item, w_user, w_item])
        train_op.apply_gradients(zip(grads, [bias_global, w_bias_user, w_bias_item, w_user, w_item]))


        if i % BATCH_SIZE == 0:

            test_embd_user = tf.nn.embedding_lookup(w_user, test_user_tf, name="test_embedding_user")
            test_embd_item = tf.nn.embedding_lookup(w_item, test_item_tf, name="test_embedding_item")
            test_bias_user = tf.nn.embedding_lookup(w_bias_user, test_user_tf, name="test_bias_user")
            test_bias_item = tf.nn.embedding_lookup(w_bias_item, test_item_tf, name="test_bias_item")
            test_cost = tf.reduce_sum(tf.multiply(test_embd_user, test_embd_item), 1)
            test_cost = sum((test_rates - (bias_global + test_bias_user + test_bias_item - test_cost)) ** 2)
            end = time.time()
            print("{:3d} {:f} {:f} {:f}(s)".format(i // BATCH_SIZE, sqrt(loss/bias_user.shape[0]),
                                                   sqrt(test_cost/test_user_tf.shape[0]), end - start))

            # early stopping
            if test_cost < mini_vaild_cost:
                mini_vaild_cost = test_cost
                patience = 1
            else:
                patience += 1
                if patience >= PATIENCE:
                    print('no more improvment')
                    break
    return bias_global, w_bias_user, w_bias_item, w_user, w_item
