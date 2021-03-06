import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST/data", one_hot=True)
#images=> image matrix with one_hot, labels=>0-9
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,mnist.test.labels #

trX = trX.reshape(-1, 28, 28, 1) # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1) # 28x28x1 test img
X = tf.placeholder("float", [None, 28, 28, 1]) #?, 28, 28, 1
Y = tf.placeholder("float", [None, 10]) #?, 10

#randomly intial weights function
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])

w_o = init_weights([625, 10])

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')) #28, 28, 1, 32
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #28, 28, 1, 16
    l1 = tf.nn.dropout(l1, p_keep_conv) #3, 3, 1, 16

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')) #28, 28, 1, 16
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #28, 28, 1, 8
    l2 = tf.nn.dropout(l2, p_keep_conv) #28, 28, 1, 8

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')) #28, 28, 1, 8
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #28, 28, 1, 4
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]]) # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv) #?, 2048

    l4 = tf.nn.relu(tf.matmul(l3, w4)) #l3=>?, 2048; w4=> 2048, 625; l4=>?, 625
    l4 = tf.nn.dropout(l4, p_keep_hidden) #?, 625
    output = tf.matmul(l4, w_o) #l4=>?, 625; w_o=>625, 10; output=>?, 10
    return output


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn. softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

batch_size = 128
test_size = 256
# Launch the graph in a session
with tf.Session() as sess:
# initialize all variables
    tf. global_variables_initializer().run()
    #training epoch
    for i in range(100):
        #make start and end indices of training data
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5})
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        #check the labels are the same as predict_op or not
        #np.argmax=> find the max value index, cause the labels data is like [0, 1, 0, .....0], which means the target is 1
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices],p_keep_conv: 1.0,p_keep_hidden: 1.0})))