from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('data_dir', './MNIST/data/', 'Directory for storing data')
mnist=input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

x=tf.placeholder(tf.float32, [None, 784])
W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))
y=tf.matmul(x, W) + b

y_=tf.placeholder(tf.float32, [None, 10])
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

#用summary 儲存結果給 tensorboard 用
#cmd =>>   tensorboard --logdir=PATH 執行完，網址localhost:6006開啟
#tensorboard --logdir=./MNIST/graph/tfboard_test
train_writer=tf.summary.FileWriter('./MNIST/graph/tfboard_test', sess.graph)
train_writer.close()

def variable_summaries(var):
    """對一個張量添加多個摘要描述"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) # 平均
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev) # 標準差
        tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
        tf.summary.histogram('histogram', var)