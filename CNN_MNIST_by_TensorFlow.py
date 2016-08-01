"""
ConvNet Architectures
INPUT -> [[CONV -> RELU] x N -> POOL?] x M -> [FC -> RELU] x K -> FC
N >= 0 (and usually N <= 3),
M >= 0, K >= 0 (and usually K < 3).
Common patterns:
INPUT -> FC, linear classifier.
INPUT -> CONV -> RELU -> FC
INPUT -> [CONV -> RELU -> POOL] x 2 -> FC -> RELU -> FC. We will build this
INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL] x 3 -> [FC -> RELU] x 2 -> FC
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    The strides middle 1,1 mean shift is one by one grid shieft by column and row.
    The pooling and convolutional ops slide a "window" across the input tensor. Using tf.nn.conv2d as an example: If the input tensor has 4 dimensions:  [batch, height, width, channels], then the convolution operates on a 2D window on the height, width dimensions.
    strides determines how much the window shifts by in each of the dimensions. The typical use sets the first (the batch) and last (the depth) stride to 1.

    Let's use a very concrete example: Running a 2-d convolution over a 32x32 greyscale input image. I say greyscale because then the input image has depth=1, which helps keep it simple. Let that image look like this:
    00 01 02 03 04 ...
    10 11 12 13 14 ...
    20 21 22 23 24 ...
    30 31 32 33 34 ...
    ...
    Let's run a 2x2 convolution window over a single example (batch size = 1). We'll give the convolution an output channel depth of 8.
    The input to the convolution has shape=[1, 32, 32, 1].
    If you specify strides=[1,1,1,1] with padding=SAME, then the output of the filter will be [1, 32, 32, 8].
    The filter will first create an output for:
    F(00 01
      10 11)
    And then for:
    F(01 02
      11 12)
    and so on. Then it will move to the second row, calculating:
    F(10, 11
      20, 21)
    then
    F(11, 12
      21, 22)
    If you specify a stride of [2, 2] it won't do overlapping windows. It will compute:
    F(00, 01
      10, 11)
    and then
    F(02, 03
      12, 13)
    The stride operates similarly for the pooling operators.
    Question 2: Why strides [1, x, y, 1] for convnets
    The first 1 is the batch: You don't usually want to skip over examples in your batch, or you shouldn't have included them in the first place. :)
    The last 1 is the depth of the convolution: You don't usually want to skip inputs, for the same reason.
    The conv2d operator is more general, so you could create convolutions that slide the window along other dimensions, but that's not a typical use in convnets. The typical use is to use them spatially.
    Why reshape to -1 -1 is a placeholder that says "adjust as necessary to match the size needed for the full tensor." It's a way of making the code be independent of the input batch size, so that you can change your pipeline and not have to adjust the batch size everywhere in the code.
    we see tf.reshape(_X,shape=[-1, 28, 28, 1]). Why -1?
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    ksize is kernel size, middle 2,2 means 2X2, and the middle 2,2 in strides means it goes over every second row and column
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

##############
# import MNIST data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784]) # each image is 28X28 pixel, hence 784 columns when vectorized
W = tf.Variable(tf.zeros([784, 10]))  # 10 means the output is a class of 10, 0..9 digits
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10])
"""
Here, the x and y_ are the actual data from each batch, here since we don't know the batch size, we
    use placeholder
"""

W_conv1 = weight_variable([5, 5, 1, 32]) # 5x5 patch, 1 is the depth, since raw black white color depth is only 1, 32 features

b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1]) # -1 means no need to specify batch size, it uses them all
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64]) # 32 is the depth of at second layer
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 7X7 is the patch size since two max pools, each max pool shrink size by half, fully connected now
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # matmul is the matrix mulplication

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10]) # now the layer is shrink to 1024 neuros, we need to the output to be of 10 classes
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # last stage is the softmax

# initialize variables and session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # the loss metric

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# Run mini-batch training on 50 elements 20000 times.
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        # train_accuracy = accuracy.eval(feed_dict={
        #     x:batch[0], y_: batch[1], keep_prob: 1.0})
        train_accuracy = sess.run(accuracy, feed_dict={
             x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


