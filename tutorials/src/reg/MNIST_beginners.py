import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    #Data, images 28x28 of handwritten numbers, onehot means that we got a array of 0s and just one 1, to specify the class
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #Model
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,784])

    #Weights
    W = tf.Variable(tf.zeros([784, 10]))
    #Biases
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #Real distribution
    y_ = tf.placeholder(tf.float32, [None, 10])
    #Cost
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    #Train cost
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #Initialize model
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))
main()

