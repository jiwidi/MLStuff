import tensorflow as tf
import time

x1=tf.constant(5)
x2=tf.constant(6)

result=tf.multiply(x1,x2)

with tf.Session() as sess:
    start=time.time()
    for u in range(1000000):
        output=sess.run(result)
    print(time.time()-start)
    print(output)
