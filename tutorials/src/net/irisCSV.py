from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

#Load data libraries
training_set = tf.contrib.learn.datasets.base.load_csv(filename='iris_training.csv', target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename='iris_test.csv',target_dtype=np.int)

#Shaping feature columns for classifier
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

#Let's build a DNN with the classifier class
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[64,32],
                                          model_dir="/tmp/iris_dnn",
                                          n_classes=3,
                                          dropout=None,)

#Train the model
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

#Let's use the test set to test accuracy
accuracy_score=classifier.evaluate(x=test_set.data,y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))

