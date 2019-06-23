import os
import tensorflow as tf
import numpy as np
import  tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

def TeacherNet(input,keep_prob_conv,keep_prob_hidden,scope="teacher"):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],stride=[1,1], activation_fn=tf.nn.relu):

            net = slim.conv2d(input, 32, kernel_size=[5, 5], scope="conv1")
            net = slim.max_pool2d(net, [2, 2], 2, scope="pool1")
            net = slim.nn.dropout(net,keep_prob_conv)

            net = slim.conv2d(net, 64, kernel_size=[3, 3], scope="conv2")
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = tf.nn.dropout(net, keep_prob_conv)

            net = slim.conv2d(net, 128, kernel_size=[3, 3], scope="conv3")
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
            net = tf.nn.dropout(net, keep_prob_conv)

            net = slim.flatten(net)

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.0)):

            net = slim.fully_connected(net,625,scope="fc1")
            net = tf.nn.dropout(net, keep_prob_hidden)
            net = slim.fully_connected(net,10,activation_fn=None,scope="fc2")

            net = tf.nn.softmax(net/temperature)
            return net

def StudentNet(input,scope="student"):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.sigmoid):
            net = slim.fully_connected(input, 1000, scope="fc1")
            net = slim.fully_connected(net, 10, activation_fn=None, scope="fc2")
            return net

def loss(prediction,output):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(output * tf.log(prediction), reduction_indices=[1]))
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return cross_entropy,accuracy

temperature = 2
alpha = 0.5
temperature = 1
start_lr = 1e-4
decay = 1e-6

log_path = os.getcwd()
logs_path = os.path.join(log_path, 'mnist_logs')
if tf.gfile.Exists(logs_path):
    tf.gfile.DeleteRecursively(logs_path)

x = tf.placeholder("float", [None, 784], name='x')
y_ = tf.placeholder("float", [None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob_hidden = tf.placeholder("float")
keep_prob_conv = tf.placeholder("float")

y_conv_teacher = TeacherNet(x_image,keep_prob_conv,keep_prob_hidden)
y_conv = StudentNet(x)

y_conv_student = tf.nn.softmax(y_conv / temperature)
y_conv_student_actual = tf.nn.softmax(y_conv)
teacher_loss, teacher_accuracy = loss(y_conv_teacher, y_)
student_loss1, student_accuracy = loss(y_conv_student_actual, y_)
student_loss2, _ = loss(y_conv_student,y_conv_teacher)
student_loss = student_loss1 + student_loss2

tf.summary.scalar("loss", teacher_loss)
tf.summary.scalar("accuracy", teacher_accuracy)
merged_summary_op = tf.summary.merge_all()

model_vars = tf.trainable_variables()
var_teacher = [var for var in model_vars if 'teacher' in var.name]
var_student = [var for var in model_vars if 'student' in var.name]

l_rate = tf.placeholder(shape=[], dtype=tf.float32)

teacher_train_step = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(teacher_loss, var_list=var_teacher)
student_train_step = tf.train.GradientDescentOptimizer(0.1).minimize(student_loss, var_list=var_student)

with tf.Session() as sess:
    data_path = '/home/ysk/code/handwriting_recognition/MNIST_data'
    if os.path.exists(data_path):
        mnist = input_data.read_data_sets(data_path, one_hot=True)
    else:
        print("file not exist!")

    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    for i in tqdm(range(10000)):
        batch = mnist.train.next_batch(128)
        lr = start_lr * 1.0 / (1.0 + i * decay)
        teacher_train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob_conv: 0.8, keep_prob_hidden: 0.5, l_rate: lr})
    print("test accuraracy: %.5f" %teacher_accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels,keep_prob_conv:1.0,keep_prob_hidden:1.0}))

    for i in tqdm(range(10000)):
        batch = mnist.train.next_batch(50)
        student_train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob_conv: 0.8, keep_prob_hidden: 0.5, l_rate: lr})
    print("test accuraracy: %.5f" %student_accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels,keep_prob_conv:1.0,keep_prob_hidden:1.0}))






