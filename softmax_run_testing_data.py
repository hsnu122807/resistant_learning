import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_save_result = dir_path + r"\19_owl_rules\multiple_output_node_bp"
process_type = r"\benign"  # benign amount:2851   malware amount:4763
DATA_DIR = dir_path+r"\19_owl_rules\testing_data"+process_type+"_process_testing_data"

maochi_outlier_method_benign_count = 0
maochi_outlier_method_total_count = 0
two_class_method_benign_count = 0
two_class_method_total_count = 0
counter = 0
for filename in os.listdir(DATA_DIR):
    with tf.Graph().as_default():
        with tf.name_scope('placeholder_x'):
            x = tf.placeholder(tf.float64, name='x_placeholder')
        with tf.name_scope('placeholder_y'):
            y_ = tf.placeholder(tf.float64, name='y_placeholder')
        # W = tf.Variable(np.loadtxt(dir_save_result + r"\weight.txt"), dtype=tf.float64, name='weight')
        # b = tf.Variable(np.loadtxt(dir_save_result + r"\bias.txt"), dtype=tf.float64, name='bias')  # true benign rate: 0/2851
        # W = tf.Variable(np.loadtxt(dir_save_result + r"\24719_weight.txt"), dtype=tf.float64, name='weight')
        # b = tf.Variable(np.loadtxt(dir_save_result + r"\24719_bias.txt"), dtype=tf.float64, name='bias')  # true benign rate: 2851/2851, but this network almost output benign for all case
        W = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
        b = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')  # true benign rate: 0/2851
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        input_process = np.loadtxt(DATA_DIR+"/"+filename, dtype=str, delimiter=" ")
        input_process = input_process[:, :input_process.shape[1] - 1]
        input_process = np.ndarray.astype(input_process, float)

        y_benign = np.zeros(20)
        y_benign[0] = 1
        # print(y_element)
        y_benign_arr = np.tile(y_benign, input_process.shape[0]).reshape(-1, 20)

        classify_is_benign_count = sess.run(accuracy, feed_dict={x: input_process, y_: y_benign_arr})

        two_class_method_benign_count += classify_is_benign_count
        two_class_method_total_count += input_process.shape[0]
        print("benign rate: {0}/{1}".format(classify_is_benign_count, input_process.shape[0]))
        # if classify_is_benign_count == input_process.shape[0]:
        #     counter += 1
        #     print("{0} process is pure benign".format(counter))

        # W = tf.Variable(np.loadtxt(dir_save_result + r"\weight.txt"), dtype=tf.float64, name='w')
        # b = tf.Variable(np.loadtxt(dir_save_result + r"\bias.txt"), dtype=tf.float64, name='b')  # true benign rate: 0/2851
        # y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
        #
        # sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run()
        # classify_is_benign_count = sess.run(accuracy, feed_dict={x: input_process, y_: y_benign_arr})
        # maochi_outlier_method_benign_count += classify_is_benign_count
        # maochi_outlier_method_total_count += input_process.shape[0]
        # print("benign rate: {0}/{1}".format(classify_is_benign_count, input_process.shape[0]))


# print("{0} process is pure benign".format(counter))

# print("benign sample rate by two class method in 2851 benign: {0}/{1}".format(two_class_method_benign_count, two_class_method_total_count))
# print("benign sample rate by maochi method in 2851 benign: {0}/{1}".format(maochi_outlier_method_benign_count, maochi_outlier_method_total_count))
# benign sample rate by two class method in 2851 benign: 1986776/2615012
# benign sample rate by maochi method in 2851 benign: 1979318/2615012
