import tensorflow as tf
import numpy as np
import time
import math
import os

dir_path = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\19_owl_rules"
input(1)

# 可能會手動調整的參數
learning_rate_eta = 0.000005
max_bp_times_in_stage = 100000
target_accuracy = 0.95

rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
data_amount_arr = np.array([65, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 39, 100, 40, 100, 100, 100, 100], dtype=str)


for rule in range(rule_arr.shape[0]):
    print('rule: '+rule_arr[rule])
    analyze_result_save_dir_name = r"\owl_rule_{0}_{1}_and_benign_{1}_softmax".format(rule_arr[rule], data_amount_arr[rule])
    # create folder to save training process
    new_path = r"{0}/".format(dir_path) + analyze_result_save_dir_name
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    start_time = time.time()

    benign_samples = np.loadtxt(dir_path+r'\owl_rule_{0}_{1}_and_benign_{1}_benign_part.txt'.format(rule_arr[rule], data_amount_arr[rule]), dtype=float, delimiter=' ')
    malware_samples = np.loadtxt(dir_path + r'\owl_rule_{0}_{1}_and_benign_{1}_rule_part.txt'.format(rule_arr[rule], data_amount_arr[rule]), dtype=float, delimiter=' ')
    np.random.shuffle(benign_samples)
    np.random.shuffle(malware_samples)

    m = malware_samples.shape[1]
    N = malware_samples.shape[0] + benign_samples.shape[0]
    majority_rate = 0.95
    bp_times = 0

    with tf.Graph().as_default():
        with tf.name_scope('placeholder_x'):
            x = tf.placeholder(tf.float64, name='x_placeholder')
        with tf.name_scope('placeholder_y'):
            y_ = tf.placeholder(tf.float64, name='y_placeholder')
        W = tf.Variable(tf.zeros([m, 2], dtype=tf.float64), dtype=tf.float64, name='weight')
        b = tf.Variable(tf.zeros([2], dtype=tf.float64), dtype=tf.float64, name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        print('terminal stage: {0}'.format(int(majority_rate * N) + 1))
        for stage in range(1, int(majority_rate * N) + 1):
            print('stage {0}'.format(stage))
            if stage == 1:
                current_stage_training_x = np.array([benign_samples[0]])
                y_element = np.zeros(2)
                y_element[0] = 1
                current_stage_training_y = np.array([y_element])
            elif stage % 2 == 0:
                current_stage_training_x = np.append(current_stage_training_x, [malware_samples[(stage // 2) - 1]], axis=0)
                y_element = np.zeros(2)
                y_element[1] = 1
                current_stage_training_y = np.append(current_stage_training_y, [y_element], axis=0)
            else:
                current_stage_training_x = np.append(current_stage_training_x, [benign_samples[(stage // 2)]], axis=0)
                y_element = np.zeros(2)
                y_element[0] = 1
                current_stage_training_y = np.append(current_stage_training_y, [y_element], axis=0)

            last_cross_entropy = 99999
            for i in range(max_bp_times_in_stage):
                # print(current_stage_training_x)
                # print(current_stage_training_y)
                correct_rate = sess.run(accuracy, feed_dict={x: current_stage_training_x, y_: current_stage_training_y})
                if correct_rate == 1:
                    print('all data in this stage correctly classified.')
                    break

                sess.run(train_step, feed_dict={x: current_stage_training_x, y_: current_stage_training_y})
                bp_times += 1

                if i % 1000 == 0:
                    loss = sess.run(cross_entropy, feed_dict={x: current_stage_training_x, y_: current_stage_training_y})
                    if (last_cross_entropy - loss) < 0.001:
                        print('learning too slow, break.')
                        break
                    else:
                        last_cross_entropy = loss
        end_time = time.time()
        print('train end, save networks')
        current_W, current_b = sess.run([W, b], feed_dict={x: current_stage_training_x, y_: current_stage_training_y})
        np.savetxt(new_path + r"\two_class_weight.txt", current_W)
        np.savetxt(new_path + r"\two_class_bias.txt", current_b)

        file = open(new_path + r"\_two_class_training_detail.txt", 'w')
        file.writelines('learning rate: {0}\n'.format(learning_rate_eta))
        file.writelines('input node amount: {0}\n'.format(m))
        file.writelines('training data amount: {0}\n'.format(N))
        file.writelines('outlier rate: {0}\n'.format(1-majority_rate))
        file.writelines('thinking times count: {0}\n'.format(bp_times))
        file.writelines('execute time: {0} sec\n'.format(end_time-start_time))
        file.close()

        file = open(new_path + r"\_training_analyze.txt", 'w')
        y_element = np.zeros(2)
        y_element[0] = 1
        benign_y = np.tile(y_element, benign_samples.shape[0]).reshape(-1, 2)
        benign_correct_rate = sess.run(accuracy, feed_dict={x: benign_samples, y_: benign_y})
        file.writelines('benign accuracy: {0}/{1} , {2}\n'.format(int(round(benign_samples.shape[0]*benign_correct_rate)), benign_samples.shape[0], benign_correct_rate))
        y_element = np.zeros(2)
        y_element[1] = 1
        malware_y = np.tile(y_element, malware_samples.shape[0]).reshape(-1, 2)
        malware_correct_rate = sess.run(accuracy, feed_dict={x: malware_samples, y_: malware_y})
        file.writelines('malware accuracy: {0}/{1} , {2}\n'.format(int(round(malware_samples.shape[0] * malware_correct_rate)), malware_samples.shape[0], malware_correct_rate))
        file.close()

        # file = open(new_path + r"\_testing_analyze.txt", 'w')
        #
        # file.close()
