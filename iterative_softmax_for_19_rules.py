import tensorflow as tf
import numpy as np
import time
import math
import os

dir_path = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\19_owl_rules"
# input(1)

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

    training_data = np.loadtxt(dir_path + r'\owl_rule_{0}_{1}_and_benign_{1}.txt'.format(rule_arr[rule], data_amount_arr[rule]), dtype=float, delimiter=' ')
    training_x = training_data[:, 1:]
    y_benign_element = np.zeros(2)
    y_mal_element = np.zeros(2)
    y_benign_element[0] = 1
    y_mal_element[1] = 1
    y_element = np.concatenate([y_mal_element, y_benign_element], axis=0)
    training_y = np.tile(y_element, int(data_amount_arr[rule])).reshape(-1, 2)
    majority_rate = 0.95
    bp_times = 0

    with tf.Graph().as_default():
        with tf.name_scope('placeholder_x'):
            x = tf.placeholder(tf.float64, name='x_placeholder')
        with tf.name_scope('placeholder_y'):
            y_ = tf.placeholder(tf.float64, name='y_placeholder')
        W = tf.Variable(tf.random_normal([m, 2], mean=0, stddev=0.1, dtype=tf.float64), dtype=tf.float64, name='weight')
        b = tf.Variable(tf.random_normal([2], mean=0, stddev=0.1, dtype=tf.float64), dtype=tf.float64, name='bias')
        # W = tf.Variable(tf.zeros([m, 2], dtype=tf.float64), dtype=tf.float64, name='weight')
        # b = tf.Variable(tf.zeros([2], dtype=tf.float64), dtype=tf.float64, name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            cross_entropy_for_sorting = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
        train_step = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # aaa = tf.argmax(y, 1)
        # bbb = tf.argmax(y_, 1)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        print('terminal stage: {0}'.format(int(majority_rate * N) + 1))
        for stage in range(m+2, int(majority_rate * N) + 1):
            print('stage {0}'.format(stage))
            if stage == (m+2):
                current_stage_training_x = training_x[:m+2]
                current_stage_training_y = training_y[:m+2]
                print(current_stage_training_x.shape)
                print(current_stage_training_y.shape)
            else:  # 用cross entropy sorting
                cross_entropy_of_all_data = sess.run(cross_entropy_for_sorting,
                                                     feed_dict={x: training_x, y_: training_y}).reshape(-1, 1)
                # print(cross_entropy_of_all_data)
                # TODO: 從小排到大 拿小的n筆

                concat_x_and_y = np.concatenate((training_x, training_y), axis=1)
                concat_entropy_and_x_y = np.concatenate((cross_entropy_of_all_data, concat_x_and_y), axis=1)
                # print(concat_entropy_and_x_y.shape)
                sort_result = concat_entropy_and_x_y[np.argsort(concat_entropy_and_x_y[:, 0])]
                x_training_data_sort_by_entropy = np.delete(sort_result, (0, m + 1, m + 2), axis=1)  # 去除0和m+1,m+2欄
                y_training_data_sort_by_entropy = np.delete(sort_result, slice(0, m + 1), axis=1)  # 去除從0到m欄
                current_stage_training_x = x_training_data_sort_by_entropy[:stage]
                current_stage_training_y = y_training_data_sort_by_entropy[:stage]
                # print(current_stage_training_x.shape)
                # print(current_stage_training_y.shape)

                # acc = sess.run(accuracy, feed_dict={x: training_x, y_: training_y})
                # print(acc)
                # input(1)
            # if stage == 1:
            #     current_stage_training_x = np.array([benign_samples[0]])
            #     y_element = np.zeros(2)
            #     y_element[0] = 1
            #     current_stage_training_y = np.array([y_element])
            # elif stage % 2 == 0:
            #     current_stage_training_x = np.append(current_stage_training_x, [malware_samples[(stage // 2) - 1]], axis=0)
            #     y_element = np.zeros(2)
            #     y_element[1] = 1
            #     current_stage_training_y = np.append(current_stage_training_y, [y_element], axis=0)
            # else:
            #     current_stage_training_x = np.append(current_stage_training_x, [benign_samples[(stage // 2)]], axis=0)
            #     y_element = np.zeros(2)
            #     y_element[0] = 1
            #     current_stage_training_y = np.append(current_stage_training_y, [y_element], axis=0)

            last_cross_entropy = 99999
            for i in range(max_bp_times_in_stage):
                # print(current_stage_training_x)
                # print(current_stage_training_y)
                correct_rate = sess.run(accuracy, feed_dict={x: current_stage_training_x, y_: current_stage_training_y})
                if correct_rate == 1:
                    if i == 0:
                        print('all data in this stage correctly classified.')
                    else:
                        print('all data in this stage correctly classified after {0} times bp.'.format(i))
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
        # input(1)

        # file = open(new_path + r"\_testing_analyze.txt", 'w')
        #
        # file.close()
