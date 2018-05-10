# coding:utf-8
import tensorflow as tf
import numpy as np
import time
import math
import os

dir_path = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\mix_19_rules_binary_classification\softmax"

# 可能會手動調整的參數
learning_rate_eta = 0.000005
max_bp_times = 100000000
target_accuracy = 0.95
sampling_rate = 1.0
sampling_amount = 100
sampling_by_rate = True  # if False: sampling by fix amount
analyze_result_save_dir_name = "all_rules_data_sample_all_softmax_separate_benign_and_malicious"
# analyze_result_save_dir_name = "all_rules_data_sample_1_of_10_softmax_separate_benign_and_malicious"
# analyze_result_save_dir_name = "all_rules_data_sample_100_softmax_separate_benign_and_malicious"

# create folder to save training process
new_path = r"{0}/".format(dir_path) + analyze_result_save_dir_name
if not os.path.exists(new_path):
    os.makedirs(new_path)

# 所有不重複pattern中抽樣分兩類
training_data_container = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=object)
testing_data_container = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=object)
# 讀檔案並且shuffle
for i in range(training_data_container.shape[0]):
    if i == training_data_container.shape[0]-1:  # 讀benign file
        # 這邊先用純benign的兩個process 可考慮加入其他被ghsom視為benign
        data_dir_name = r"19_owl_rules\no_duplicate_pattern_19_rules\benign_chrome_and_filezilla_no_duplicate_label_1_at_index_0"
        benign_data = np.loadtxt(data_dir_name + ".txt", dtype=float, delimiter=" ")
        benign_data = np.unique(benign_data, axis=0)
        np.random.shuffle(benign_data)
        benign_data = benign_data[:, 1:]
        mal_sample_quantity = 0
        for r in range(1, training_data_container.shape[0]):
            mal_sample_quantity += training_data_container[r].shape[0]
        if mal_sample_quantity > benign_data.shape[0]:
            training_data_container[0] = benign_data[:benign_data.shape[0]]
            testing_data_container[0] = benign_data[benign_data.shape[0]:]
        else:
            training_data_container[0] = benign_data[:mal_sample_quantity]
            testing_data_container[0] = benign_data[mal_sample_quantity:]
    else:
        owl_rule = str(i+1)
        data_dir_name = r"19_owl_rules\no_duplicate_pattern_19_rules\rule_{0}_no_duplicate_with_label_-1_at_index_0".format(owl_rule)
        mal_data = np.loadtxt(data_dir_name + ".txt", dtype=float, delimiter=" ")
        mal_data = np.unique(mal_data, axis=0)
        np.random.shuffle(mal_data)
        mal_data = mal_data[:, 1:]
        if sampling_by_rate:
            training_data_container[i + 1] = mal_data[:int(mal_data.shape[0]*sampling_rate)]
            testing_data_container[i + 1] = mal_data[int(mal_data.shape[0]*sampling_rate):]
        else:
            if sampling_amount > mal_data.shape[0]:
                training_data_container[i + 1] = mal_data[:mal_data.shape[0]]
                testing_data_container[i + 1] = mal_data[mal_data.shape[0]:]
            else:
                training_data_container[i + 1] = mal_data[:sampling_amount]
                testing_data_container[i + 1] = mal_data[sampling_amount:]

for i in range(2):
    # np.savetxt('___rule'+str(i), all_major_samples[i])
    y_element = np.zeros(2)
    y_element[i] = 1
    # print(y_element)

    # print(y_element_arr.shape)
    if i == 0:  # benign
        y_element_arr = np.tile(y_element, training_data_container[0].shape[0]).reshape(-1, 2)
        training_x = training_data_container[0]
        training_y = y_element_arr

    else:  # mal
        for j in range(19):
            y_element_arr = np.tile(y_element, training_data_container[j+1].shape[0]).reshape(-1, 2)
            training_x = np.append(training_x, training_data_container[j+1], axis=0)
            training_y = np.append(training_y, y_element_arr, axis=0)
    print(training_x.shape)
    print(training_y.shape)

bp_times_count = 0
with tf.Graph().as_default():
    with tf.name_scope('placeholder_x'):
        x = tf.placeholder(tf.float64, name='x_placeholder')
    with tf.name_scope('placeholder_y'):
        y_ = tf.placeholder(tf.float64, name='y_placeholder')
    W = tf.Variable(tf.zeros([training_x.shape[1],
                              training_y.shape[1]], dtype=tf.float64), dtype=tf.float64, name='weight')
    b = tf.Variable(tf.zeros([training_y.shape[1]], dtype=tf.float64), dtype=tf.float64, name='bias')
    # W = tf.Variable(np.loadtxt(dir_save_result + r"\weight.txt"), dtype=tf.float64, name='weight')
    # b = tf.Variable(np.loadtxt(dir_save_result + r"\bias.txt"), dtype=tf.float64, name='bias')
    # W = tf.Variable(np.loadtxt(dir_save_result + r"\24719_weight.txt"), dtype=tf.float64, name='weight')
    # b = tf.Variable(np.loadtxt(dir_save_result + r"\24719_bias.txt"), dtype=tf.float64, name='bias')
    # W = tf.Variable(np.loadtxt(new_path + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
    # b = tf.Variable(np.loadtxt(new_path + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    last_loss = 10000000.0
    # last_times_count = np.loadtxt(dir_save_result + r"\times_count.txt")
    # last_execute_time = np.loadtxt(dir_save_result + r"\execute_time.txt")
    # last_times_count = np.loadtxt(dir_save_result + r"\24719_times_count.txt")
    # last_execute_time = np.loadtxt(dir_save_result + r"\24719_execute_time.txt")
    # last_times_count = np.loadtxt(new_path + r"\two_class_times_count.txt")
    # last_execute_time = np.loadtxt(new_path + r"\two_class_execute_time.txt")
    last_times_count = 0
    last_execute_time = 0
    last_correct_rate = 0.0
    execute_start_time = time.time()

    # writer = tf.summary.FileWriter("C:/logfile", sess.graph)
    # writer.close()
    # input(123)

    for i in range(max_bp_times):
        sess.run(train_step, feed_dict={x: training_x, y_: training_y})
        bp_times_count += 1
        correct_rate = sess.run(accuracy, feed_dict={x: training_x, y_: training_y})
        if correct_rate == 1:
            current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
            # np.savetxt(dir_save_result + r"\softmax_weight.txt", current_W)
            # np.savetxt(dir_save_result + r"\softmax_bias.txt", current_b)
            np.savetxt(analyze_result_save_dir_name + r"\two_class_softmax_weight.txt", current_W)
            np.savetxt(analyze_result_save_dir_name + r"\two_class_softmax_bias.txt", current_b)

            # training_detail = open(dir_save_result + r"\_training_detail_softmax.txt", 'w')
            # training_detail.writelines('train times count:' + "\n" + str(i+last_times_count) + "\n")
            # training_detail.writelines("execution time: \n" + str(time.time() - execute_start_time + last_execute_time) + " seconds" + "\n")
            # training_detail.close()

            # current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
            # np.savetxt(dir_save_result + r"\24719_weight.txt", current_W)
            # np.savetxt(dir_save_result + r"\24719_bias.txt", current_b)
            # np.savetxt(dir_save_result + r"\24719_times_count.txt", np.array([i]).reshape(1, 1))
            # np.savetxt(dir_save_result + r"\24719_execute_time.txt",
            #            np.array([time.time() - execute_start_time]).reshape(1, 1))
            break
        if i % 1000 == 0:
            loss = sess.run(cross_entropy, feed_dict={x: training_x, y_: training_y})
            if math.isnan(loss):
                print('loss = nan, train end')
                W = tf.Variable(np.loadtxt(new_path + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
                b = tf.Variable(np.loadtxt(new_path + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')

                break
            # loss = sess.run(average_squared_residual, feed_dict={x: training_x, y_: training_y})
            if last_loss > loss and correct_rate >= last_correct_rate:
                current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
                # np.savetxt(dir_save_result + r"\weight.txt", current_W)
                # np.savetxt(dir_save_result + r"\bias.txt", current_b)
                # np.savetxt(dir_save_result + r"\times_count.txt", np.array([i+last_times_count]).reshape(1, 1))
                # np.savetxt(dir_save_result + r"\execute_time.txt", np.array([time.time() - execute_start_time + last_execute_time]).reshape(1, 1))

                np.savetxt(new_path + r"\two_class_weight.txt", current_W)
                np.savetxt(new_path + r"\two_class_bias.txt", current_b)
                np.savetxt(new_path + r"\two_class_times_count.txt", np.array([i + last_times_count]).reshape(1, 1))
                np.savetxt(new_path + r"\two_class_execute_time.txt", np.array([time.time() - execute_start_time + last_execute_time]).reshape(1, 1))

                # current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
                # np.savetxt(dir_save_result + r"\24719_weight.txt", current_W)
                # np.savetxt(dir_save_result + r"\24719_bias.txt", current_b)
                # np.savetxt(dir_save_result + r"\24719_times_count.txt", np.array([i+last_times_count]).reshape(1, 1))
                # np.savetxt(dir_save_result + r"\24719_execute_time.txt",
                #            np.array([time.time() - execute_start_time+last_times_count]).reshape(1, 1))

                # current_hw, current_ht, current_ow, current_ot = sess.run([hidden_weights, hidden_thresholds, output_weights, output_threshold], feed_dict={x: training_x, y_: training_y})
                if last_loss - loss < 1e-5:
                    print('cross entropy change too slow, train end.')
                    current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
                    np.savetxt(new_path + r"\two_class_weight.txt", current_W)
                    np.savetxt(new_path + r"\two_class_bias.txt", current_b)
                    np.savetxt(new_path + r"\two_class_times_count.txt", np.array([i + last_times_count]).reshape(1, 1))
                    np.savetxt(new_path + r"\two_class_execute_time.txt",
                               np.array([time.time() - execute_start_time + last_execute_time]).reshape(1, 1))
                    break
                last_loss = loss
                last_correct_rate = correct_rate
            print('training data predict accuracy: '+str(correct_rate * 100)+'%   cross entropy: '+str(loss))
        if correct_rate >= target_accuracy:
            print('training data correct rate >= {0}%, train end'.format(target_accuracy*100))
            current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
            np.savetxt(new_path + r"\two_class_weight.txt", current_W)
            np.savetxt(new_path + r"\two_class_bias.txt", current_b)
            np.savetxt(new_path + r"\two_class_times_count.txt", np.array([i + last_times_count]).reshape(1, 1))
            np.savetxt(new_path + r"\two_class_execute_time.txt", np.array([time.time() - execute_start_time + last_execute_time]).reshape(1, 1))
            break
    # tf.train.SummaryWriter soon be deprecated, use following
    # writer = tf.summary.FileWriter("C:/logfile", sess.graph)
    # writer.close()

    predict_y = sess.run([y], {x: training_x, y_: training_y})
    correct_rate = sess.run(accuracy, feed_dict={x: training_x, y_: training_y})
    loss = sess.run(cross_entropy, feed_dict={x: training_x, y_: training_y})

    file = open(new_path + r"\_two_class_training_detail.txt", 'w')
    file.writelines("learning_rate: " + str(learning_rate_eta) + "\n")
    file.writelines("input_node_amount: " + str(training_x.shape[1]) + "\n")
    file.writelines("output_node_amount: " + str(training_y.shape[1]) + "\n")
    file.writelines("training_data_amount: " + str(training_x.shape[0]) + "\n")
    file.writelines("training_data_classification_correct_rate: " + str(correct_rate*100) + "%\n")
    file.writelines("loss_of_the_model: " + str(loss) + "\n")
    file.writelines("bp_times_count: " + str(bp_times_count) + "\n")
    file.writelines("total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
    # file.writelines("potential anomaly amount: {0}(Benign)    {1}(Malware)\n".format(potential_anomaly_benign_amount, potential_anomaly_mal_amount))

    if sampling_by_rate:
        file.writelines("sampling rate: {0}\n".format(sampling_rate))
    else:
        file.writelines("sampling amount: {0}\n".format(sampling_amount))
    file.writelines("training data amount in each rule: [")
    for i in range(training_data_container.shape[0]):
        file.writelines("{0}, ".format(training_data_container[i].shape[0]))
    file.writelines("]\n")
    file.writelines("testing data in each rule: [")
    for i in range(testing_data_container.shape[0]):
        file.writelines("{0}, ".format(testing_data_container[i].shape[0]))
    file.writelines("]\n")

    file.close()
    print("bp times: %s" % bp_times_count)
    print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))

    # analyze part
    for i in range(2):
        total_correct_count = 0
        total_sample_count = 0
        if i == 0:
            file = open(new_path + r"\_training_data_analyze.txt", 'w')
        elif i == 1:
            file = open(new_path + r"\_testing_data_analyze.txt", 'w')
        for j in range(training_data_container.shape[0]):
            if i == 0:
                input_data = training_data_container[j]
            elif i == 1:
                input_data = testing_data_container[j]
            x_input_data = input_data
            y_element = np.zeros(2)
            total_sample_count += x_input_data.shape[0]

            # predict_y = sess.run([output_layer], {x_placeholder: x_input_data})[0]
            if j == 0:
                y_element[0] = 1
                y_input_data = np.tile(y_element, x_input_data.shape[0]).reshape(-1, 2)
                predict_acc = sess.run([accuracy], feed_dict={x: x_input_data, y_: y_input_data})[0]
                if math.isnan(predict_acc):
                    classify_as_mal_count = 0
                else:
                    classify_as_benign_count = int(round(x_input_data.shape[0]*predict_acc))
                total_correct_count += classify_as_benign_count
                file.writelines("benign classification accuracy: {0}/{1}\n".format(classify_as_benign_count, x_input_data.shape[0]))
            else:
                y_element[1] = 1
                y_input_data = np.tile(y_element, x_input_data.shape[0]).reshape(-1, 2)
                predict_acc = sess.run([accuracy], feed_dict={x: x_input_data, y_: y_input_data})[0]
                if math.isnan(predict_acc):
                    classify_as_mal_count = 0
                else:
                    classify_as_mal_count = int(round(x_input_data.shape[0]*predict_acc))
                total_correct_count += classify_as_mal_count
                file.writelines("rule {0} classification accuracy: {1}/{2}\n".format(j, classify_as_mal_count, x_input_data.shape[0]))
        file.writelines("total classification accuracy: {0}/{1}\n".format(total_correct_count, total_sample_count))

        file.close()
