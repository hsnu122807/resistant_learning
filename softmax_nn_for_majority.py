import tensorflow as tf
import numpy as np
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
all_major_samples = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=object)


for i in range(19):
    # print('rule '+owl_rules[i])
    # dir_target = dir_path + r"\19_owl_rules\owl_rule_"+str(i+1)+"_all_training_data_all"
    dir_target = dir_path + r"\19_owl_rules\owl_rule_" + str(i + 1) + "_all_training_data_sigma_2"

    # training_data_result = np.loadtxt(dir_target+r"\two_class_training_data_fit_x_y_yp.txt")
    # major_data_x = training_data_result[:int(training_data_result.shape[0] * 0.95), 1:53]
    # major_data_y = training_data_result[:int(training_data_result.shape[0] * 0.95), 53].reshape((-1, 1))

    training_data_result = np.loadtxt(dir_target + r"\training_data_residual_predict_output_desire_output_desire_input.txt")
    major_data_x = training_data_result[:int(training_data_result.shape[0] * 0.95), 3:55]
    major_data_y = training_data_result[:int(training_data_result.shape[0] * 0.95), 2].reshape((-1, 1))

    # print(major_data_x.shape[0])
    # print(major_data_y.shape[0])
    major_data_x_mal_part = major_data_x[np.where(major_data_y == -1)[0]]
    major_data_x_benign_part = major_data_x[np.where(major_data_y == 1)[0]]
    major_data_x_mal_part = np.unique(major_data_x_mal_part, axis=0)
    all_major_samples[i+1] = major_data_x_mal_part
    if i == 0:
        all_major_samples[0] = major_data_x_benign_part
    else:
        all_major_samples[0] = np.append(all_major_samples[0], major_data_x_benign_part, axis=0)
all_major_samples[0] = np.unique(all_major_samples[0], axis=0)
print(all_major_samples[0].shape)

for d in range(20):
    if d == 0:
        np.savetxt("all_binary_majority_benign.txt", all_major_samples[d])
    else:
        np.savetxt("all_binary_majority_rule_"+str(d)+".txt", all_major_samples[d])
# input(123)

# for i in range(20):
#     # print('rule '+owl_rules[i])
#     dir_target = dir_path + r"\19_owl_rules"
#     if i == 0:
#         chrome = np.loadtxt(dir_target+r'\benign_chrome_sample.txt', dtype=float)
#         filezilla = np.loadtxt(dir_target+r'\benign_filezilla_sample.txt', dtype=float)
#         training_data = np.concatenate((chrome, filezilla), axis=0)
#     else:
#         training_data = np.loadtxt(dir_target+r"\owl_rule_"+str(i)+".txt", dtype=str)[:, :52]
#         training_data = np.ndarray.astype(training_data, float)
#     training_data = np.unique(training_data, axis=0)
#     # print(training_data.shape)
#     all_major_samples[i] = training_data
# # input(123)

for i in range(all_major_samples.shape[0]):
    # np.savetxt('___rule'+str(i), all_major_samples[i])
    y_element = np.zeros(20)
    y_element[i] = 1
    # print(y_element)
    y_element_arr = np.tile(y_element, all_major_samples[i].shape[0]).reshape(-1, 20)
    # print(y_element_arr.shape)
    if i == 0:
        training_x = all_major_samples[i]
        training_y = y_element_arr
    else:
        training_x = np.append(training_x, all_major_samples[i], axis=0)
        training_y = np.append(training_y, y_element_arr, axis=0)
print(training_x.shape)
# input(123)
# training_x.shape == (24719, 52)
# np.savetxt("_.txt", training_y.reshape(-1, 20))

# dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_binary"
dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_envelope"
bp_times_count = 0
with tf.Graph().as_default():
    with tf.name_scope('placeholder_x'):
        x = tf.placeholder(tf.float64, [training_x.shape[0], training_x.shape[1]], name='x_placeholder')
    with tf.name_scope('placeholder_y'):
        y_ = tf.placeholder(tf.float64, [training_y.shape[0], training_y.shape[1]], name='y_placeholder')
    # W = tf.Variable(tf.zeros([training_x.shape[1], training_y.shape[1]], dtype=tf.float64), dtype=tf.float64, name='weight')
    # b = tf.Variable(tf.zeros([training_y.shape[1]], dtype=tf.float64), dtype=tf.float64, name='bias')
    # W = tf.Variable(np.loadtxt(dir_save_result + r"\weight.txt"), dtype=tf.float64, name='weight')
    # b = tf.Variable(np.loadtxt(dir_save_result + r"\bias.txt"), dtype=tf.float64, name='bias')
    # W = tf.Variable(np.loadtxt(dir_save_result + r"\24719_weight.txt"), dtype=tf.float64, name='weight')
    # b = tf.Variable(np.loadtxt(dir_save_result + r"\24719_bias.txt"), dtype=tf.float64, name='bias')
    W = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
    b = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    last_loss = 10000000.0
    # last_times_count = np.loadtxt(dir_save_result + r"\times_count.txt")
    # last_execute_time = np.loadtxt(dir_save_result + r"\execute_time.txt")
    # last_times_count = np.loadtxt(dir_save_result + r"\24719_times_count.txt")
    # last_execute_time = np.loadtxt(dir_save_result + r"\24719_execute_time.txt")
    last_times_count = np.loadtxt(dir_save_result + r"\two_class_times_count.txt")
    last_execute_time = np.loadtxt(dir_save_result + r"\two_class_execute_time.txt")
    # last_times_count = 0
    # last_execute_time = 0
    last_correct_rate = 0.923
    execute_start_time = time.time()

    # writer = tf.summary.FileWriter("C:/logfile", sess.graph)
    # writer.close()
    # input(123)

    for i in range(10000000):
        sess.run(train_step, feed_dict={x: training_x, y_: training_y})
        correct_rate = sess.run(accuracy, feed_dict={x: training_x, y_: training_y})
        if correct_rate == 1:
            current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
            # np.savetxt(dir_save_result + r"\softmax_weight.txt", current_W)
            # np.savetxt(dir_save_result + r"\softmax_bias.txt", current_b)
            np.savetxt(dir_save_result + r"\two_class_softmax_weight.txt", current_W)
            np.savetxt(dir_save_result + r"\two_class_softmax_bias.txt", current_b)

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
            # loss = sess.run(average_squared_residual, feed_dict={x: training_x, y_: training_y})
            if last_loss > loss and correct_rate >= last_correct_rate:
                current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
                # np.savetxt(dir_save_result + r"\weight.txt", current_W)
                # np.savetxt(dir_save_result + r"\bias.txt", current_b)
                # np.savetxt(dir_save_result + r"\times_count.txt", np.array([i+last_times_count]).reshape(1, 1))
                # np.savetxt(dir_save_result + r"\execute_time.txt", np.array([time.time() - execute_start_time + last_execute_time]).reshape(1, 1))

                np.savetxt(dir_save_result + r"\two_class_weight.txt", current_W)
                np.savetxt(dir_save_result + r"\two_class_bias.txt", current_b)
                np.savetxt(dir_save_result + r"\two_class_times_count.txt", np.array([i + last_times_count]).reshape(1, 1))
                np.savetxt(dir_save_result + r"\two_class_execute_time.txt", np.array([time.time() - execute_start_time + last_execute_time]).reshape(1, 1))

                # current_W, current_b = sess.run([W, b], feed_dict={x: training_x, y_: training_y})
                # np.savetxt(dir_save_result + r"\24719_weight.txt", current_W)
                # np.savetxt(dir_save_result + r"\24719_bias.txt", current_b)
                # np.savetxt(dir_save_result + r"\24719_times_count.txt", np.array([i+last_times_count]).reshape(1, 1))
                # np.savetxt(dir_save_result + r"\24719_execute_time.txt",
                #            np.array([time.time() - execute_start_time+last_times_count]).reshape(1, 1))

                # current_hw, current_ht, current_ow, current_ot = sess.run([hidden_weights, hidden_thresholds, output_weights, output_threshold], feed_dict={x: training_x, y_: training_y})

                last_loss = loss
                last_correct_rate = correct_rate
            print('training data predict accuracy: '+str(correct_rate * 100)+'%   cross entropy: '+str(loss))
