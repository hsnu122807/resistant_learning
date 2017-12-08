import tensorflow as tf
import numpy as np
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))

all_major_samples = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=object)

# owl_rule = "2"
# data_amount = "100"
sigma = "2"
# owl_rules = np.arange(1, 20)
owl_rules = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
data_amount = ['65', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '39', '100', '40', '100', '100', '100', '100', ]
outlier_amount = np.array([6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 10, 4, 10, 10, 10, 10], dtype=int)
# owl_rules = np.array([16], dtype=str)
# data_amount = np.array([100], dtype=str)
# outlier_amount = np.array([10], dtype=int)

# print(owl_rules)
for i in range(owl_rules.shape[0]):
    # print('rule '+owl_rules[i])
    dir_target = dir_path + r"\19_owl_rules\owl_rule_"+str(owl_rules[i])+"_"+data_amount[i]+"_and_benign_"+data_amount[i]+"_sigma_" + sigma

    training_data_result = np.loadtxt(dir_target+r"\training_data_residual_predict_output_desire_output_desire_input.txt")
    major_data_x = training_data_result[:int(int(data_amount[i]) * 2 - outlier_amount[i]), 3:]
    major_data_y = training_data_result[:int(int(data_amount[i]) * 2 - outlier_amount[i]), 2].reshape((-1, 1))
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

for i in range(all_major_samples.shape[0]):
    # np.savetxt('___rule'+str(i), all_major_samples[i])
    if i == 0:
        major_samples_arr = all_major_samples[i]
    else:
        major_samples_arr = np.append(major_samples_arr, all_major_samples[i], axis=0)

for i in range(1, 20):

    for j in range(all_major_samples.shape[0]):
        if j == 0:
            current_stage_y = np.tile(100, all_major_samples[j].shape[0])
        elif j == i:
            current_stage_y = np.append(current_stage_y, np.tile(-100, all_major_samples[j].shape[0]), axis=0)
        else:
            current_stage_y = np.append(current_stage_y, np.tile(100, all_major_samples[j].shape[0]), axis=0)
    current_stage_y = current_stage_y.reshape(-1, 1)

    major_data_x_mal_part = np.arange(0)
    major_data_x_benign_part = np.arange(0)
    # print(major_data_x)
    # print(major_data_y)

    major_data_x_mal_part = major_samples_arr[np.where(current_stage_y == -100)[0]]
    major_data_x_benign_part = major_samples_arr[np.where(current_stage_y == 100)[0]]
    # np.savetxt("___.txt", major_data_x_mal_part)
    # np.savetxt("____.txt", major_data_x_benign_part)
    # input(123)

    execute_start_time = time.time()

    learning_rate_eta = 0.00001
    # squared_residual_tolerance = 0.01
    bp_times_limit = 50000

    print('rule ' + owl_rules[i-1])
    dir_target = dir_path + r"\19_owl_rules\owl_rule_" + str(owl_rules[i-1]) + "_" + data_amount[i-1] + "_and_benign_" + data_amount[i-1] + "_sigma_" + sigma

    training_detail = open(dir_target + r"\precise_classify_neuron_network_detail.txt", 'w')

    hidden_node_amount = 0
    bp_times_count = 0
    hidden_node_is_not_enough = True
    while hidden_node_is_not_enough:
        hidden_node_amount += 1
        with tf.Graph().as_default():
            x_placeholder = tf.placeholder(tf.float64)
            y_placeholder = tf.placeholder(tf.float64)

            hw = tf.random_normal([major_data_x.shape[1], hidden_node_amount], dtype=tf.float64, mean=0, stddev=1)
            ht = tf.random_normal([hidden_node_amount], dtype=tf.float64, mean=0, stddev=1)
            ow = tf.random_normal([hidden_node_amount, major_data_y.shape[1]], dtype=tf.float64, mean=0, stddev=1)
            ot = tf.random_normal([major_data_y.shape[1]], dtype=tf.float64, mean=0, stddev=1)

            output_threshold = tf.Variable(ot, dtype=tf.float64)
            output_weights = tf.Variable(ow, dtype=tf.float64)
            hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
            hidden_weights = tf.Variable(hw, dtype=tf.float64)

            # hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
            hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds)/1024)
            output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

            average_squared_residual = tf.reduce_mean(
                tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
            train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_squared_residual)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            can_not_classify = True
            counter = 0
            # last_max_residual = 100000000.0
            last_alpha_minus_beta = -100000000.0
            bp_not_good_count = 0
            while can_not_classify:
                # predict_y = sess.run([output_layer], {x_placeholder: major_samples_arr, y_placeholder: current_stage_y})
                # current_stage_squared_residuals = np.square(current_stage_y - predict_y[0])
                # current_stage_max_residual = max(current_stage_squared_residuals)

                predict_mal = sess.run([output_layer], {x_placeholder: major_data_x_mal_part})[0]
                predict_benign = sess.run([output_layer], {x_placeholder: major_data_x_benign_part})[0]
                alpha = min(predict_benign)[0]
                beta = max(predict_mal)[0]
                if (counter % 300) == 0:
                    # print(current_stage_max_residual)
                    alpha_minus_beta = alpha - beta
                    print('alpha: '+str(alpha)+'   beta: '+str(beta))
                    if counter > bp_times_limit:
                        print('bp times larger than limit, ' + str(hidden_node_amount) + ' hidden node not enough, add another hidden node')
                        can_not_classify = False
                    if (alpha_minus_beta - last_alpha_minus_beta) <= 0.0:
                        bp_not_good_count += 1
                    else:
                        bp_not_good_count = 0
                        last_alpha_minus_beta = alpha_minus_beta
                    if bp_not_good_count > 2:
                        print('last_alpha_minus_beta > alpha_minus_beta, ' + str(
                            hidden_node_amount) + ' hidden node can not learning well, add another hidden node')
                        can_not_classify = False

                if alpha > beta:
                    # print(current_stage_max_residual)
                    print('finish training after ' + str(counter) + ' times bp.')
                    can_not_classify = False
                    hidden_node_is_not_enough = False

                    curr_hidden_neuron_weight, curr_hidden_threshold, curr_output_neuron_weight, curr_output_threshold = sess.run(
                        [hidden_weights, hidden_thresholds,
                         output_weights, output_threshold],
                        {x_placeholder: major_data_x, y_placeholder: major_data_y})
                    np.savetxt(dir_target + r"\precise_classify_neuron_network_hidden_neuron_weight.txt",
                               curr_hidden_neuron_weight)
                    np.savetxt(dir_target + r"\precise_classify_neuron_network_hidden_threshold.txt",
                               curr_hidden_threshold)
                    np.savetxt(dir_target + r"\precise_classify_neuron_network_output_neuron_weight.txt",
                               curr_output_neuron_weight)
                    np.savetxt(dir_target + r"\precise_classify_neuron_network_output_threshold.txt",
                               curr_output_threshold)

                    # print(major_data_x_mal_part)
                    # print(major_data_x_benign_part)
                    # predict_mal = sess.run([output_layer], {x_placeholder: major_data_x_mal_part})[0]
                    # predict_benign = sess.run([output_layer], {x_placeholder: major_data_x_benign_part})[0]
                    # alpha = min(predict_benign)[0]
                    # beta = max(predict_mal)[0]
                    print('alpha(class 1 min):')
                    print(alpha)
                    print('beta(class 2 max):')
                    print(beta)
                    print('(alpha + beta) / 2:')
                    print((alpha + beta) / 2)
                    training_detail.writelines('alpha(class 1 min):' + "\n" + str(alpha) + "\n")
                    training_detail.writelines('beta(class 2 max):' + "\n" + str(beta) + "\n")
                    training_detail.writelines('(alpha + beta) / 2:' + "\n" + str((alpha + beta) / 2) + "\n")
                    training_detail.writelines('hidden node amount:' + "\n" + str(hidden_node_amount) + "\n")
                    training_detail.writelines('total bp times:' + "\n" + str(bp_times_count) + "\n")
                    training_detail.writelines(
                        "total execution time: \n" + str(time.time() - execute_start_time) + " seconds" + "\n")
                    training_detail.close()
                else:
                    if hidden_node_amount > 300:
                        print('rule ' + owl_rules[i-1]+' training failed.')
                        can_not_classify = False
                        hidden_node_is_not_enough = False
                    else:
                        bp_times_count += 1
                        counter += 1
                        sess.run(train, feed_dict={x_placeholder: major_data_x, y_placeholder: major_data_y})
