import tensorflow as tf
import numpy as np
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))

x_placeholder = tf.placeholder(tf.float64)
y_placeholder = tf.placeholder(tf.float64)
learning_rate_eta = 0.005
squared_residual_tolerance = 0.001
bp_times_limit = 300000

# # rule = "1"
# # ntu = "1"
# # data_amount = "1250"
# rules = [10]
# data_amount = ['1250']
# ntu = ['3']
# light = ['']
# # # light = "" or "_light"
# # light = ""
# sigma = "2"
# for i in range(3):
#     dir_target = dir_path + r"\TensorFlow_input_detection_rule_"+str(rules[i])+"_"+data_amount[i]+"_and_ntu_"+ntu[i]+"_benign_"+data_amount[i]+light[i]+"_no_label"+"_sigma_" + sigma

# owl_rule = "2"
# data_amount = "100"
sigma = "2"
# owl_rules = np.arange(1, 20)
# owl_rules = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
# data_amount = ['65', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '39', '100', '40', '100', '100', '100', '100', ]
# outlier_amount = np.array([6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 10, 4, 10, 10, 10, 10], dtype=int)
owl_rules = np.array([5, 6, 16], dtype=str)
data_amount = ['100', '100', '100']
outlier_amount = np.array([10, 10, 10], dtype=int)

# print(owl_rules)
for i in range(owl_rules.shape[0]):
    print('rule '+owl_rules[i])
    dir_target = dir_path + r"\19_owl_rules\owl_rule_"+str(owl_rules[i])+"_"+data_amount[i]+"_and_benign_"+data_amount[i]+"_sigma_" + sigma

    # create file to save training process
    training_detail = open(dir_target + r"\_second_phase_classify_neuron_network_detail.txt", 'w')

    training_data_result = np.loadtxt(dir_target+r"\training_data_residual_predict_output_desire_output_desire_input.txt")
    # major_data_x = training_data_result[:int(int(data_amount[i]) * 1.9), 3:]
    # major_data_y = training_data_result[:int(int(data_amount[i]) * 1.9), 2].reshape((-1, 1))
    major_data_x = training_data_result[:int(int(data_amount[i]) * 2 - outlier_amount[i]), 3:]
    major_data_y = training_data_result[:int(int(data_amount[i]) * 2 - outlier_amount[i]), 2].reshape((-1, 1))
    # print(training_data_result[:int(10), 2])

    execute_start_time = time.time()

    hidden_node_amount = 0
    bp_times_count = 0
    hidden_node_is_not_enough = True
    while hidden_node_is_not_enough:
        hidden_node_amount += 1

        # read outlier nn from file(已驗證效果不好，寧願直接拿原始的nn來做分類，bp會讓nn變糟)
        # ot = np.loadtxt(dir_target+r"\output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
        # ow = np.loadtxt(dir_target+r"\output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
        # ht = np.loadtxt(dir_target+r"\hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
        # hw = np.loadtxt(dir_target+r"\hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))

        hw = tf.random_normal([major_data_x.shape[1], hidden_node_amount], dtype=tf.float64)
        ht = tf.random_normal([hidden_node_amount], dtype=tf.float64)
        ow = tf.random_normal([hidden_node_amount, major_data_y.shape[1]], dtype=tf.float64)
        ot = tf.random_normal([major_data_y.shape[1]], dtype=tf.float64)

        output_threshold = tf.Variable(ot, dtype=tf.float64)
        output_weights = tf.Variable(ow, dtype=tf.float64)
        hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
        hidden_weights = tf.Variable(hw, dtype=tf.float64)

        hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
        output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

        average_squared_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
        train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_squared_residual)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        not_in_tolerance = True
        counter = 0
        last_max_residual = 100000000.0
        while not_in_tolerance:
            predict_y = sess.run([output_layer], {x_placeholder: major_data_x, y_placeholder: major_data_y})
            current_stage_squared_residuals = np.square(major_data_y - predict_y[0])
            current_stage_max_residual = max(current_stage_squared_residuals)
            if (counter % 500) == 0:
                print(current_stage_max_residual)
                if counter > bp_times_limit:
                    print(str(hidden_node_amount) + ' hidden node can not learning well, add another hidden node')
                    not_in_tolerance = False
                if (last_max_residual - current_stage_max_residual) < 0.001:
                    print('last_max_residual < current_stage_max_residual, '+str(hidden_node_amount) + ' hidden node can not learning well, add another hidden node')
                    not_in_tolerance = False
                else:
                    last_max_residual = current_stage_max_residual

            if current_stage_max_residual < squared_residual_tolerance:
                print(current_stage_max_residual)
                print('finish training after ' + str(counter) + ' times bp.')
                not_in_tolerance = False
                hidden_node_is_not_enough = False

                curr_hidden_neuron_weight, curr_hidden_threshold, curr_output_neuron_weight, curr_output_threshold = sess.run(
                    [hidden_weights, hidden_thresholds,
                     output_weights, output_threshold],
                    {x_placeholder: major_data_x, y_placeholder: major_data_y})
                np.savetxt(dir_target + r"\second_phase_classify_neuron_network_hidden_neuron_weight.txt", curr_hidden_neuron_weight)
                np.savetxt(dir_target + r"\second_phase_classify_neuron_network_hidden_threshold.txt", curr_hidden_threshold)
                np.savetxt(dir_target + r"\second_phase_classify_neuron_network_output_neuron_weight.txt", curr_output_neuron_weight)
                np.savetxt(dir_target + r"\second_phase_classify_neuron_network_output_threshold.txt", curr_output_threshold)

                major_data_x_mal_part = np.arange(0)
                major_data_x_benign_part = np.arange(0)
                # print(major_data_x)
                # print(major_data_y)
                for j in range(major_data_x.shape[0]):
                    if major_data_y[j] == -1:
                        major_data_x_mal_part = np.append(major_data_x_mal_part, major_data_x[j])
                    if major_data_y[j] == 1:
                        major_data_x_benign_part = np.append(major_data_x_benign_part, major_data_x[j])
                major_data_x_mal_part = major_data_x_mal_part.reshape((-1, major_data_x.shape[1]))
                major_data_x_benign_part = major_data_x_benign_part.reshape((-1, major_data_x.shape[1]))
                # print(major_data_x_mal_part)
                # print(major_data_x_benign_part)
                predict_mal = sess.run([output_layer], {x_placeholder: major_data_x_mal_part})[0]
                predict_benign = sess.run([output_layer], {x_placeholder: major_data_x_benign_part})[0]
                alpha = min(predict_benign)[0]
                beta = max(predict_mal)[0]
                print('alpha(class 1 min):')
                print(alpha)
                print('beta(class 2 max):')
                print(beta)
                print('(alpha + beta) / 2:')
                print((alpha+beta)/2)
                training_detail.writelines('alpha(class 1 min):'+"\n"+str(alpha)+"\n")
                training_detail.writelines('beta(class 2 max):'+"\n"+str(beta)+"\n")
                training_detail.writelines('(alpha + beta) / 2:'+"\n"+str((alpha+beta)/2)+"\n")
                training_detail.writelines('hidden node amount:' + "\n" + str(hidden_node_amount) + "\n")
                training_detail.writelines('total bp times:' + "\n" + str(bp_times_count) + "\n")
                training_detail.writelines("total execution time: \n" + str(time.time() - execute_start_time) + " seconds" + "\n")
                training_detail.close()
            else:
                bp_times_count += 1
                counter += 1
                sess.run(train, feed_dict={x_placeholder: major_data_x, y_placeholder: major_data_y})




