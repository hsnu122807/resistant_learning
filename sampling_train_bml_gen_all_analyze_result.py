# coding:utf-8
import tensorflow as tf
import numpy as np
import time
import os

dir_path = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\mix_19_rules_binary_classification\bml"

# 可能會手動調整的參數
every_stage_max_thinking_times = 1
outlier_rate = 0.05
sampling_rate = 0.1
sampling_amount = 100
sampling_by_rate = False  # if False: sampling by fix amount
# analyze_result_save_dir_name = "all_rules_data_sample_all_bml_separate_benign_and_malicious"
# analyze_result_save_dir_name = "all_rules_data_sample_1_of_10_bml_separate_benign_and_malicious"
analyze_result_save_dir_name = "all_rules_data_sample_100_bml_separate_benign_and_malicious"

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

# 把要拿來當first slfn計算對象的資料放在前m+1個
each_rule_take_amount = 2  # 初始slfn會拿19rule*2其餘benign
for i in range(1, 20):
    if i == 1:
        heads = training_data_container[i][:each_rule_take_amount]
        remains = training_data_container[i][each_rule_take_amount:]
    else:
        heads = np.concatenate((heads, training_data_container[i][:each_rule_take_amount]), axis=0)
        remains = np.concatenate((remains, training_data_container[i][each_rule_take_amount:]), axis=0)

h_b = np.concatenate((heads, training_data_container[0]), axis=0)
all_training_data = np.concatenate((h_b, remains), axis=0)
# input(123)
x_training_data = all_training_data[:, 1:]
y_training_data = all_training_data[:, 0].reshape((-1, 1))
x_training_data_mal_part = x_training_data[np.where(y_training_data == -1)[0]]
x_training_data_benign_part = x_training_data[np.where(y_training_data == 1)[0]]
y_training_data_mal_part = y_training_data[np.where(y_training_data == -1)[0]]
y_training_data_benign_part = y_training_data[np.where(y_training_data == 1)[0]]

mal_sample_amount = y_training_data_mal_part.shape[0]
benign_sample_amount = y_training_data_benign_part.shape[0]

execute_start_time = time.time()

# Network Parameters
input_node_amount = x_training_data.shape[1]
hidden_node_amount = 1
output_node_amount = 1
learning_rate_eta = 0.0000005

# Parameters
data_size = x_training_data.shape[0]
zeta = 0.05
Lambda = 100000

# create folder to save training process
new_path = r"{0}/".format(dir_path) + analyze_result_save_dir_name
if not os.path.exists(new_path):
    os.makedirs(new_path)

# create file to save training process
training_process_log = open(new_path + r"\_two_class_training_process.txt", 'w')

# counters
thinking_times_count = 0
cramming_times_count = 0
softening_thinking_times_count = 0
pruning_success_times_count = 0

# init = tf.global_variables_initializer()
sess = tf.Session()
# sess.run(init)

with tf.name_scope('calculate_first_slfn_weights'):
    # 首先架構初始SLFN
    m = input_node_amount
    first_slfn_output_weight = (np.max(y_training_data) - np.min(y_training_data) + 2.0).reshape(1, 1)
    first_slfn_output_threshold = (np.min(y_training_data) - 1.0).reshape(1)
    desi_slice_y = y_training_data[:m+1]
    yc = np.arctanh((desi_slice_y - first_slfn_output_threshold) / first_slfn_output_weight).reshape(m+1, 1)
    desi_slice_x = x_training_data[:m+1]
    hidden_node_threshold_vector = tf.ones([m + 1, 1], dtype=tf.float64, name='one')
    xc = sess.run(tf.concat(axis=1, values=[desi_slice_x, hidden_node_threshold_vector]))
    answer = sess.run(tf.matrix_solve_ls(xc, yc, fast=False))
    first_slfn_hidden_weight = answer[:m]
    first_slfn_hidden_threshold = answer[m:]

# 架構第一個SLFN的tensor
# placeholders
with tf.name_scope('inputs'):
    x_placeholder = tf.placeholder(tf.float64, name='x_input')
    y_placeholder = tf.placeholder(tf.float64, name='y_input')

# network architecture
with tf.name_scope('hidden_layer'):
    hidden_thresholds = tf.Variable(first_slfn_hidden_threshold, dtype=tf.float64, name='hidden_threshold')
    hidden_weights = tf.Variable(first_slfn_hidden_weight, dtype=tf.float64, name='hidden_weight')
    hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
with tf.name_scope('output_layer'):
    output_threshold = tf.Variable(first_slfn_output_threshold, dtype=tf.float64, name='output_threshold')
    output_weights = tf.Variable(first_slfn_output_weight, dtype=tf.float64, name='hidden_weight')
    output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

# learning goal & optimizer
with tf.name_scope('loss'):
    average_square_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_square_residual)

# saver
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

with tf.name_scope('calculate_alpha_T'):
    # alpha_T的算法，會用在加hidden node的時候
    beta_k_placeholder = tf.placeholder(tf.float64, name='beta_k')
    x_c_placeholder = tf.placeholder(tf.float64, name='x_c')
    x_k_placeholder = tf.placeholder(tf.float64, name='x_k')
    test = tf.sqrt(tf.reduce_sum(tf.square(beta_k_placeholder, name='square'), name='reduce_sum'), name='sqrt')
    alpha = tf.div(beta_k_placeholder, test, name='alpha')
    alpha_T = tf.transpose(alpha, name='transpose')
    Cal_table2 = tf.reduce_sum(tf.matmul(tf.subtract(x_c_placeholder, x_k_placeholder), alpha_T), name='test_alpha')

with tf.name_scope('calculate_new_hidden_threshold'):
    # new hidden node threshold的算法
    alpha_T_placeholder = tf.placeholder(tf.float64, name='alpha_T')
    with tf.name_scope('hidden_threshold_1'):
        calculate_new_hidden_node_1_threshold = zeta - Lambda * tf.matmul(x_k_placeholder, alpha_T_placeholder)
    with tf.name_scope('hidden_threshold_2'):
        calculate_new_hidden_node_2_threshold = zeta + Lambda * tf.matmul(x_k_placeholder, alpha_T_placeholder)

with tf.name_scope('calculate_new_output_weight'):
    # calculate new output weight
    y_k_minus_output_placeholder = tf.placeholder(tf.float64, name='y_k_minus_output')
    two = tf.cast(2.0, tf.float64, name='2')
    calculate_new_output_weight = y_k_minus_output_placeholder / (two * tf.cast(tf.tanh(zeta), tf.float64))

# pick n most fit data(每個stage都要維持alpha-beta的最大值)
predict_y_mal_part = sess.run([output_layer], {x_placeholder: x_training_data_mal_part, y_placeholder: y_training_data_mal_part})[0]
predict_y_benign_part = sess.run([output_layer], {x_placeholder: x_training_data_benign_part,
                                                  y_placeholder: y_training_data_benign_part})[0]
min_mal_predict_value = min(predict_y_mal_part)
max_benign_predict_value = max(predict_y_benign_part)
# print("max class 1: {0}    min class 2: {1}".format(max_benign_predict_value, min_mal_predict_value))
close_to_most_mal_value_mal_part = predict_y_mal_part - min_mal_predict_value
close_to_most_benign_value_benign_part = max_benign_predict_value - predict_y_benign_part

x_y_mal_part = np.concatenate((x_training_data_mal_part, y_training_data_mal_part), axis=1)
x_y_yp_mal_part = np.concatenate((x_y_mal_part, predict_y_mal_part), axis=1)
value_x_y_yp_mal_part = np.concatenate((close_to_most_mal_value_mal_part, x_y_yp_mal_part), axis=1)
# value_x_y_yp_mal_part_sorted_by_value = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, 0])]
value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1] - 1])]

x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
# value_x_y_yp_benign_part_sorted_by_value = value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, 0])]
value_x_y_yp_benign_part_sorted_by_yp = np.flip(value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, value_x_y_yp_benign_part.shape[1] - 1])], axis=0)
# global max alpha - beta
if (m+1 - 2) < value_x_y_yp_benign_part_sorted_by_yp.shape[0]:
    mal_index = 0
    benign_index = m+1 - 2
    global_max_mal_index = mal_index
    global_max_benign_index = benign_index
else:
    mal_index = (m+1 - 2) - (value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1)
    benign_index = value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1
    global_max_mal_index = mal_index
    global_max_benign_index = benign_index
global_max_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
while mal_index < (m+1 - 2) and mal_index < (value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 1) and benign_index > 0:
    mal_index += 1
    benign_index -= 1
    curr_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - value_x_y_yp_mal_part_sorted_by_yp[mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
    if curr_alpha_minus_beta > global_max_alpha_minus_beta:
        global_max_alpha_minus_beta = curr_alpha_minus_beta
        global_max_mal_index = mal_index
        global_max_benign_index = benign_index
alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1]
beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]

print("terminal stage is: {0}".format(int(data_size * (1 - outlier_rate))))
print('-----stage: ' + str(m+1) + '-----')
print('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))

last_alpha = alpha
last_beta = beta

# concat_x_training_data = np.concatenate((x_training_data_mal_part, x_training_data_benign_part), axis=0)
# concat_y_training_data = np.concatenate((y_training_data_mal_part, y_training_data_benign_part), axis=0)
# concat_x_and_y = np.concatenate((concat_x_training_data, concat_y_training_data), axis=1)

for n in range(m+3, int(data_size * (1 - outlier_rate) + 1)):
    print('-----stage: ' + str(n-1) + '-----')
    training_process_log.writelines('-----stage: ' + str(n-1) + '-----' + "\n")

    # pick n most fit data(每個stage都要維持alpha-beta的最大值)
    predict_y_mal_part = sess.run([output_layer], {x_placeholder: x_training_data_mal_part, y_placeholder: y_training_data_mal_part})[0]
    predict_y_benign_part = sess.run([output_layer], {x_placeholder: x_training_data_benign_part, y_placeholder: y_training_data_benign_part})[0]
    min_mal_predict_value = min(predict_y_mal_part)
    max_benign_predict_value = max(predict_y_benign_part)
    # print("max class 1: {0}    min class 2: {1}".format(max_benign_predict_value, min_mal_predict_value))
    close_to_most_mal_value_mal_part = predict_y_mal_part - min_mal_predict_value
    close_to_most_benign_value_benign_part = max_benign_predict_value - predict_y_benign_part

    x_y_mal_part = np.concatenate((x_training_data_mal_part, y_training_data_mal_part), axis=1)
    x_y_yp_mal_part = np.concatenate((x_y_mal_part, predict_y_mal_part), axis=1)
    value_x_y_yp_mal_part = np.concatenate((close_to_most_mal_value_mal_part, x_y_yp_mal_part), axis=1)
    value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1]-1])]

    x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
    x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
    value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
    value_x_y_yp_benign_part_sorted_by_yp = np.flip(value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, value_x_y_yp_benign_part.shape[1]-1])], axis=0)

    # global max (n-1) alpha - beta
    # n 必定大於2
    if (n - 2 - 1) < value_x_y_yp_benign_part_sorted_by_yp.shape[0]:
        mal_index = 0
        benign_index = n - 2 - 1
        global_max_mal_index = mal_index
        global_max_benign_index = benign_index
    else:
        mal_index = (n - 2 - 1) - (value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1)
        benign_index = value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1
        global_max_mal_index = mal_index
        global_max_benign_index = benign_index
    global_max_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][
                                      value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - \
                                  value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][
                                      value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
    while mal_index < (n - 2 - 1) and mal_index < (value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 1) and benign_index > 0:
        mal_index += 1
        benign_index -= 1
        # print(mal_index)
        # print(benign_index)
        curr_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[benign_index][
                                    value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - \
                                value_x_y_yp_mal_part_sorted_by_yp[mal_index][
                                    value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
        if curr_alpha_minus_beta > global_max_alpha_minus_beta:
            global_max_alpha_minus_beta = curr_alpha_minus_beta
            global_max_mal_index = mal_index
            global_max_benign_index = benign_index

    yp_index = value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1
    last_alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
    last_beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
    if global_max_benign_index > value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 2:
        # 沒benign挑了
        last_pick_is_mal = True
        global_max_mal_index += 1
        alpha = last_alpha
        beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
    elif global_max_mal_index > value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 2:
        # 沒mal挑了
        last_pick_is_mal = False
        global_max_benign_index += 1
        alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
        beta = last_beta
    else:
        # 比較哪個會讓alpha-beta最大則挑哪個
        pick_benign_alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index + 1][yp_index]
        pick_mal_beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index + 1][yp_index]
        if (pick_benign_alpha - last_beta) > (last_alpha - pick_mal_beta):
            last_pick_is_mal = False
            global_max_benign_index += 1
            alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
            beta = last_beta
        else:
            last_pick_is_mal = True
            global_max_mal_index += 1
            alpha = last_alpha
            beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
    print("global max benign index: {0}, global max mal index: {1}".format(global_max_benign_index, global_max_mal_index))
    if alpha == last_alpha:
        last_pick_is_mal = True
    else:
        last_pick_is_mal = False
    if last_pick_is_mal:
        c_d_m = value_x_y_yp_mal_part_sorted_by_yp[:global_max_mal_index]
        c_d_b = value_x_y_yp_benign_part_sorted_by_yp[:global_max_benign_index+1]
        c_d_last = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index].reshape(1, -1)
        temp_c_s_d = np.concatenate((c_d_m, c_d_b), axis=0)
        current_stage_data = np.concatenate((temp_c_s_d, c_d_last), axis=0)
    else:
        c_d_m = value_x_y_yp_mal_part_sorted_by_yp[:global_max_mal_index+1]
        c_d_b = value_x_y_yp_benign_part_sorted_by_yp[:global_max_benign_index]
        c_d_last = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index].reshape(1, -1)
        temp_c_s_d = np.concatenate((c_d_m, c_d_b), axis=0)
        current_stage_data = np.concatenate((temp_c_s_d, c_d_last), axis=0)

    current_stage_y_training_data = np.delete(np.delete(current_stage_data, slice(0, m+1), axis=1), 1, axis=1)  # 去除從0到m&m+2欄
    current_stage_y_predict = np.delete(current_stage_data, slice(0, m+2), axis=1)

    print('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))
    training_process_log.writelines('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))
    if alpha > beta:
        print('new training case can be classified without additional action.')
        training_process_log.writelines('new training case can be classified without additional action.' + "\n")

    else:
        print('new training case violate condition L, apply GradientDescent to change weights & thresholds.')
        training_process_log.writelines('new training case larger than epsilon, apply GradientDescent to change weights & thresholds.' + "\n")
        # BP
        print('start BP.')
        training_process_log.writelines('start BP.' + "\n")
        bp_failed = False

        current_stage_x_training_data = np.delete(current_stage_data, (0, m + 1, m + 2), axis=1)  # 去除0和m+1和m+2欄
        # 找出違反condition L的training case是哪一種，並給予分類邊界，設為其學習目標
        if last_pick_is_mal:
            current_stage_learning_target = np.concatenate((np.tile([min_mal_predict_value], global_max_mal_index).reshape(-1, 1), np.tile([max_benign_predict_value], global_max_benign_index + 2).reshape(-1, 1)), axis=0)
            current_stage_learning_target[n - 1] = last_beta
        else:
            current_stage_learning_target = np.concatenate((np.tile([min_mal_predict_value], global_max_mal_index+1).reshape(-1, 1), np.tile([max_benign_predict_value], global_max_benign_index+1).reshape(-1, 1)), axis=0)
            current_stage_learning_target[n - 1] = last_alpha

        saver.save(sess, r"{0}/model.ckpt".format(dir_path))
        last_alpha_minus_beta = -9999999
        for stage in range(every_stage_max_thinking_times):
            sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_learning_target})
            thinking_times_count += 1

            temp_predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_learning_target})[0]
            temp_beta = max(temp_predict_y[np.where(current_stage_y_training_data == -1)[0]])
            temp_alpha = min(temp_predict_y[np.where(current_stage_y_training_data == 1)[0]])
            # print(current_stage_y_training_data)
            # print(temp_predict_y)
            if stage % 500 == 0 and stage != 0:
                temp_alpha_minus_beta = temp_alpha - temp_beta
                print('temp alpha:' + str(temp_alpha) + '   temp beta:' + str(temp_beta))
                print(temp_alpha_minus_beta)
                if temp_alpha_minus_beta - last_alpha_minus_beta > 0.0:
                    last_alpha_minus_beta = temp_alpha_minus_beta
                else:
                    bp_failed = True
                    print('BP failed: after {0} times training, learning too slow.'.format(
                        (stage + 1)))
                    training_process_log.writelines('BP failed: after {0} times training, learning change too slow.'.format((stage + 1)) + "\n")
                    # MUST restore before cramming(因為調權重可能會讓先前的資料違反condition L)
                    saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
                    print('restore weights.')
                    training_process_log.writelines('restore weights.' + "\n")
                    break
            if temp_alpha > temp_beta:
                print('BP {0} times, all this stage training data can be separated, thinking success!!!'.format((stage+1)))
                training_process_log.writelines('BP {0} times, all this stage training data can be separated, thinking success!!!'.format((stage + 1)) + "\n")
                break
            else:
                if stage == (every_stage_max_thinking_times - 1):
                    bp_failed = True
                    print('BP failed: after {0} times training, all this stage training data cannot be separated.'.format((stage + 1)))
                    training_process_log.writelines('BP failed: after {0} times training, all this stage training data cannot be separated.'.format((stage + 1)) + "\n")
                    # MUST restore before cramming(因為調權重可能會讓先前的資料違反condition L)
                    saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
                    print('restore weights.')
                    training_process_log.writelines('restore weights.' + "\n")

        if bp_failed:
            # add two hidden nodes to make the new training case square residual less than tolerance
            print('add two hidden nodes')
            training_process_log.writelines('add two hidden nodes.' + "\n")
            cramming_times_count += 1
            hidden_node_amount += 2
            # calculate relevant parameters
            # 取得現有的weight&threshold陣列
            current_hidden_weights, current_hidden_thresholds, current_output_weights, current_output_threshold = sess.run([hidden_weights, hidden_thresholds, output_weights, output_threshold], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_learning_target})

            x_c = current_stage_x_training_data[:n - 1]
            x_k = current_stage_x_training_data[n - 1:]
            y_k = current_stage_learning_target[n - 1:]
            # print(x_c.shape)
            # print(x_k.shape)
            print(x_k)
            print(y_k)
            # input(1)

            # calculate new hidden weight
            alpha_success = False
            while not alpha_success:
                beta_k = np.random.random_sample((1, m)) + 1
                if sess.run([Cal_table2], {beta_k_placeholder: beta_k, x_c_placeholder: x_c, x_k_placeholder: x_k})[0] != 0:
                    alpha_success = True

            current_stage_alpha_T = sess.run([alpha_T], {beta_k_placeholder: beta_k})[0]

            new_hidden_node_1_neuron_weights = Lambda * current_stage_alpha_T
            new_hidden_node_2_neuron_weights = -Lambda * current_stage_alpha_T

            # calculate new hidden threshold
            new_hidden_node_1_threshold = sess.run([calculate_new_hidden_node_1_threshold], {x_k_placeholder: x_k, alpha_T_placeholder: current_stage_alpha_T})[0].reshape(1)
            new_hidden_node_2_threshold = sess.run([calculate_new_hidden_node_2_threshold], {x_k_placeholder: x_k, alpha_T_placeholder: current_stage_alpha_T})[0].reshape(1)

            # calculate new output weight
            y_k_output = sess.run([output_layer], {x_placeholder: x_k, y_placeholder: y_k})
            y_k_minus_output = y_k - y_k_output
            new_output_weight = sess.run([calculate_new_output_weight], {y_k_minus_output_placeholder: y_k_minus_output})[0].reshape(1, 1)

            # combine weights & thresholds
            new_hidden_weights_temp = np.append(current_hidden_weights, new_hidden_node_1_neuron_weights.reshape(input_node_amount, 1), axis=1)
            new_hidden_weights = np.append(new_hidden_weights_temp, new_hidden_node_2_neuron_weights.reshape(input_node_amount, 1), axis=1)
            # print(new_hidden_weights)
            new_hidden_thresholds_temp = np.append(current_hidden_thresholds, new_hidden_node_1_threshold)
            new_hidden_thresholds = np.append(new_hidden_thresholds_temp, new_hidden_node_2_threshold)
            # print(new_hidden_thresholds)
            new_output_weights_temp = np.append(current_output_weights, new_output_weight)
            new_output_weights = np.append(new_output_weights_temp, new_output_weight).reshape(hidden_node_amount, 1)
            # print(current_output_weights)
            # print(new_output_weights)

            # create new graph & session
            with tf.Graph().as_default():  # Create a new graph, and make it the default.
                with tf.name_scope('inputs'):
                    # placeholders
                    x_placeholder = tf.placeholder(tf.float64, name='x_input')
                    y_placeholder = tf.placeholder(tf.float64, name='y_input')

                with tf.name_scope('hidden_layer'):
                    hidden_thresholds = tf.Variable(new_hidden_thresholds, name='hidden_threshold')
                    hidden_weights = tf.Variable(new_hidden_weights, name='hidden_weight')
                    hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))

                with tf.name_scope('output_layer'):
                    # network architecture
                    output_threshold = tf.Variable(current_output_threshold, name='output_threshold')
                    output_weights = tf.Variable(new_output_weights, name='output_weight')
                    output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

                # learning goal & optimizer
                with tf.name_scope('loss'):
                    average_square_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
                with tf.name_scope('train'):
                    train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_square_residual)

                # saver
                saver = tf.train.Saver()

                with tf.name_scope('calculate_alpha_T'):
                    # alpha_T的算法，會用在加hidden node的時候
                    beta_k_placeholder = tf.placeholder(tf.float64, name='beta_k')
                    x_c_placeholder = tf.placeholder(tf.float64, name='x_c')
                    x_k_placeholder = tf.placeholder(tf.float64, name='x_k')
                    test = tf.sqrt(tf.reduce_sum(tf.square(beta_k_placeholder, name='square'), name='reduce_sum'), name='sqrt')
                    alpha = tf.div(beta_k_placeholder, test, name='alpha')
                    alpha_T = tf.transpose(alpha, name='transpose')
                    Cal_table2 = tf.reduce_sum(tf.matmul(tf.subtract(x_c_placeholder, x_k_placeholder), alpha_T), name='test_alpha')

                with tf.name_scope('calculate_new_hidden_threshold'):
                    # new hidden node threshold的算法
                    alpha_T_placeholder = tf.placeholder(tf.float64, name='alpha_T')
                    with tf.name_scope('hidden_threshold_1'):
                        calculate_new_hidden_node_1_threshold = zeta - Lambda * tf.matmul(x_k_placeholder,
                                                                                          alpha_T_placeholder)
                    with tf.name_scope('hidden_threshold_2'):
                        calculate_new_hidden_node_2_threshold = zeta + Lambda * tf.matmul(x_k_placeholder,
                                                                                          alpha_T_placeholder)

                with tf.name_scope('calculate_new_output_weight'):
                    # calculate new output weight
                    y_k_minus_output_placeholder = tf.placeholder(tf.float64, name='y_k_minus_output')
                    two = tf.cast(2.0, tf.float64, name='2')
                    calculate_new_output_weight = y_k_minus_output_placeholder / (two * tf.cast(tf.tanh(zeta), tf.float64))

                init = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(init)

                # try if cramming success
                # pick n most fit data(每挑一個都要維持alpha-beta的最大值)
                predict_y_mal_part = sess.run([output_layer], {x_placeholder: x_training_data_mal_part,
                                                               y_placeholder: y_training_data_mal_part})[0]
                predict_y_benign_part = sess.run([output_layer], {x_placeholder: x_training_data_benign_part,
                                                                  y_placeholder: y_training_data_benign_part})[0]
                min_mal_predict_value = min(predict_y_mal_part)
                max_benign_predict_value = max(predict_y_benign_part)
                close_to_most_mal_value_mal_part = predict_y_mal_part - min_mal_predict_value
                close_to_most_benign_value_benign_part = max_benign_predict_value - predict_y_benign_part

                x_y_mal_part = np.concatenate((x_training_data_mal_part, y_training_data_mal_part), axis=1)
                x_y_yp_mal_part = np.concatenate((x_y_mal_part, predict_y_mal_part), axis=1)
                value_x_y_yp_mal_part = np.concatenate((close_to_most_mal_value_mal_part, x_y_yp_mal_part), axis=1)
                value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1] - 1])]

                x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
                x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
                value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
                value_x_y_yp_benign_part_sorted_by_yp = np.flip(value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, value_x_y_yp_benign_part.shape[1] - 1])], axis=0)

                # global max (n-1) alpha - beta
                # n 必定大於2
                if (n - 2 - 1) < value_x_y_yp_benign_part_sorted_by_yp.shape[0]:
                    mal_index = 0
                    benign_index = n - 2 - 1
                    global_max_mal_index = mal_index
                    global_max_benign_index = benign_index
                else:
                    mal_index = (n - 2 - 1) - (value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1)
                    benign_index = value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1
                    global_max_mal_index = mal_index
                    global_max_benign_index = benign_index
                global_max_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][
                                                  value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - \
                                              value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][
                                                  value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
                while mal_index < (n - 2 - 1) and mal_index < (
                            value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 1) and benign_index > 0:
                    mal_index += 1
                    benign_index -= 1
                    # print(mal_index)
                    # print(benign_index)
                    curr_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[benign_index][
                                                value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - \
                                            value_x_y_yp_mal_part_sorted_by_yp[mal_index][
                                                value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
                    if curr_alpha_minus_beta > global_max_alpha_minus_beta:
                        global_max_alpha_minus_beta = curr_alpha_minus_beta
                        global_max_mal_index = mal_index
                        global_max_benign_index = benign_index

                yp_index = value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1
                last_alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
                last_beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
                if global_max_benign_index > value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 2:
                    # 沒benign挑了
                    last_pick_is_mal = True
                    global_max_mal_index += 1
                    alpha = last_alpha
                    beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
                elif global_max_mal_index > value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 2:
                    # 沒mal挑了
                    last_pick_is_mal = False
                    global_max_benign_index += 1
                    alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
                    beta = last_beta
                else:
                    # 比較哪個會讓alpha-beta最大則挑哪個
                    pick_benign_alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index + 1][yp_index]
                    pick_mal_beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index + 1][yp_index]
                    if (pick_benign_alpha - last_beta) > (last_alpha - pick_mal_beta):
                        last_pick_is_mal = False
                        global_max_benign_index += 1
                        alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
                        beta = last_beta
                    else:
                        last_pick_is_mal = True
                        global_max_mal_index += 1
                        alpha = last_alpha
                        beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
                print("after cramming:")
                print('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))
                if alpha > beta:
                    print('after add two hidden node, new training case is satisfy condition L')
                else:
                    print('Warning! After cramming, new training case is still violate condition L')
                    input(1)

# close the recording file of training process
training_process_log.close()

# tf.train.SummaryWriter soon be deprecated, use following
# writer = tf.summary.FileWriter("C:/logfile", sess.graph)
# writer.close()

# train end, get NN status
curr_hidden_neuron_weight, curr_hidden_threshold, curr_output_neuron_weight, curr_output_threshold, curr_average_loss, curr_output = sess.run(
            [hidden_weights, hidden_thresholds,
             output_weights, output_threshold, average_square_residual,
             output_layer],
            {x_placeholder: x_training_data, y_placeholder: y_training_data})


# pick n most fit data(每個stage都要維持alpha-beta的最大值)
predict_y_mal_part = sess.run([output_layer], {x_placeholder: x_training_data_mal_part, y_placeholder: y_training_data_mal_part})[0]
predict_y_benign_part = sess.run([output_layer], {x_placeholder: x_training_data_benign_part, y_placeholder: y_training_data_benign_part})[0]
min_mal_predict_value = min(predict_y_mal_part)
max_benign_predict_value = max(predict_y_benign_part)
# print("max class 1: {0}    min class 2: {1}".format(max_benign_predict_value, min_mal_predict_value))
close_to_most_mal_value_mal_part = predict_y_mal_part - min_mal_predict_value
close_to_most_benign_value_benign_part = max_benign_predict_value - predict_y_benign_part

x_y_mal_part = np.concatenate((x_training_data_mal_part, y_training_data_mal_part), axis=1)
x_y_yp_mal_part = np.concatenate((x_y_mal_part, predict_y_mal_part), axis=1)
value_x_y_yp_mal_part = np.concatenate((close_to_most_mal_value_mal_part, x_y_yp_mal_part), axis=1)
value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1] - 1])]

x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
value_x_y_yp_benign_part_sorted_by_yp = np.flip(value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, value_x_y_yp_benign_part.shape[1] - 1])], axis=0)

# global max alpha - beta
# n 必定大於2
if (int(data_size * (1 - outlier_rate)) - 2) < value_x_y_yp_benign_part_sorted_by_yp.shape[0]:
    mal_index = 0
    benign_index = int(data_size * (1 - outlier_rate)) - 2
    global_max_mal_index = mal_index
    global_max_benign_index = benign_index
else:
    mal_index = (int(data_size * (1 - outlier_rate)) - 2) - (value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1)
    benign_index = value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1
    global_max_mal_index = mal_index
    global_max_benign_index = benign_index
global_max_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
while mal_index < (int(data_size * (1 - outlier_rate)) - 2) and mal_index < (value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 1) and benign_index > 0:
    mal_index += 1
    benign_index -= 1
    curr_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - value_x_y_yp_mal_part_sorted_by_yp[mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
    if curr_alpha_minus_beta > global_max_alpha_minus_beta:
        global_max_alpha_minus_beta = curr_alpha_minus_beta
        global_max_mal_index = mal_index
        global_max_benign_index = benign_index
alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1]
beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]

mal_data_sorted_majority_part = value_x_y_yp_mal_part_sorted_by_yp[:global_max_mal_index + 1]
mal_data_sorted_outlier_part = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index + 1:]
benign_data_sorted_majority_part = value_x_y_yp_benign_part_sorted_by_yp[:global_max_benign_index + 1]
benign_data_sorted_outlier_part = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index + 1:]
majority_data = np.concatenate((mal_data_sorted_majority_part, benign_data_sorted_majority_part), axis=0)
outlier_data = np.concatenate((mal_data_sorted_outlier_part, benign_data_sorted_outlier_part), axis=0)
sort_result = np.concatenate((majority_data, outlier_data), axis=0)

np.savetxt(new_path + r"\two_class_hidden_neuron_weight.txt", curr_hidden_neuron_weight)
np.savetxt(new_path + r"\two_class_hidden_threshold.txt", curr_hidden_threshold)
np.savetxt(new_path + r"\two_class_output_neuron_weight.txt", curr_output_neuron_weight)
np.savetxt(new_path + r"\two_class_output_threshold.txt", curr_output_threshold)
np.savetxt(new_path + r"\two_class_training_data_fit_x_y_yp.txt", sort_result)

file = open(new_path + r"\_two_class_training_detail.txt", 'w')
file.writelines("learning_rate: " + str(learning_rate_eta) + "\n")
file.writelines("input_node_amount: " + str(input_node_amount) + "\n")
file.writelines("hidden_node_amount: " + str(hidden_node_amount) + "\n")
file.writelines("output_node_amount: " + str(output_node_amount) + "\n")
file.writelines("training_data_amount: " + str(data_size) + "\n")
file.writelines("outlier_rate: " + str(outlier_rate*100) + "%\n")
file.writelines("average_loss_of_the_model: " + str(curr_average_loss) + "\n")
file.writelines("thinking_times_count: " + str(thinking_times_count) + "\n")
file.writelines("cramming_times_count: " + str(cramming_times_count) + "\n")
file.writelines("majority benign sample amount: " + str(global_max_benign_index+1) + "\n")
file.writelines("majority malicious sample amount: " + str(global_max_mal_index+1) + "\n")
file.writelines("alpha(min majority benign output): " + str(alpha) + "\n")
file.writelines("beta(max majority malicious output): " + str(beta) + "\n")
file.writelines("classify middle point((alpha+beta)/2): " + str((alpha+beta)/2) + "\n")
file.writelines("potential anomaly amount: {0}(Benign)    {1}(Malware)\n".format(benign_data_sorted_outlier_part.shape[0], mal_data_sorted_outlier_part.shape[0]))
file.writelines("total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
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
print("thinking times: %s" % thinking_times_count)
print("hidden node: %s nodes" % hidden_node_amount)
print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))

# analyze part
middle_point = (alpha+beta)/2
for i in range(2):
    if i == 0:
        file = open(new_path + r"\_training_data_analyze.txt", 'w')
    elif i == 1:
        file = open(new_path + r"\_testing_data_analyze.txt", 'w')
    for j in range(training_data_container.shape[0]):
        if i == 0:
            input_data = training_data_container[j]
        elif i == 1:
            input_data = testing_data_container[j]
        x_input_data = input_data[:, 1:]
        y_input_data = input_data[:, 0].reshape((-1, 1))
        predict_y = sess.run([output_layer], {x_placeholder: x_input_data})[0]
        if j == 0:
            classify_as_benign_count = predict_y[np.where(predict_y >= middle_point)[0]].shape[0]
            file.writelines("benign classification accuracy: {0}/{1}\n".format(classify_as_benign_count, x_input_data.shape[0]))
        else:
            classify_as_mal_count = predict_y[np.where(predict_y < middle_point)[0]].shape[0]
            file.writelines("rule {0} classification accuracy: {1}/{2}\n".format(j, classify_as_mal_count, x_input_data.shape[0]))

    file.close()

