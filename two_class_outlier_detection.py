# coding:utf-8
import tensorflow as tf
import numpy as np
import time
import random
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


# # read fake data from file(test data)
# data_amount = '100'
# file_input = "tensorflow_binary_input_" + data_amount
# file_output = "tensorflow_binary_output_" + data_amount
# file_name = file_input
# x_training_data = np.loadtxt(file_input + ".txt", dtype=float, delimiter=" ").reshape((-1, 1))
# y_training_data = np.loadtxt(file_output + ".txt", dtype=float, delimiter=" ").reshape((-1, 1))
# # print(x_training_data)
# # print(y_training_data)

# # read maochi data from file
# file_name = "traffic"
# data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=",")
# x_training_data = np.delete(data, 5, axis=1)
# y_training_data = np.delete(data, slice(0, 5), axis=1).reshape((-1, 1))
# # print(x_training_data)
# # print(y_training_data)

# # read ntu data from file
# rule = "10"
# ntu = "3"
# data_amount = "1250"
# # light = "" or "_light"
# light = ""
# file_input = "TensorFlow_input_detection_rule_"+rule+"_"+data_amount+"_and_ntu_"+ntu+"_benign_"+data_amount+light+"_no_label"
# file_output = "TensorFlow_output_for_" + data_amount
# # file_name = file_input
# x_training_data = np.loadtxt(file_input + ".txt", dtype=float, delimiter=" ")
# y_training_data = np.loadtxt(file_output + ".txt", dtype=float, delimiter=" ").reshape((-1, 1))
# # print(x_training_data)
# # print(y_training_data)

# # read ntu nominal data from file
# # file_input = r"nominal_data\ntu_rule_2_4_6_7_10_11_12_100_and_benign_100"
# # file_output = r"nominal_data\ntu_rule_2_4_6_7_10_11_12_100_and_benign_100_output"
# # file_input = r"nominal_data\ntu_rule_15_16_19_22_100_and_benign_100"
# # file_output = r"nominal_data\ntu_rule_15_16_19_22_100_and_benign_100_output"
# # file_input = r"nominal_data\ntu_rule_2_4_6_7_10_11_12_100_and_benign_700"
# # file_output = r"nominal_data\ntu_rule_2_4_6_7_10_11_12_100_and_benign_700_output"
# file_input = r"nominal_data\ntu_rule_15_16_19_22_100_and_benign_400"
# file_output = r"nominal_data\ntu_rule_15_16_19_22_100_and_benign_400_output"
# # file_name = file_input
# x_training_data = np.loadtxt(file_input + ".txt", dtype=str, delimiter=" ")
# y_training_data = np.loadtxt(file_output + ".txt", dtype=float, delimiter=" ").reshape((-1, 1))
# # delete timestamp & convert type
# x_training_data = x_training_data[:, :x_training_data.shape[1]-1]
# x_training_data = np.ndarray.astype(x_training_data, float)
# # print(x_training_data)
# # print(y_training_data)

# 取100個sample 19種 要用時全部底下往右縮排*2
# for owl in range(1, 20):
#     with tf.Graph().as_default():
#         # read owl data from file
#         owl_rule = str(owl)
#         sample_amount_arr = ['65', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '39', '100', '40', '100', '100', '100', '100']
#         sample_amount = sample_amount_arr[owl-1]
#         file_name = r"19_owl_rules\owl_rule_"+owl_rule+"_"+sample_amount+"_and_benign_"+sample_amount
#         file_input = file_name
#         training_data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=" ")
#         x_training_data = training_data[:, 1:]
#         y_training_data = training_data[:, 0].reshape((-1, 1))
#         # x_training_data_mal_part = x_training_data[np.where(y_training_data == -1)[0]]
#         # x_training_data_benign_part = x_training_data[np.where(y_training_data == 1)[0]]
#         # y_training_data_mal_part = y_training_data[np.where(y_training_data == -1)[0]]
#         # y_training_data_benign_part = y_training_data[np.where(y_training_data == 1)[0]]
#         # print(x_training_data)
#         # print(y_training_data)
#
#         # # read owl data from file
#         # file_name = 'simulation_data_1'
#         # file_input = file_name
#         # training_data = np.loadtxt(file_name + ".csv", dtype=float, delimiter=",")
#         # x_training_data = training_data[:, 0].reshape((-1, 1))
#         # y_training_data = training_data[:, 1].reshape((-1, 1))
#         # # print(x_training_data)
#         # # print(y_training_data)

# 所有sample 19種 要用時全部底下往右縮排*2
for owl in range(1, 20):
    with tf.Graph().as_default():
        # read all owl data from file
        owl_rule = str(owl)
        file_name = r"19_owl_rules\owl_rule_" + owl_rule + "_all_training_data"
        file_input = file_name
        training_data = np.loadtxt(file_name + ".txt", dtype=str, delimiter=" ")
        training_data = training_data[:, :training_data.shape[1]-1]
        training_data = np.ndarray.astype(training_data, float)
        x_training_data = training_data[:, 1:]
        y_training_data = training_data[:, 0].reshape((-1, 1))
    # x_training_data_mal_part = x_training_data[np.where(y_training_data == -1)[0]]
    # x_training_data_benign_part = x_training_data[np.where(y_training_data == 1)[0]]
    # y_training_data_mal_part = y_training_data[np.where(y_training_data == -1)[0]]
    # y_training_data_benign_part = y_training_data[np.where(y_training_data == 1)[0]]
    # print(x_training_data)
    # print(y_training_data)

# # 所有不重複分兩類
# for owl in range(1, 20):
#     # read all owl data from file
#     owl_rule = str(owl)
#     file_name = r"19_owl_rules\owl_rule_" + owl_rule + "_all_training_data"
#
#     training_data = np.loadtxt(file_name + ".txt", dtype=str, delimiter=" ")
#     training_data = training_data[:, :training_data.shape[1] - 1]
#     training_data = np.ndarray.astype(training_data, float)
#     if owl == 1:
#         all_training_data = training_data
#     else:
#         all_training_data = np.concatenate((all_training_data, training_data), axis=0)
# # np.savetxt("3333333333.txt", all_training_data)
# # input(123)
# all_training_data = np.unique(all_training_data, axis=0)
# x_training_data = all_training_data[:, 1:]
# y_training_data = all_training_data[:, 0].reshape((-1, 1))
# print(x_training_data.shape)
# print(y_training_data.shape)
# print(np.where(y_training_data == 1)[0].shape)
# # input(123)
# x_training_data_mal_part = x_training_data[np.where(y_training_data == -1)[0]]
# x_training_data_benign_part = x_training_data[np.where(y_training_data == 1)[0]]
# y_training_data_mal_part = y_training_data[np.where(y_training_data == -1)[0]]
# y_training_data_benign_part = y_training_data[np.where(y_training_data == 1)[0]]
# file_input = "all_rules_data"

# # 所有不重複中抽10%分兩類
# sample_data_container = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=object)
# for owl in range(0, 20):
#     # read all owl data from file
#     if owl == 19:
#         file_name = r"19_owl_rules\no_duplicate_pattern_19_rules\benign_chrome_and_filezilla_no_duplicate_label_1_at_index_0"
#         training_data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=" ")
#         sample_quantity = 0
#         for r in range(1, sample_data_container.shape[0]):
#             sample_quantity += sample_data_container[r].shape[0]
#         # sample_quantity = 2164  # 1/10 sampling of malware samples
#         sample_data = training_data[:sample_quantity]
#         sample_data_container[0] = sample_data
#     else:
#         owl_rule = str(owl+1)
#         file_name = r"19_owl_rules\no_duplicate_pattern_19_rules\rule_{0}_no_duplicate_with_label_-1_at_index_0".format(owl_rule)
#         training_data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=" ")
#         sample_data = training_data[:int(training_data.shape[0]/10)]
#         sample_data_container[owl+1] = sample_data
#     # print(int(training_data.shape[0]/10))
#     # sss += int(training_data.shape[0]/10)
#     # sample_data_container[owl] = training_data
# for i in range(1, 20):
#     if i == 1:
#         heads = sample_data_container[i][:2]
#         remains = sample_data_container[i][2:]
#     else:
#         heads = np.concatenate((heads, sample_data_container[i][:2]), axis=0)
#         remains = np.concatenate((remains, sample_data_container[i][2:]), axis=0)
#
# h_b = np.concatenate((heads, sample_data_container[0]), axis=0)
# all_training_data = np.concatenate((h_b, remains), axis=0)
# # input(123)
# x_training_data = all_training_data[:, 1:]
# y_training_data = all_training_data[:, 0].reshape((-1, 1))
# x_training_data_mal_part = x_training_data[np.where(y_training_data == -1)[0]]
# x_training_data_benign_part = x_training_data[np.where(y_training_data == 1)[0]]
# y_training_data_mal_part = y_training_data[np.where(y_training_data == -1)[0]]
# y_training_data_benign_part = y_training_data[np.where(y_training_data == 1)[0]]
# file_input = "all_rules_data_sample_1_of_10"

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
        every_stage_max_thinking_times = 10000000
        data_size = x_training_data.shape[0]
        outlier_rate = 0.05
        # square_residual_tolerance = 0.5
        zeta = 0.05
        Lambda = 100000

        # file_name = file_input + "_sigma_2"
        # file_name = file_input + "_sigma_" + str(sigma_multiplier) + "_wrong_10_label"
        file_name = file_input + "_bml"

        # create folder to save training process
        new_path = r"{0}/".format(dir_path) + file_name
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

        # with tf.name_scope('calculate_envelope_width'):
        #     # 算出envelope width: epsilon
        #     opt = sess.run(tf.matrix_solve_ls(x_training_data, y_training_data, fast=False, name='solve_matrix'))
        #     # 算出所有資料用此模型得到的輸出值
        #     opt_output = sess.run(tf.matmul(x_training_data, opt, name='matmul'))
        #     # 輸出值減掉實際的y值後取絕對值，得到此模型的差的矩陣
        #     opt_distance = sess.run(tf.abs(opt_output - y_training_data, name='abs'))
        #     # 取得差矩陣的平均值(用不到)以及變異數
        #     mean, var = tf.nn.moments(tf.stack(opt_distance, name='stack'), axes=[0], name='get_var')
        #     # stander deviation(全體資料的線性迴歸的標準差)
        #     sigma = sess.run(tf.sqrt(var, name='sqrt'))
        #     # envelope width(可以調整幾倍的標準差e.g. 2*sigma是95%的資料)
        #     epsilon = sigma_multiplier * sigma

        with tf.name_scope('calculate_first_slfn_weights'):
            # 首先架構初始SLFN
            m = input_node_amount
            # 第一次取m+1筆資料算出first SLFN 的初始權重，算法是做矩陣列運算解聯立方程式，讓前m+1筆資料都可以完美符合這個模型
            # m+1筆資料(x,y)，m+1個變數(m個weight，1個hidden threshold)，解聯立方程式，得到正確答案
            # 取前m+1筆y資料，並且套公式給定output weight和threshold
            # 不知道為什麼output weight & threshold要這樣給，給weight 1 ,threshold 0不好嗎? 可問蔡老師
            first_slfn_output_weight = (np.max(y_training_data) - np.min(y_training_data) + 2.0).reshape(1, 1)
            first_slfn_output_threshold = (np.min(y_training_data) - 1.0).reshape(1)
            # print(first_slfn_output_weight)
            # print(first_slfn_output_threshold)
            desi_slice_y = y_training_data[:m+1]
            # print(desi_slice_y.shape)
            # for i in range(desi_slice_y.shape[0]):
            #     print(desi_slice_y[i])
            # input(123)
            # 取得x經過運算後應該得到的hidden value(做tanh運算之前)
            yc = np.arctanh((desi_slice_y - first_slfn_output_threshold) / first_slfn_output_weight).reshape(m+1, 1)
            # print(yc.shape)
            # 對應給定的output weight & threshold，解hidden weight & threshold的聯立方程式
            desi_slice_x = x_training_data[:m+1]
            # 由於x原本只有m維，所以要加上1倍的threshold來變成m+1個變數，m+1筆資料，解方程式
            hidden_node_threshold_vector = tf.ones([m + 1, 1], dtype=tf.float64, name='one')
            xc = sess.run(tf.concat(axis=1, values=[desi_slice_x, hidden_node_threshold_vector]))
            # print(xc.shape)
            # 使用tf.matrix_solve_ls做矩陣運算解聯立方程式得到hidden weight & threshold
            answer = sess.run(tf.matrix_solve_ls(xc, yc, fast=False))
            # answer的前m個是hidden weight 最後一個是hidden threshold
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

        # 要用到pruning再說
        # tool_graph = tf.Graph()
        # with tool_graph.as_default():
        #     tool_alpha = tf.placeholder(tf.float64)
        #     tool_beta = tf.placeholder(tf.float64)
        #     min_alpha = tf.reduce_min(tool_alpha)
        #     max_beta = tf.reduce_max(tool_beta)
        #     tool_init = tf.global_variables_initializer()
        # tool_sess = tf.Session(graph=tool_graph)
        # tool_sess.run([tool_init])

        # 如果想看所有default graph裡面的node 可以用下面這段code
        # for node in tf.get_default_graph().as_graph_def().node:
        #     print(node.name)

        # print(predict_y)
        # input('123')

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
        # n 必定大於2 這裡確定m+1筆不會違反condition L 所以可以直接取最大
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

        print('-----stage: ' + str(m+1) + '-----')
        print('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))

        last_alpha = alpha
        last_beta = beta

        concat_x_training_data = np.concatenate((x_training_data_mal_part, x_training_data_benign_part), axis=0)
        concat_y_training_data = np.concatenate((y_training_data_mal_part, y_training_data_benign_part), axis=0)
        concat_x_and_y = np.concatenate((concat_x_training_data, concat_y_training_data), axis=1)

        for n in range(m+2, int(data_size * (1 - outlier_rate) + 1)):
            # for n in range(4221, int(data_size * (1 - outlier_rate) + 1)):
            print('-----stage: ' + str(n) + '-----')
            training_process_log.writelines('-----stage: ' + str(n) + '-----' + "\n")

            # # pick n most fit data(distance) 作廢->改成維持local兩類距離最大值
            # predict_y_mal_part = sess.run([output_layer], {x_placeholder: x_training_data_mal_part, y_placeholder: y_training_data_mal_part})[0]
            # predict_y_benign_part = sess.run([output_layer], {x_placeholder: x_training_data_benign_part, y_placeholder: y_training_data_benign_part})[0]
            # most_fit_mal_value = min(predict_y_mal_part)
            # most_fit_benign_value = max(predict_y_benign_part)
            # fit_value_mal_part = most_fit_benign_value - predict_y_mal_part
            # fit_value_benign_part = predict_y_benign_part - most_fit_mal_value
            #
            # concat_predict_y = np.concatenate((predict_y_mal_part, predict_y_benign_part), axis=0)
            # concat_fit_value = np.concatenate((fit_value_mal_part, fit_value_benign_part), axis=0)
            #
            # concat_xy_and_y_predict = np.concatenate((concat_x_and_y, concat_predict_y), axis=1)
            # concat_fit_and_x_y_yp = np.concatenate((concat_fit_value, concat_xy_and_y_predict), axis=1)
            # # sort fit value from big to small
            # sort_result = np.flip(concat_fit_and_x_y_yp[np.argsort(concat_fit_and_x_y_yp[:, 0])], axis=0)
            # # np.savetxt('sort_result.txt', sort_result)
            # # take first n row of data,
            # current_stage_data = sort_result[:n]
            #
            # current_stage_y_training_data = np.delete(np.delete(current_stage_data, slice(0, m+1), axis=1), 1, axis=1)  # 去除從0到m&m+2欄
            # current_stage_y_predict = np.delete(current_stage_data, slice(0, m+2), axis=1)
            # # print(current_stage_y_training_data)
            # # print(current_stage_y_predict)
            #
            # alpha = min(current_stage_y_predict[np.where(current_stage_y_training_data == 1)[0]])[0]
            # beta = max(current_stage_y_predict[np.where(current_stage_y_training_data == -1)[0]])[0]
            # # print(alpha)
            # # print(beta)
            # print('alpha:' + str(alpha) + '   beta:' + str(beta))

            # pick n most fit data(每個stage都要維持n-1筆資料alpha-beta的最大值)
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
            # value_x_y_yp_mal_part_sorted_by_value = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, 0])]
            value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1]-1])]

            x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
            x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
            value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
            # value_x_y_yp_benign_part_sorted_by_value = value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, 0])]
            value_x_y_yp_benign_part_sorted_by_yp = np.flip(value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, value_x_y_yp_benign_part.shape[1]-1])], axis=0)

            # global max (n-1) alpha - beta
            # n 必定大於2
            if (n-2-1) < value_x_y_yp_benign_part_sorted_by_yp.shape[0]:
                mal_index = 0
                benign_index = n - 2-1
                global_max_mal_index = mal_index
                global_max_benign_index = benign_index
            else:
                mal_index = (n - 2-1) - (value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1)
                benign_index = value_x_y_yp_benign_part_sorted_by_yp.shape[0] - 1
                global_max_mal_index = mal_index
                global_max_benign_index = benign_index
            global_max_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1]-1] - value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1]-1]
            while mal_index < (n-2-1) and mal_index < (value_x_y_yp_mal_part_sorted_by_yp.shape[0]-1) and benign_index > 0:
                mal_index += 1
                benign_index -= 1
                # print(mal_index)
                # print(benign_index)
                curr_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[benign_index][value_x_y_yp_benign_part_sorted_by_yp.shape[1]-1] - value_x_y_yp_mal_part_sorted_by_yp[mal_index][value_x_y_yp_mal_part_sorted_by_yp.shape[1]-1]
                if curr_alpha_minus_beta > global_max_alpha_minus_beta:
                    global_max_alpha_minus_beta = curr_alpha_minus_beta
                    global_max_mal_index = mal_index
                    global_max_benign_index = benign_index

            yp_index = value_x_y_yp_benign_part_sorted_by_yp.shape[1]-1
            last_alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
            last_beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
            if global_max_benign_index > value_x_y_yp_benign_part_sorted_by_yp.shape[0]-2:
                # 沒benign挑了
                last_pick_is_mal = True
                global_max_mal_index += 1
                alpha = last_alpha
                beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][yp_index]
            elif global_max_mal_index > value_x_y_yp_mal_part_sorted_by_yp.shape[0]-2:
                # 沒mal挑了
                last_pick_is_mal = False
                global_max_benign_index += 1
                alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][yp_index]
                beta = last_beta
            else:
                # 比較哪個會讓alpha-beta最大則挑哪個
                pick_benign_alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index+1][yp_index]
                pick_mal_beta = value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index+1][yp_index]
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
            # np.savetxt("_aaa_watch_current_stage_data.txt", current_stage_data)

            current_stage_y_training_data = np.delete(np.delete(current_stage_data, slice(0, m+1), axis=1), 1, axis=1)  # 去除從0到m&m+2欄
            current_stage_y_predict = np.delete(current_stage_data, slice(0, m+2), axis=1)
            # print(current_stage_y_training_data)
            # print(current_stage_y_predict)
            print('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))
            training_process_log.writelines('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))
            if alpha > beta:
                print('new training case can be classified without additional action.')
                training_process_log.writelines('new training case can be classified without additional action.' + "\n")

                # last_alpha = alpha
                # last_beta = beta
            else:
                print('new training case violate condition L, apply GradientDescent to change weights & thresholds.')
                training_process_log.writelines('new training case larger than epsilon, apply GradientDescent to change weights & thresholds.' + "\n")
                # BP
                print('start BP.')
                training_process_log.writelines('start BP.' + "\n")
                bp_failed = False

                current_stage_x_training_data = np.delete(current_stage_data, (0, m + 1, m + 2), axis=1)  # 去除0和m+1和m+2欄
                # print(current_stage_x_training_data)
                # 找出違反condition L的training case是哪一種，並給予分類邊界，設為其學習目標
                if last_pick_is_mal:
                    current_stage_learning_target = np.concatenate((np.tile([min_mal_predict_value], global_max_mal_index).reshape(-1, 1), np.tile([max_benign_predict_value], global_max_benign_index + 2).reshape(-1, 1)), axis=0)
                    current_stage_learning_target[n - 1] = last_beta
                else:
                    current_stage_learning_target = np.concatenate((np.tile([min_mal_predict_value], global_max_mal_index+1).reshape(-1, 1), np.tile([max_benign_predict_value], global_max_benign_index+1).reshape(-1, 1)), axis=0)
                    current_stage_learning_target[n - 1] = last_alpha
                # np.savetxt("___check.txt", current_stage_learning_target)
                # input(123)

                # if last_alpha == alpha:
                #     current_stage_learning_target[mal_index] = last_beta
                # else:
                #     current_stage_learning_target[n-1] = last_alpha

                saver.save(sess, r"{0}/model.ckpt".format(dir_path))
                last_alpha_minus_beta = -9999999
                for stage in range(every_stage_max_thinking_times):
                    sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_learning_target})
                    thinking_times_count += 1

                    temp_predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_learning_target})[0]
                    # print(temp_predict_y)
                    temp_beta = max(temp_predict_y[np.where(current_stage_y_training_data == -1)[0]])
                    temp_alpha = min(temp_predict_y[np.where(current_stage_y_training_data == 1)[0]])
                    # print('temp alpha:' + str(temp_alpha) + '   temp beta:' + str(temp_beta))
                    # input(123)
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
                        # last_alpha = temp_alpha
                        # last_beta = temp_beta
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
                    # predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data})
                    # print('current hidden weights:')
                    # print(current_hidden_weights)
                    # print('current hidden thresholds:')
                    # print(current_hidden_thresholds)
                    # print('current output weights:')
                    # print(current_output_weights)
                    # print('current output threshold:')
                    # print(current_output_threshold)

                    x_c = current_stage_x_training_data[:n - 1]
                    x_k = current_stage_x_training_data[n - 1:]
                    y_k = current_stage_learning_target[n - 1:]
                    # print(x_c.shape)
                    # print(x_k.shape)
                    # print(x_k)
                    # print(y_k)
                    # input(1)

                    # calculate new hidden weight
                    alpha_success = False
                    while not alpha_success:
                        beta_k = np.random.random_sample((1, m)) + 1
                        if sess.run([Cal_table2], {beta_k_placeholder: beta_k, x_c_placeholder: x_c, x_k_placeholder: x_k})[0] != 0:
                            alpha_success = True

                    current_stage_alpha_T = sess.run([alpha_T], {beta_k_placeholder: beta_k})[0]
                    # print(current_stage_alpha_T)
                    new_hidden_node_1_neuron_weights = Lambda * current_stage_alpha_T
                    new_hidden_node_2_neuron_weights = -Lambda * current_stage_alpha_T
                    # print('new hidden node 1 weights:')
                    # print(new_hidden_node_1_neuron_weights)
                    # print('new hidden node 2 weights:')
                    # print(new_hidden_node_2_neuron_weights)

                    # calculate new hidden threshold
                    new_hidden_node_1_threshold = sess.run([calculate_new_hidden_node_1_threshold], {x_k_placeholder: x_k, alpha_T_placeholder: current_stage_alpha_T})[0].reshape(1)
                    new_hidden_node_2_threshold = sess.run([calculate_new_hidden_node_2_threshold], {x_k_placeholder: x_k, alpha_T_placeholder: current_stage_alpha_T})[0].reshape(1)
                    # print('new hidden node 1 threshold:')
                    # print(new_hidden_node_1_threshold)
                    # print('new hidden node 2 threshold:')
                    # print(new_hidden_node_2_threshold)

                    # calculate new output weight
                    y_k_output = sess.run([output_layer], {x_placeholder: x_k, y_placeholder: y_k})
                    y_k_minus_output = y_k - y_k_output
                    new_output_weight = sess.run([calculate_new_output_weight], {y_k_minus_output_placeholder: y_k_minus_output})[0].reshape(1, 1)

                    # print('predict value of most recent training case: ' + str(predict_y[0][k - 1]))
                    # print('new output weight:')
                    # print(new_output_node_neuron_weight)

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
                        # value_x_y_yp_mal_part_sorted_by_value = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, 0])]
                        value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1] - 1])]

                        x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
                        x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
                        value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
                        # value_x_y_yp_benign_part_sorted_by_value = value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, 0])]
                        value_x_y_yp_benign_part_sorted_by_yp = np.flip(value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, value_x_y_yp_benign_part.shape[1] - 1])], axis=0)

                        # local max alpha - beta的作法,有bug因為n-1 stage BP成功後如果不挑global max會使得n筆資料當中可能有不只1筆違反condition L
                        # mal_index = 0
                        # benign_index = 0
                        # while (mal_index + benign_index) < (n - 2):
                        #     if mal_index == mal_sample_amount - 1:
                        #         benign_index += 1
                        #     elif benign_index == benign_sample_amount - 1:
                        #         mal_index += 1
                        #     else:
                        #         a = value_x_y_yp_benign_part_sorted_by_value[benign_index][
                        #             value_x_y_yp_benign_part_sorted_by_value.shape[1] - 1]
                        #         b = value_x_y_yp_mal_part_sorted_by_value[mal_index][
                        #             value_x_y_yp_mal_part_sorted_by_value.shape[1] - 1]
                        #         a_1 = value_x_y_yp_benign_part_sorted_by_value[benign_index + 1][
                        #             value_x_y_yp_benign_part_sorted_by_value.shape[1] - 1]
                        #         b_1 = value_x_y_yp_mal_part_sorted_by_value[mal_index + 1][
                        #             value_x_y_yp_mal_part_sorted_by_value.shape[1] - 1]
                        #         pick_mal_alpha_minus_beta_value = a - b_1
                        #         pick_benign_alpha_minus_beta_value = a_1 - b
                        #         if pick_mal_alpha_minus_beta_value > pick_benign_alpha_minus_beta_value:
                        #             mal_index += 1
                        #         else:
                        #             benign_index += 1
                        # current_stage_data = np.concatenate((value_x_y_yp_mal_part_sorted_by_value[:mal_index + 1],
                        #                                      value_x_y_yp_benign_part_sorted_by_value[:benign_index + 1]),
                        #                                     axis=0)

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
                        # current_stage_y_training_data = np.delete(np.delete(current_stage_data, slice(0, m + 1), axis=1), 1,
                        #                                           axis=1)  # 去除從0到m&m+2欄
                        # current_stage_y_predict = np.delete(current_stage_data, slice(0, m + 2), axis=1)
                        # # print(current_stage_y_training_data)
                        # # print(current_stage_y_predict)
                        #
                        # alpha = min(current_stage_y_predict[np.where(current_stage_y_training_data == 1)[0]])[0]
                        # beta = max(current_stage_y_predict[np.where(current_stage_y_training_data == -1)[0]])[0]
                        # print(alpha)
                        # print(beta)
                        print("after cramming:")
                        print('alpha:' + str(alpha) + '   beta:' + str(beta) + "   (alpha - beta):" + str(alpha-beta))
                        if alpha > beta:
                            print('after add two hidden node, new training case is satisfy condition L')
                        else:
                            print('Warning! After cramming, new training case is still violate condition L')
                            input(1)
                        # last_alpha = alpha
                        # last_beta = beta

                        # # verify add hidden node effect
                        # predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data,
                        #                                       y_placeholder: current_stage_y_training_data})[0]
                        # print('predict y:')
                        # print(predict_y)
                        # print('origin y:')
                        # print(current_stage_y_training_data)


                    #
                    #             # softening
                    #             # save variables
                    #             saver.save(sess, r"{0}/model.ckpt".format(dir_path))
                    #
                    #             # change tau value of newest hidden node
                    #             newest_hidden_node_tau_value = tau_in_each_hidden_node[hidden_node_amount - 1]
                    #             print(newest_hidden_node_tau_value)
                    #             while newest_hidden_node_tau_value > 1:
                    #                 newest_hidden_node_tau_value -= 1
                    #                 tau_in_each_hidden_node[hidden_node_amount - 1] = newest_hidden_node_tau_value
                    #                 print('tau array:')
                    #                 print(tau_in_each_hidden_node)
                    #                 softening_success = False
                    #
                    #                 for times in range(thinking_times):
                    #                     # forward pass
                    #                     predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data,
                    #                                                           y_placeholder: current_stage_y_training_data,
                    #                                                           tau_placeholder: tau_in_each_hidden_node})
                    #                     # print(predict_y)
                    #
                    #                     # check condition L
                    #                     class_1_output = [tf.double.max]
                    #                     class_2_output = [tf.double.min]
                    #                     it = np.nditer(predict_y, flags=['f_index'])
                    #                     while not it.finished:
                    #                         if current_stage_y_training_data[it.index] == 1:
                    #                             class_1_output.append(it[0])
                    #                         else:  # if current_stage_y_training_data[it.index] == -1:
                    #                             class_2_output.append(it[0])
                    #                         it.iternext()
                    #                     test_alpha, test_beta = tool_sess.run(
                    #                         [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
                    #
                    #                     # print(alpha)
                    #                     # print(beta)
                    #                     if test_alpha > test_beta:
                    #                         alpha = test_alpha
                    #                         beta = test_beta
                    #                         print(
                    #                             'softening success, gradient descent trained {0} times, #{1} tau value decrease by 1, current tau value: {2}'.format(
                    #                                 times, hidden_node_amount, newest_hidden_node_tau_value))
                    #                         softening_success = True
                    #                         saver.save(sess, r"{0}/model.ckpt".format(dir_path))
                    #                         break
                    #                     else:
                    #                         sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data,
                    #                                                    y_placeholder: current_stage_y_training_data,
                    #                                                    tau_placeholder: tau_in_each_hidden_node})
                    #                         softening_thinking_times_count += 1
                    #
                    #                 if not softening_success:
                    #                     print(
                    #                         'softening failed, gradient descent trained {0} times, restore #{1} tau value , restore tau value: {2}'.format(
                    #                             times, hidden_node_amount, newest_hidden_node_tau_value + 1))
                    #                     tau_in_each_hidden_node[hidden_node_amount - 1] = newest_hidden_node_tau_value + 1
                    #                     saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
                    #                     break
                    #
                    #             # PRUNING
                    #             if hidden_node_amount > 1:  # equals to the number of hidden nodes
                    #                 # get current weights & thresholds
                    #                 current_hidden_weights = hidden_weights.eval(sess)
                    #                 current_hidden_thresholds = hidden_thresholds.eval(sess)
                    #                 current_output_weights = output_weights.eval(sess)
                    #                 current_output_threshold = output_threshold.eval(sess)
                    #                 """
                    #                 print('#' * 10)
                    #                 print(current_hidden_weights)
                    #                 print(current_hidden_thresholds)
                    #                 print(current_output_weights)
                    #                 print(current_output_threshold)
                    #                 print('#' * 10)
                    #                 """
                    #
                    #                 # then try pruning from the begining hidden node
                    #                 for remove_index in range(hidden_node_amount):
                    #                     # 算出欲檢驗的結構之 weight 和 threshold
                    #                     exam_hidden_weights = np.concatenate(
                    #                         (
                    #                             current_hidden_weights[..., :remove_index],
                    #                             current_hidden_weights[..., remove_index + 1:]),
                    #                         axis=1)
                    #                     exam_hidden_thresholds = np.concatenate(
                    #                         (current_hidden_thresholds[:remove_index], current_hidden_thresholds[remove_index + 1:]),
                    #                         axis=0)
                    #                     exam_tau = np.delete(tau_in_each_hidden_node, remove_index)
                    #                     exam_output_weights = np.concatenate(
                    #                         (current_output_weights[:remove_index], current_output_weights[remove_index + 1:]), axis=0)
                    #
                    #                     # 建立測試 pruning 可行性的 Graph
                    #                     exam_graph = tf.Graph()
                    #                     with exam_graph.as_default():
                    #                         # placeholders
                    #                         exam_x_holder = tf.placeholder(tf.float64)
                    #                         exam_y_holder = tf.placeholder(tf.float64)
                    #                         exam_tau_holder = tf.placeholder(tf.float64)
                    #
                    #                         # exam variables
                    #                         exam_hidden_weights_var = tf.Variable(exam_hidden_weights)
                    #                         exam_hidden_thresholds_var = tf.Variable(exam_hidden_thresholds)
                    #                         exam_output_weights_var = tf.Variable(exam_output_weights)
                    #                         exam_output_threshold_var = tf.Variable(current_output_threshold)
                    #
                    #                         # exam tensors
                    #                         exam_hidden_layer_before_tanh = tf.add(tf.matmul(exam_x_holder, exam_hidden_weights_var),
                    #                                                                exam_hidden_thresholds_var)
                    #                         exam_hidden_layer = tf.tanh(
                    #                             tf.multiply(exam_hidden_layer_before_tanh,
                    #                                         tf.pow(tf.constant(2.0, dtype=tf.float64), exam_tau_holder)))
                    #                         exam_output_layer = tf.add(tf.matmul(exam_hidden_layer, exam_output_weights_var),
                    #                                                    exam_output_threshold_var)
                    #
                    #                         # exam goal & optimizer
                    #                         exam_average_square_residual = tf.reduce_mean(
                    #                             tf.reduce_sum(tf.square(exam_y_holder - exam_output_layer), reduction_indices=[1]))
                    #                         exam_train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(
                    #                             exam_average_square_residual)
                    #                         exam_init = tf.global_variables_initializer()
                    #
                    #                         # saver
                    #                         exam_saver = tf.train.Saver()
                    #
                    #                     # 使用 exam Session 執行 exam Graph
                    #                     exam_sess = tf.Session(graph=exam_graph)
                    #                     exam_sess.run(exam_init)
                    #                     exam_h_w_val, exam_h_t_val, exam_o_w_val, exam_o_t_val, exam_predict_y = exam_sess.run(
                    #                         [
                    #                             exam_hidden_weights_var,
                    #                             exam_hidden_thresholds_var,
                    #                             exam_output_weights_var,
                    #                             exam_output_threshold_var,
                    #                             exam_output_layer
                    #                         ],
                    #                         {
                    #                             exam_x_holder: current_stage_x_training_data,
                    #                             exam_y_holder: current_stage_y_training_data,
                    #                             exam_tau_holder: exam_tau
                    #                         }
                    #                     )
                    #
                    #                     # check if exam_alpha & exam_beta match condition L
                    #                     class_1_output = [tf.double.max]
                    #                     class_2_output = [tf.double.min]
                    #
                    #                     it = np.nditer(exam_predict_y, flags=['f_index'])
                    #                     while not it.finished:
                    #                         if current_stage_y_training_data[it.index] == 1:
                    #                             class_1_output.append(it[0])
                    #                         else:  # if current_stage_y_training_data[it.index] == -1:
                    #                             class_2_output.append(it[0])
                    #                         it.iternext()
                    #                     exam_alpha, exam_beta = tool_sess.run(
                    #                         [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
                    #                     # print('*' * 5 + "exam removing hidden node #{}".format(remove_index) + '*' * 5)
                    #                     # print("***** exam #{0} variables *****".format(remove_index))
                    #                     # print(exam_h_w_val)
                    #                     # print(exam_h_t_val)
                    #                     # print(exam_o_w_val)
                    #                     # print(exam_o_t_val)
                    #                     # print("***** exam #{0} predict_y *****".format(remove_index))
                    #                     # print(exam_predict_y)
                    #                     print("***** exam #{0} alpha & beta *****".format(remove_index))
                    #                     # print(list(zip(exam_predict_y, current_stage_y_training_data)))
                    #                     print('exam_alpha= ' + str(exam_alpha))
                    #                     print('exam_beta= ' + str(exam_beta))
                    #                     if exam_alpha <= exam_beta:
                    #                         print('pruning current hidden node #{0} will violate condition L'.format(remove_index))
                    #                         print('*' * 10)
                    #                         continue
                    #                     else:
                    #                         alpha = exam_alpha
                    #                         beta = exam_beta
                    #                         pruning_success_times_count += 1
                    #                         print("pruning current hidden node #{0} won't violate condition L".format(remove_index))
                    #                         print("!!!!! REMOVE hidden node #{0} !!!!!".format(remove_index))
                    #                         print('*' * 10)
                    #                         hidden_node_amount -= 1
                    #                         # 直接使用測試成功的 Session 和 Graph 取代舊的
                    #                         sess = exam_sess
                    #                         # 更換 placeholders 操作指標
                    #                         x_placeholder = exam_x_holder
                    #                         y_placeholder = exam_y_holder
                    #                         tau_placeholder = exam_tau_holder
                    #                         # 更換 variables 操作指標
                    #                         hidden_weights = exam_hidden_weights_var
                    #                         hidden_thresholds = exam_hidden_thresholds_var
                    #                         output_weights = exam_output_weights_var
                    #                         output_threshold = exam_output_threshold_var
                    #                         output_layer = exam_output_layer
                    #                         # 更換其他操作指標
                    #                         hidden_layer_before_tanh = exam_hidden_layer_before_tanh
                    #                         hidden_layer = exam_hidden_layer
                    #                         output_layer = exam_output_layer
                    #                         average_square_residual = exam_average_square_residual
                    #                         train = exam_train
                    #                         saver = exam_saver
                    #                         # modify constant
                    #                         tau_in_each_hidden_node = exam_tau
                    #                         break
                    # # if k == data_size:
                    # #     new_path = r"{0}/".format(dir_path) + file_output
                    # #     if not os.path.exists(new_path):
                    # #         os.makedirs(new_path)
                    # #
                    # #     file = open(new_path + r"\_training_detail.txt", 'w')
                    # #     file.writelines("learning_rate: " + str(learning_rate_eta) + "\n")
                    # #     file.writelines("input_node_amount: " + str(input_node_amount) + "\n")
                    # #     file.writelines("hidden_node_amount: " + str(hidden_node_amount) + "\n")
                    # #     file.writelines("output_node_amount: " + str(output_node_amount) + "\n")
                    # #     file.writelines("training_data_amount: " + str(data_size) + "\n")
                    # #     file.writelines("alpha(class 1 min value): " + str(alpha) + "\n")
                    # #     file.writelines("beta(class 2 max value): " + str(beta) + "\n")
                    # #     file.writelines("thinking_times_count: " + str(thinking_times_count) + "\n")
                    # #     file.writelines("cramming_times_count: " + str(cramming_times_count) + "\n")
                    # #     file.writelines("softening_thinking_times_count: " + str(softening_thinking_times_count) + "\n")
                    # #     file.writelines("pruning_success_times_count: " + str(pruning_success_times_count) + "\n")
                    # #     file.writelines(
                    # #         "total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
                    # #     file.close()
                    # #     curr_hidden_neuron_weight = sess.run([hidden_weights], {x_placeholder: x_training_data,
                    # #                                                             y_placeholder: y_training_data,
                    # #                                                             tau_placeholder: tau_in_each_hidden_node})
                    # #     np.savetxt(new_path + r"\hidden_neuron_weight.txt", curr_hidden_neuron_weight)
                    # #     curr_hidden_threshold = sess.run([hidden_thresholds],
                    # #                                      {x_placeholder: x_training_data, y_placeholder: y_training_data,
                    # #                                       tau_placeholder: tau_in_each_hidden_node})
                    # #     np.savetxt(new_path + r"\hidden_threshold.txt", curr_hidden_threshold)
                    # #     curr_output_neuron_weight = sess.run([output_weights], {x_placeholder: x_training_data,
                    # #                                                             y_placeholder: y_training_data,
                    # #                                                             tau_placeholder: tau_in_each_hidden_node})
                    # #     np.savetxt(new_path + r"\output_neuron_weight.txt", curr_output_neuron_weight)
                    # #     curr_output_threshold = sess.run([output_threshold],
                    # #                                      {x_placeholder: x_training_data, y_placeholder: y_training_data,
                    # #                                       tau_placeholder: tau_in_each_hidden_node})
                    # #     np.savetxt(new_path + r"\output_threshold.txt", curr_output_threshold)
                    # #
                    # #     curr_average_loss = sess.run([average_square_residual],
                    # #                                  {x_placeholder: x_training_data, y_placeholder: y_training_data,
                    # #                                   tau_placeholder: tau_in_each_hidden_node})
                    # #     file.writelines("average_loss_of_the_model: " + str(curr_average_loss) + "\n")
                    # #
                    # #     np.savetxt(new_path + r"\tau_in_each_hidden_node.txt", tau_in_each_hidden_node)
                    # #
                    # #     print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))

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
        # concat_predict_y = np.concatenate((predict_y_mal_part, predict_y_benign_part), axis=0)
        # concat_fit_value = np.concatenate((fit_value_mal_part, fit_value_benign_part), axis=0)
        # concat_xy_and_y_predict = np.concatenate((concat_x_and_y, concat_predict_y), axis=1)
        # concat_fit_and_x_y_yp = np.concatenate((concat_fit_value, concat_xy_and_y_predict), axis=1)
        # # sort fit value from big to small
        # sort_result = np.flip(concat_fit_and_x_y_yp[np.argsort(concat_fit_and_x_y_yp[:, 0])], axis=0)
        # alpha = min(current_stage_y_predict[np.where(current_stage_y_training_data == 1)[0]])[0]
        # beta = max(current_stage_y_predict[np.where(current_stage_y_training_data == -1)[0]])[0]

        # predict_y_mal_part = sess.run([output_layer], {x_placeholder: x_training_data_mal_part, y_placeholder: y_training_data_mal_part})[0]
        # predict_y_benign_part = sess.run([output_layer], {x_placeholder: x_training_data_benign_part, y_placeholder: y_training_data_benign_part})[0]
        # min_mal_predict_value = min(predict_y_mal_part)
        # max_benign_predict_value = max(predict_y_benign_part)
        # close_to_most_mal_value_mal_part = predict_y_mal_part - min_mal_predict_value
        # close_to_most_benign_value_benign_part = max_benign_predict_value - predict_y_benign_part
        #
        # x_y_mal_part = np.concatenate((x_training_data_mal_part, y_training_data_mal_part), axis=1)
        # x_y_yp_mal_part = np.concatenate((x_y_mal_part, predict_y_mal_part), axis=1)
        # value_x_y_yp_mal_part = np.concatenate((close_to_most_mal_value_mal_part, x_y_yp_mal_part), axis=1)
        # value_x_y_yp_mal_part_sorted_by_value = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, 0])]
        #
        # x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
        # x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
        # value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
        # value_x_y_yp_benign_part_sorted_by_value = value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, 0])]
        #
        # mal_index = 0
        # benign_index = 0
        # while (mal_index + benign_index) < (n - 2):
        #     if mal_index == mal_sample_amount - 1:
        #         benign_index += 1
        #     elif benign_index == benign_sample_amount - 1:
        #         mal_index += 1
        #     else:
        #         a = value_x_y_yp_benign_part_sorted_by_value[benign_index][
        #             value_x_y_yp_benign_part_sorted_by_value.shape[1] - 1]
        #         b = value_x_y_yp_mal_part_sorted_by_value[mal_index][
        #             value_x_y_yp_mal_part_sorted_by_value.shape[1] - 1]
        #         a_1 = value_x_y_yp_benign_part_sorted_by_value[benign_index + 1][
        #             value_x_y_yp_benign_part_sorted_by_value.shape[1] - 1]
        #         b_1 = value_x_y_yp_mal_part_sorted_by_value[mal_index + 1][
        #             value_x_y_yp_mal_part_sorted_by_value.shape[1] - 1]
        #         pick_mal_alpha_minus_beta_value = a - b_1
        #         pick_benign_alpha_minus_beta_value = a_1 - b
        #         if pick_mal_alpha_minus_beta_value > pick_benign_alpha_minus_beta_value:
        #             mal_index += 1
        #         else:
        #             benign_index += 1
        #
        # current_stage_data = np.concatenate((value_x_y_yp_mal_part_sorted_by_value[:mal_index + 1], value_x_y_yp_benign_part_sorted_by_value[:benign_index + 1]), axis=0)
        # current_stage_y_training_data = np.delete(np.delete(current_stage_data, slice(0, m + 1), axis=1), 1, axis=1)  # 去除從0到m&m+2欄
        # current_stage_y_predict = np.delete(current_stage_data, slice(0, m + 2), axis=1)
        # alpha = min(current_stage_y_predict[np.where(current_stage_y_training_data == 1)[0]])[0]
        # beta = max(current_stage_y_predict[np.where(current_stage_y_training_data == -1)[0]])[0]

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
        # value_x_y_yp_mal_part_sorted_by_value = value_x_y_yp_mal_part[np.argsort(value_x_y_yp_mal_part[:, 0])]
        value_x_y_yp_mal_part_sorted_by_yp = value_x_y_yp_mal_part[
            np.argsort(value_x_y_yp_mal_part[:, value_x_y_yp_mal_part.shape[1] - 1])]

        x_y_benign_part = np.concatenate((x_training_data_benign_part, y_training_data_benign_part), axis=1)
        x_y_yp_benign_part = np.concatenate((x_y_benign_part, predict_y_benign_part), axis=1)
        value_x_y_yp_benign_part = np.concatenate((close_to_most_benign_value_benign_part, x_y_yp_benign_part), axis=1)
        # value_x_y_yp_benign_part_sorted_by_value = value_x_y_yp_benign_part[np.argsort(value_x_y_yp_benign_part[:, 0])]
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
        global_max_alpha_minus_beta = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][
                                          value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1] - \
                                      value_x_y_yp_mal_part_sorted_by_yp[global_max_mal_index][
                                          value_x_y_yp_mal_part_sorted_by_yp.shape[1] - 1]
        while mal_index < (int(data_size * (1 - outlier_rate)) - 2) and mal_index < (value_x_y_yp_mal_part_sorted_by_yp.shape[0] - 1) and benign_index > 0:
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
        alpha = value_x_y_yp_benign_part_sorted_by_yp[global_max_benign_index][
            value_x_y_yp_benign_part_sorted_by_yp.shape[1] - 1]
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
        # file.writelines("softening_thinking_times_count: " + str(softening_thinking_times_count) + "\n")
        # file.writelines("pruning_success_times_count: " + str(pruning_success_times_count) + "\n")
        file.writelines("total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
        file.close()
        print("thinking times: %s" % thinking_times_count)
        print("hidden node: %s nodes" % hidden_node_amount)
        print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))
