import tensorflow as tf
import numpy as np
import os
import sys
import time

# 5/5 算出100/all的softmax三種FP/FN

dir_path = os.path.dirname(os.path.realpath(__file__))

# sum_s = 0
false_classification_count = 0

# csv_file = open("__softmax_testing_can_copy_to_figure.csv", 'a')
csv_file = open("___softmax_testing_can_copy_to_figure_unique.csv", 'w')
# csv_file.writelines(', Benign Training Data False Positive Rate, Malicious Training Data False Negative Rate, Benign Testing Data False Positive Rate, Benign Training Data False Negative Rate' + "\n")
# csv_file.writelines(', test_molecular, test_denominator, Testing Data Correct Rate, train_molecular, train_denominator, Training Data Correct Rate' + "\n")

data_amount = ['65', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100', '39', '100', '40', '100', '100', '100', '100']
softmax_nn_arr = ['sampling_envelope', 'sampling_bml', 'run_sampling_training_data', 'all_envelope', 'all_bml', 'run_all_training_data']

benign_testing_all = np.loadtxt(dir_path + r"\19_owl_rules\owl_benign_samples.txt", dtype=str, delimiter=" ")[:, :-1]
# benign_testing_all = np.loadtxt(dir_path + r"\19_owl_rules\owl_rule_1.txt", dtype=str, delimiter=" ")[:, :-1]
benign_testing_all = np.ndarray.astype(benign_testing_all, float)
benign_testing_all = np.unique(benign_testing_all, axis=0)

for www in range(6):
    csv_file.writelines('{0}, test_molecular, test_denominator, Testing Data Correct Rate, train_molecular, train_denominator, Training Data Correct Rate'.format(softmax_nn_arr[www]) + "\n")
    softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_" + softmax_nn_arr[www]
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\multiple_output_node_bp"
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\multiple_output_node_bp_new"
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_binary"
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_envelope"
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_bml"
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_sampling_bml"

    # no select majority by majority learning
    # softmax_nn_dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_run_sampling_training_data"
    benign_training_correct_classification_count = 0
    benign_training_total_sample_amount = 0
    for z in range(20):
        if z == 19:
            input_file_name = "benign_samples"
            file_input = benign_testing_all
        else:
            input_file_name = "rule_"+str(z+1)
            file_input = np.loadtxt(dir_path + r"\19_owl_rules\owl_"+input_file_name+".txt", dtype=str, delimiter=" ")
            file_input = file_input[:, :-1]
            file_input = np.ndarray.astype(file_input, float)
            file_input = np.unique(file_input, axis=0)

        if not z == 19:
            if www <= 2:
                training_data_mal_part = np.loadtxt(dir_path + r"\19_owl_rules\owl_"+input_file_name+"_"+data_amount[z]+"_and_benign_"+data_amount[z]+"_rule_part.txt", dtype=float, delimiter=" ")
                training_data_benign_part = np.loadtxt(dir_path + r"\19_owl_rules\owl_" + input_file_name + "_" + data_amount[z] + "_and_benign_" + data_amount[z] + "_benign_part.txt", dtype=float, delimiter=" ")
                training_data_mal_part = np.unique(training_data_mal_part, axis=0)
                training_data_benign_part = np.unique(training_data_benign_part, axis=0)
                benign_training_total_sample_amount += training_data_benign_part.shape[0]
            else:
                training_data_mal_part = np.loadtxt(dir_path + r"\19_owl_rules\owl_" + input_file_name + ".txt", dtype=str, delimiter=" ")[:, :-1]
                training_data_mal_part = np.ndarray.astype(training_data_mal_part, float)
                training_data_benign_part = np.loadtxt(dir_path + r"\19_owl_rules\owl_" + input_file_name + "_equal_number_benign_sample.txt", dtype=str, delimiter=" ")[:, :-1]
                training_data_benign_part = np.ndarray.astype(training_data_benign_part, float)
                training_data_mal_part = np.unique(training_data_mal_part, axis=0)
                training_data_benign_part = np.unique(training_data_benign_part, axis=0)
                benign_training_total_sample_amount += training_data_benign_part.shape[0]

        # file_input = np.loadtxt(dir_path + r"\envelope_majority_" + input_file_name + ".txt", dtype=float, delimiter=" ")
        # file_input = np.loadtxt(dir_path + r"\binary_majority_" + input_file_name + ".txt", dtype=float, delimiter=" ")
        # file_input = np.loadtxt(dir_path + r"\19_owl_rules\owl_rule_"+str(z)+"_"+data_amount[z-1]+"_and_benign_"+data_amount[z-1]+"_rule_part.txt", dtype=float, delimiter=" ")
        # file_input = np.loadtxt(dir_path + r"\19_owl_rules\owl_rule_" + str(z) + "_" + data_amount[z - 1] + "_and_benign_" + data_amount[z - 1] + "_benign_part.txt", dtype=float, delimiter=" ")

        rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
        # data_amount_arr = np.array([65, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 39, 100, 40, 100, 100, 100, 100], dtype=str)

        # match_arr = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=bool)
        # match_count_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float64, name='x_placeholder')
            W = tf.Variable(np.loadtxt(softmax_nn_dir_save_result + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
            b = tf.Variable(np.loadtxt(softmax_nn_dir_save_result + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')
            y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

            prediction = tf.argmax(y, 1)

            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()

            # start_time = time.time()
            testing_result = sess.run([prediction], {x: file_input})[0]
            # end_time = time.time()
            # print(end_time-start_time)
            # print(result[np.where(result != 0)])

            training_data_mal_predict_result = sess.run([prediction], {x: training_data_mal_part})[0]
            training_data_benign_predict_result = sess.run([prediction], {x: training_data_benign_part})[0]
            mal_training_match_count = np.where(training_data_mal_predict_result == (z+1))[0].shape[0]
            benign_training_match_count = np.where(training_data_benign_predict_result == 0)[0].shape[0]
            if not z == 19:
                benign_training_correct_classification_count += benign_training_match_count
            # print(np.where(training_data_mal_predict_result == (z + 1))[0].shape[0])
            # print(np.where(training_data_benign_predict_result == 0)[0].shape[0])
            # input(1)

        # 19 output node
        match_str = ""
        for i in range(1, 20):
            if testing_result[np.where(testing_result == i)].shape[0] > 0:
                match_str = match_str + rule_arr[i - 1] + " "
        # print("Malware " + input_file_name + " matches rule: " + match_str)
        # analyze_result.writelines("Malware "+input_file_name+" matches rule: "+match_str+"\n")
        # print("Malware " + input_file_name + " has " + str(file_input.shape[0]) + " samples.")
        # analyze_result.writelines("Malware "+input_file_name+" has "+str(file_input.shape[0])+" samples.\n")
        for i in range(0, 20):
            # match_count = result[np.where(result == i)].shape[0]
            if i == 19:
                testing_match_count = testing_result[np.where(testing_result == 0)[0]].shape[0]
                # print("Match benign sample amount: " + str(testing_match_count))
                # analyze_result.writelines("Match rule benign sample amount: " + str(match_count) + "\n")
                # sum_s += match_count
                if z == i:
                    csv_file.writelines("Benign, {0}, {1}, {2}, {3}, {4}, {5}".format(testing_match_count-benign_training_correct_classification_count, file_input.shape[0]-benign_training_total_sample_amount, float((testing_match_count-benign_training_correct_classification_count) / (file_input.shape[0]-benign_training_total_sample_amount)), benign_training_correct_classification_count, benign_training_total_sample_amount, float(benign_training_correct_classification_count/benign_training_total_sample_amount)) + "\n")
            else:
                testing_match_count = testing_result[np.where(testing_result == (i+1))[0]].shape[0]
                # print("Match rule " + str(i+1) + " sample amount: " + str(testing_match_count))
                # analyze_result.writelines("Match rule "+str(i)+" sample amount: "+str(match_count)+"\n")
                if z == i:
                    if not file_input.shape[0]-training_data_mal_part.shape[0] == 0:
                        csv_file.writelines("Rule {0}, {1}, {2}, {3}, {4}, {5}, {6}".format(i + 1, testing_match_count-mal_training_match_count, file_input.shape[0]-training_data_mal_part.shape[0], float((testing_match_count-mal_training_match_count) / (file_input.shape[0]-training_data_mal_part.shape[0])), mal_training_match_count, training_data_mal_part.shape[0], float(mal_training_match_count/training_data_mal_part.shape[0])) + "\n")
                    else:
                        csv_file.writelines("Rule {0}, {1}, {2}, {3}, {4}, {5}, {6}".format(i + 1, 0, 0, 0, mal_training_match_count, training_data_mal_part.shape[0], float(mal_training_match_count/training_data_mal_part.shape[0])) + "\n")

    csv_file.writelines("\n"+"\n"+"\n"+"\n"+"\n")

csv_file.close()
# print(sum_s)
