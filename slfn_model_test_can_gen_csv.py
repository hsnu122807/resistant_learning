import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

testing_data_dir = dir_path + r"\19_owl_rules"
# benign_samples = np.loadtxt(testing_data_dir+r"\owl_benign_samples.txt", dtype=str, delimiter=" ")
# benign_samples = benign_samples[:, :benign_samples.shape[1]-1]
# benign_samples = np.ndarray.astype(benign_samples, float)
# print('end read file')

rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
data_amount_arr = np.array([65, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 39, 100, 40, 100, 100, 100, 100], dtype=str)
middle_point_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

csv_file = open("__can_copy_to_figure.csv", 'w')
csv_file.writelines(', Benign Training Data False Positive Rate, Malicious Training Data False Negative Rate, Benign Testing Data False Positive Rate, Benign Training Data False Negative Rate' + "\n")

for a in range(rule_arr.shape[0]):
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float64)
        rule = rule_arr[a]
        data_amount = data_amount_arr[a]
        sigma = "2"
        # file_input = "owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_sigma_"+sigma
        # dir_target = dir_path + r"\19_owl_rules"+r"\owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_sigma_"+sigma
        file_input = "Rule "+rule
        # dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_all_training_data_all"
        dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_bml"
        # dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_all_training_data_bml"

        ot = np.loadtxt(dir_target+r"\two_class_output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
        ow = np.loadtxt(dir_target+r"\two_class_output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
        ht = np.loadtxt(dir_target+r"\two_class_hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
        hw = np.loadtxt(dir_target+r"\two_class_hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))

        file = open(dir_target + r"\_two_class_training_detail.txt")
        while 1:
            line = file.readline()
            if not line:
                print("find middle point error!")
                break
            if "classify middle point((alpha+beta)/2): " in line:
                middle_point = float(line.replace("classify middle point((alpha+beta)/2): ", ""))
                middle_point_arr[a] = middle_point
                break
        file.close()

        training_data_result = np.loadtxt(dir_target+r"\two_class_training_data_fit_x_y_yp.txt", dtype=float, delimiter=" ")

        training_data_predict_output = training_data_result[:, 54]
        training_data_desire_output = training_data_result[:, 53]
        # print(training_data_predict_output)
        # print(training_data_desire_output.shape[0])
        not_outlier_data_amount = int(training_data_desire_output.shape[0] * 0.95)
        outliers_predict_output = training_data_predict_output[not_outlier_data_amount:]
        outliers_desire_output = training_data_desire_output[not_outlier_data_amount:]
        # print(outliers_predict_output)
        # print(outliers_desire_output)
        potential_outlier_benign_count = 0
        potential_outlier_malware_count = 0
        for i in range(outliers_desire_output.shape[0]):
            if outliers_desire_output[i] == 1:
                potential_outlier_benign_count += 1
            elif outliers_desire_output[i] == -1:
                potential_outlier_malware_count += 1
        # input("123")

        training_data_benign_count = 0
        training_data_malware_count = 0
        training_data_benign_predict_correct_count = 0
        training_data_malware_predict_correct_count = 0
        for i in range(training_data_desire_output.shape[0]):
            if training_data_desire_output[i] == 1:
                training_data_benign_count += 1
                if training_data_predict_output[i] >= middle_point_arr[a]:
                    training_data_benign_predict_correct_count += 1
            elif training_data_desire_output[i] == -1:
                training_data_malware_count += 1
                if training_data_predict_output[i] < middle_point_arr[a]:
                    training_data_malware_predict_correct_count += 1

        malware_samples = np.loadtxt(testing_data_dir+r"\owl_rule_"+rule+".txt", dtype=str, delimiter=" ")
        # malware_samples_training_part = np.loadtxt(testing_data_dir+r"\owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_rule_part.txt", dtype=float, delimiter=" ")
        malware_samples = malware_samples[:, :malware_samples.shape[1]-1]
        malware_samples = np.ndarray.astype(malware_samples, float)
        benign_samples_training_part = np.loadtxt(testing_data_dir+r"\owl_rule_"+rule+"_equal_number_benign_sample.txt", dtype=str, delimiter=" ")
        benign_samples_training_part = benign_samples_training_part[:, :benign_samples_training_part.shape[1] - 1]
        benign_samples_training_part = np.ndarray.astype(benign_samples_training_part, float)

        # malware_samples = np.loadtxt(testing_data_dir + r"\owl_rule_" + rule + ".txt", dtype=str, delimiter=" ")
        # malware_samples_training_part = np.loadtxt(
        #     testing_data_dir + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_rule_part.txt",
        #     dtype=float, delimiter=" ")
        # malware_samples = malware_samples[:, :malware_samples.shape[1] - 1]
        # malware_samples = np.ndarray.astype(malware_samples, float)
        # benign_samples_training_part = np.loadtxt(
        #     testing_data_dir + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_benign_part.txt",
        #     dtype=float, delimiter=" ")

        # print(malware_samples.shape)
        # print(ot.shape)
        # print(ow.shape)
        # print(ht.shape)
        # print(hw.shape)

        output_threshold = tf.Variable(ot, dtype=tf.float64)
        output_weights = tf.Variable(ow, dtype=tf.float64)
        hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
        hidden_weights = tf.Variable(hw, dtype=tf.float64)

        hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
        output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        predict_malware = sess.run([output_layer], {x_placeholder: malware_samples})[0]
        # print(predict_malware.shape)
        # print(predict_malware[np.where(predict_malware >= 0)].shape)
        testing_data_malware_count = predict_malware.shape[0] - training_data_malware_count
        testing_data_malware_predict_wrong_count = predict_malware[np.where(predict_malware >= middle_point_arr[a])].shape[0] - (training_data_malware_count - training_data_malware_predict_correct_count)

        # predict_benign = sess.run([output_layer], {x_placeholder: benign_samples})[0]
        # # print(predict_benign.shape)
        # # print(predict_benign[np.where(predict_benign < 0)].shape)
        # testing_data_benign_count = predict_benign.shape[0] - training_data_benign_count
        # testing_data_benign_predict_wrong_count = predict_benign[np.where(predict_benign < middle_point_arr[a])].shape[0] - (training_data_benign_count - training_data_benign_predict_correct_count)

        print(file_input)
        print("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count-training_data_benign_predict_correct_count, training_data_benign_count))
        print("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count-training_data_malware_predict_correct_count, training_data_malware_count))
        # print("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count))
        print("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count))
        print("There are {0} benign & {1} malware in potential outliers".format(potential_outlier_benign_count, potential_outlier_malware_count))

        fpb_train = (training_data_benign_count-training_data_benign_predict_correct_count) / training_data_benign_count
        fnr_train = (training_data_malware_count-training_data_malware_predict_correct_count) / training_data_malware_count
        # fpb_test = testing_data_benign_predict_wrong_count / testing_data_benign_count
        if testing_data_malware_count == 0:
            fnr_test = 0
        else:
            fnr_test = testing_data_malware_predict_wrong_count / testing_data_malware_count
        # csv_file.writelines("{0}, {1}, {2}, {3}, {4}".format(file_input, fpb_train, fnr_train, fpb_test, fnr_test) + "\n")
csv_file.close()
