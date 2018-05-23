import tensorflow as tf
import numpy as np
import os
import time

# 我想實驗看看outlier得到的nn判斷準確度較高還是新生成的nn判斷準確度較高
# 實驗結果: 有好有壞

# 手動改的
is_all = True
print_env = False
print_softmax = True
print_bml = False
input(1)

dir_path = os.path.dirname(os.path.realpath(__file__))


testing_data_dir = dir_path + r"\19_owl_rules"
start_time = time.time()
benign_samples = np.loadtxt(testing_data_dir+r"\owl_benign_samples.txt", dtype=str, delimiter=" ")
# benign_samples = np.loadtxt(testing_data_dir+r"\owl_rule_1.txt", dtype=str, delimiter=" ")  # 用來測試 code 正確性
benign_samples = benign_samples[:, :benign_samples.shape[1]-1]
benign_samples = np.ndarray.astype(benign_samples, float)
end_time = time.time()
print(end_time - start_time)

rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
# rule_arr = np.array([1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19], dtype=str)
data_amount_arr = np.array([65, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 39, 100, 40, 100, 100, 100, 100], dtype=str)
outlier_amount = np.array([6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 10, 4, 10, 10, 10, 10], dtype=int)
# for a in range(1):
for a in range(4, rule_arr.shape[0]):
    # TODO: rule 5,6 OOM in allocate large tensor, try to fix it
    # for a in range(6, 19):
    rule = rule_arr[a]
    data_amount = data_amount_arr[a]
    sigma = "2"
    malware_samples = np.loadtxt(testing_data_dir + r"\owl_rule_" + rule + ".txt", dtype=str, delimiter=" ")
    malware_samples = malware_samples[:, :malware_samples.shape[1] - 1]
    malware_samples = np.ndarray.astype(malware_samples, float)
    if is_all:
        dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_all_training_data_sigma_" + sigma
    else:
        dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_sigma_" + sigma
    result_file = open(dir_target + r"\_outlier_nn_vs_pure_bp_nn.txt", 'a')
    if print_env:
        with tf.Graph().as_default():
            x_placeholder = tf.placeholder(tf.float64)

            rule = rule_arr[a]
            data_amount = data_amount_arr[a]
            sigma = "2"
            # file_input = "owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_sigma_"+sigma
            if is_all:
                dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_all_training_data_sigma_" + sigma
                training_data_result = np.loadtxt(
                    dir_target + r"\training_data_residual_predict_output_desire_output_desire_input.txt", dtype=float,
                    delimiter=" ")
                majority_amount = int(training_data_result.shape[0]*0.95)
                major_data_x = training_data_result[:majority_amount, 3:]
                major_data_y = training_data_result[:majority_amount, 2].reshape((-1, 1))
                major_data_predict_y = training_data_result[:majority_amount, 1].reshape(
                    (-1, 1))
            else:
                dir_target = dir_path + r"\19_owl_rules"+r"\owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_sigma_"+sigma
                training_data_result = np.loadtxt(
                    dir_target + r"\training_data_residual_predict_output_desire_output_desire_input.txt", dtype=float,
                    delimiter=" ")
                major_data_x = training_data_result[:int(int(data_amount) * 2 - outlier_amount[a]), 3:]
                major_data_y = training_data_result[:int(int(data_amount) * 2 - outlier_amount[a]), 2].reshape((-1, 1))
                major_data_predict_y = training_data_result[:int(int(data_amount) * 2 - outlier_amount[a]), 1].reshape(
                    (-1, 1))
            alpha = min(major_data_predict_y[np.where(major_data_y == 1)])
            beta = max(major_data_predict_y[np.where(major_data_y == -1)])
            middle_point = (alpha + beta) / 2
            # print(rule)
            # print((alpha+beta)/2)

            ot = np.loadtxt(dir_target+r"\output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
            ow = np.loadtxt(dir_target+r"\output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
            ht = np.loadtxt(dir_target+r"\hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
            hw = np.loadtxt(dir_target+r"\hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))

            training_data_result = np.loadtxt(dir_target+r"\training_data_residual_predict_output_desire_output_desire_input.txt")

            training_data_predict_output = training_data_result[:, 1]
            training_data_desire_output = training_data_result[:, 2]
            # # print(training_data_predict_output)
            # # print(training_data_desire_output.shape[0])

            training_data_benign_count = 0
            training_data_malware_count = 0
            training_data_benign_predict_correct_count = 0
            training_data_malware_predict_correct_count = 0
            for i in range(training_data_desire_output.shape[0]):
                if training_data_desire_output[i] == 1:
                    training_data_benign_count += 1
                    if training_data_predict_output[i] >= middle_point:
                        training_data_benign_predict_correct_count += 1
                elif training_data_desire_output[i] == -1:
                    training_data_malware_count += 1
                    if training_data_predict_output[i] < middle_point:
                        training_data_malware_predict_correct_count += 1

            malware_samples = np.loadtxt(testing_data_dir + r"\owl_rule_" + rule + ".txt", dtype=str, delimiter=" ")
            if is_all:
                malware_samples_training_part = malware_samples
            else:
                malware_samples_training_part = np.loadtxt(
                    testing_data_dir + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_rule_part.txt",
                    dtype=float, delimiter=" ")
            malware_samples = malware_samples[:, :malware_samples.shape[1] - 1]
            malware_samples = np.ndarray.astype(malware_samples, float)
            if is_all:
                benign_samples_training_part = np.loadtxt(testing_data_dir + r"\owl_rule_{0}.txt".format(rule), dtype=str,
                                                          delimiter=" ")[:, :-1]
                benign_samples_training_part = np.ndarray.astype(benign_samples_training_part, float)
            else:
                benign_samples_training_part = np.loadtxt(
                    testing_data_dir + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_benign_part.txt",
                    dtype=float, delimiter=" ")

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
            # print(predict_malware[np.where(predict_malware >= middle_point)].shape)
            testing_data_malware_count = predict_malware.shape[0] - training_data_malware_count
            testing_data_malware_predict_wrong_count = predict_malware[np.where(predict_malware >= middle_point)].shape[0] - (training_data_malware_count - training_data_malware_predict_correct_count)

            start_time = time.time()
            predict_benign = sess.run([output_layer], {x_placeholder: benign_samples})[0]
            end_time = time.time()
            print("forward pass 120W need "+str(end_time-start_time)+" seconds")
            # print(predict_benign.shape)
            # print(predict_benign[np.where(predict_benign < middle_point)].shape)
            testing_data_benign_count = predict_benign.shape[0] - training_data_benign_count
            testing_data_benign_predict_wrong_count = predict_benign[np.where(predict_benign < middle_point)].shape[0] - (training_data_benign_count - training_data_benign_predict_correct_count)

            # print(file_input)
            print("outlier neuron network")
            print("middle point: {0}".format(middle_point))
            print("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count-training_data_benign_predict_correct_count, training_data_benign_count))
            print("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count-training_data_malware_predict_correct_count, training_data_malware_count))
            print("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count))
            print("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count))

            result_file.writelines("outlier neuron network"+"\n")
            result_file.writelines("alpha: {0}".format(alpha)+"\n")
            result_file.writelines("beta: {0}".format(beta) + "\n")
            result_file.writelines("middle point: {0}".format(middle_point)+"\n")
            result_file.writelines("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count-training_data_benign_predict_correct_count, training_data_benign_count)+"\n")
            result_file.writelines("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count - training_data_malware_predict_correct_count, training_data_malware_count)+"\n")
            result_file.writelines("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count)+"\n")
            result_file.writelines("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count)+"\n")
            result_file.writelines("\n")

    if print_softmax:
        with tf.Graph().as_default():
            if is_all:
                dir_save_result = dir_path + r"\19_owl_rules\owl_rule_" + rule + "_all_training_data_softmax"
                malware_samples_training_part = malware_samples
            else:
                dir_save_result = dir_path + r"\19_owl_rules\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_softmax"
                malware_samples_training_part = np.loadtxt(
                    testing_data_dir + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_rule_part.txt",
                    dtype=float, delimiter=" ")

            x = tf.placeholder(tf.float64, name='x_placeholder')
            W = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
            b = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')
            y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

            prediction = tf.argmax(y, 1)

            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()

            # target_dir = dir_path + r"\19_owl_rules\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_softmax"
            # file = open(target_dir + r"\_training_analyze.txt")
            file = open(dir_save_result + r"\_training_analyze.txt")
            line_index = 1
            while 1:
                line = file.readline()
                if not line:
                    break
                if line_index == 1:
                    training_data_benign_predict_correct_count = int(line.split(": ")[1].split("/")[0])
                    training_data_benign_count = int(line.split(": ")[1].split("/")[1].split(' ')[0])
                if line_index == 2:
                    training_data_malware_predict_correct_count = int(line.split(": ")[1].split("/")[0])
                    training_data_malware_count = int(line.split(": ")[1].split("/")[1].split(' ')[0])
                line_index += 1
            file.close()

            predict_malware = sess.run([prediction], {x: malware_samples})[0]
            testing_data_malware_count = predict_malware.shape[0] - training_data_malware_count
            testing_data_malware_predict_wrong_count = predict_malware[np.where(predict_malware == 0)[0]].shape[0] - (training_data_malware_count - training_data_malware_predict_correct_count)

            predict_benign = sess.run([prediction], {x: benign_samples})[0]
            testing_data_benign_count = predict_benign.shape[0] - training_data_benign_count
            testing_data_benign_predict_wrong_count = predict_benign[np.where(predict_benign == 1)].shape[0] - (training_data_benign_count - training_data_benign_predict_correct_count)

            print("softmax neuron network")
            print("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count - training_data_benign_predict_correct_count, training_data_benign_count))
            print("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count - training_data_malware_predict_correct_count, training_data_malware_count))
            print("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count))
            print("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count))

            if print_softmax:
                result_file.writelines("softmax neural network" + "\n")
                result_file.writelines("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count - training_data_benign_predict_correct_count, training_data_benign_count) + "\n")
                result_file.writelines("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count - training_data_malware_predict_correct_count, training_data_malware_count) + "\n")
                result_file.writelines("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count) + "\n")
                result_file.writelines("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count) + "\n")
                result_file.writelines("\n")

        # ot = np.loadtxt(dir_target + r"\precise_classify_neuron_network_output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
        # ow = np.loadtxt(dir_target + r"\precise_classify_neuron_network_output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
        # ht = np.loadtxt(dir_target + r"\precise_classify_neuron_network_hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
        # hw = np.loadtxt(dir_target + r"\precise_classify_neuron_network_hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))
        #
        # output_threshold = tf.Variable(ot, dtype=tf.float64)
        # output_weights = tf.Variable(ow, dtype=tf.float64)
        # hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
        # hidden_weights = tf.Variable(hw, dtype=tf.float64)
        #
        # hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
        # output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)
        #
        # init = tf.global_variables_initializer()
        # sess = tf.Session()
        # sess.run(init)
        #
        # # predict_major_training_data = sess.run([output_layer], {x_placeholder: major_data_x})[0]
        # # alpha = min(predict_major_training_data[np.where(major_data_y == 1)])
        # # beta = max(predict_major_training_data[np.where(major_data_y == -1)])
        # # middle_point = (alpha + beta) / 2
        # precise_bp_nn_middle_points = np.array([0.0780018986075, 0.0772721429457, 3.73336504032, 0.398415658672, 0.255079691781, 0.743015386106], dtype=float)
        # middle_point = precise_bp_nn_middle_points[a]
        #
        # file_name = r"19_owl_rules\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount
        # training_data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=" ")
        # x_training_data = training_data[:, 1:]
        # y_training_data = training_data[:, 0].reshape((-1, 1))
        # predict_training_data = sess.run([output_layer], {x_placeholder: x_training_data})[0]
        # predict_training_data_benign_part = predict_training_data[np.where(y_training_data == 1)]
        # predict_training_data_mal_part = predict_training_data[np.where(y_training_data == -1)]
        #
        # training_data_benign_count = predict_training_data_benign_part.shape[0]
        # training_data_malware_count = predict_training_data_mal_part.shape[0]
        # training_data_benign_predict_correct_count = predict_training_data_benign_part[np.where(predict_training_data_benign_part >= middle_point)].shape[0]
        # training_data_malware_predict_correct_count = predict_training_data_mal_part[np.where(predict_training_data_mal_part < middle_point)].shape[0]
        #
        # predict_malware = sess.run([output_layer], {x_placeholder: malware_samples})[0]
        # testing_data_malware_count = predict_malware.shape[0] - training_data_malware_count
        # testing_data_malware_predict_wrong_count = predict_malware[np.where(predict_malware >= middle_point)].shape[0] - (training_data_malware_count - training_data_malware_predict_correct_count)
        #
        # predict_benign = sess.run([output_layer], {x_placeholder: benign_samples})[0]
        # testing_data_benign_count = predict_benign.shape[0] - training_data_benign_count
        # testing_data_benign_predict_wrong_count = predict_benign[np.where(predict_benign < middle_point)].shape[0] - (training_data_benign_count - training_data_benign_predict_correct_count)
        #
        # print("precise bp neuron network")
        # print("middle point: {0}".format(middle_point))
        # print("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count - training_data_benign_predict_correct_count, training_data_benign_count))
        # print("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count - training_data_malware_predict_correct_count, training_data_malware_count))
        # print("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count))
        # print("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count))
        #
        # result_file.writelines("precise bp neuron network" + "\n")
        # result_file.writelines("alpha and beta was record in training detail.\n")
        # # result_file.writelines("alpha: {0}".format(alpha) + "\n")
        # # result_file.writelines("beta: {0}".format(beta) + "\n")
        # result_file.writelines("middle point: {0}".format(middle_point) + "\n")
        # result_file.writelines("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count - training_data_benign_predict_correct_count, training_data_benign_count) + "\n")
        # result_file.writelines("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count - training_data_malware_predict_correct_count, training_data_malware_count) + "\n")
        # result_file.writelines("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count) + "\n")
        # result_file.writelines("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count) + "\n")
        # result_file.writelines("\n")
    if print_bml:
        with tf.Graph().as_default():
            x_placeholder = tf.placeholder(tf.float64)
            if is_all:
                dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_all_training_data_bml"
            else:
                dir_target = dir_path + r"\19_owl_rules" + r"\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount + "_bml"

            ot = np.loadtxt(dir_target + r"\two_class_output_threshold.txt", dtype=float,
                            delimiter=" ").reshape(1)
            ow = np.loadtxt(dir_target + r"\two_class_output_neuron_weight.txt", dtype=float,
                            delimiter=" ").reshape((-1, 1))
            ht = np.loadtxt(dir_target + r"\two_class_hidden_threshold.txt", dtype=float,
                            delimiter=" ").reshape((1, -1))
            hw = np.loadtxt(dir_target + r"\two_class_hidden_neuron_weight.txt", dtype=float,
                            delimiter=" ").reshape((-1, ht.shape[1]))

            output_threshold = tf.Variable(ot, dtype=tf.float64)
            output_weights = tf.Variable(ow, dtype=tf.float64)
            hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
            hidden_weights = tf.Variable(hw, dtype=tf.float64)

            hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
            output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            training_data_result = np.loadtxt(
                dir_target + r"\two_class_training_data_fit_x_y_yp.txt", dtype=float,
                delimiter=" ")
            if is_all:
                majority_amount = int(training_data_result.shape[0]*0.95)
                major_data_x = training_data_result[:majority_amount, 1:53]
                major_data_y = training_data_result[:majority_amount, 53].reshape((-1, 1))
                major_data_predict_y = training_data_result[:majority_amount, 54].reshape(
                    (-1, 1))
            else:
                major_data_x = training_data_result[:int(int(data_amount) * 2 - outlier_amount[a]), 1:53]
                major_data_y = training_data_result[:int(int(data_amount) * 2 - outlier_amount[a]), 53].reshape((-1, 1))
                major_data_predict_y = training_data_result[:int(int(data_amount) * 2 - outlier_amount[a]), 54].reshape((-1, 1))
            alpha = min(major_data_predict_y[np.where(major_data_y == 1)])
            beta = max(major_data_predict_y[np.where(major_data_y == -1)])
            # print(alpha)
            # print(beta)
            middle_point = (alpha + beta) / 2
            # print(middle_point)

            if is_all:
                file_name = r"19_owl_rules\owl_rule_" + rule + "_all_training_data"
                training_data = np.loadtxt(file_name + ".txt", dtype=str, delimiter=" ")[:, :-1]
                training_data = np.ndarray.astype(training_data, float)
            else:
                file_name = r"19_owl_rules\owl_rule_" + rule + "_" + data_amount + "_and_benign_" + data_amount
                training_data = np.loadtxt(file_name + ".txt", dtype=float, delimiter=" ")
            x_training_data = training_data[:, 1:]
            y_training_data = training_data[:, 0].reshape((-1, 1))
            predict_training_data = sess.run([output_layer], {x_placeholder: x_training_data})[0]
            predict_training_data_benign_part = predict_training_data[np.where(y_training_data == 1)]
            predict_training_data_mal_part = predict_training_data[np.where(y_training_data == -1)]

            training_data_benign_count = predict_training_data_benign_part.shape[0]
            training_data_malware_count = predict_training_data_mal_part.shape[0]
            training_data_benign_predict_correct_count = predict_training_data_benign_part[np.where(predict_training_data_benign_part >= middle_point)].shape[0]
            training_data_malware_predict_correct_count = predict_training_data_mal_part[np.where(predict_training_data_mal_part < middle_point)].shape[0]

            predict_malware = sess.run([output_layer], {x_placeholder: malware_samples})[0]
            testing_data_malware_count = predict_malware.shape[0] - training_data_malware_count
            testing_data_malware_predict_wrong_count = predict_malware[np.where(predict_malware >= middle_point)].shape[
                                                           0] - (
                                                       training_data_malware_count - training_data_malware_predict_correct_count)

            predict_benign = sess.run([output_layer], {x_placeholder: benign_samples})[0]
            testing_data_benign_count = predict_benign.shape[0] - training_data_benign_count
            testing_data_benign_predict_wrong_count = predict_benign[np.where(predict_benign < middle_point)].shape[0] - (training_data_benign_count - training_data_benign_predict_correct_count)

            print("bml")
            print("middle point: {0}".format(middle_point))
            print("False Positive of Benign Training Data: {0}/{1}".format(
                training_data_benign_count - training_data_benign_predict_correct_count, training_data_benign_count))
            print("False Negative of Rule Training Data: {0}/{1}".format(
                training_data_malware_count - training_data_malware_predict_correct_count, training_data_malware_count))
            print("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count,
                                                                          testing_data_benign_count))
            print("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count,
                                                                        testing_data_malware_count))

            if print_bml:
                result_file.writelines("bml" + "\n")
                result_file.writelines("alpha: {0}".format(alpha) + "\n")
                result_file.writelines("beta: {0}".format(beta) + "\n")
                result_file.writelines("middle point: {0}".format(middle_point) + "\n")
                result_file.writelines("False Positive of Benign Training Data: {0}/{1}".format(
                    training_data_benign_count - training_data_benign_predict_correct_count, training_data_benign_count) + "\n")
                result_file.writelines("False Negative of Rule Training Data: {0}/{1}".format(
                    training_data_malware_count - training_data_malware_predict_correct_count,
                    training_data_malware_count) + "\n")
                result_file.writelines(
                    "False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count,
                                                                            testing_data_benign_count) + "\n")
                result_file.writelines(
                    "False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count,
                                                                          testing_data_malware_count) + "\n")
                result_file.writelines("\n")

    result_file.close()
