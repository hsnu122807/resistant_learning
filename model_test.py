import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

x_placeholder = tf.placeholder(tf.float64)
# y_placeholder = tf.placeholder(tf.float64)

rule = "6"
ntu = "3"
data_amount = "1250"
# light = "" or "_light"
light = ""
sigma = "2"
file_input = "TensorFlow_input_detection_rule_"+rule+"_"+data_amount+"_and_ntu_"+ntu+"_benign_"+data_amount+light+"_no_label"+"_sigma_" + sigma + "_wrong_10_label"
# file_input = "TensorFlow_input_detection_rule_"+rule+"_"+data_amount+"_and_ntu_"+ntu+"_benign_"+data_amount+light+"_no_label"
dir_target = dir_path + r"\TensorFlow_input_detection_rule_"+rule+"_"+data_amount+"_and_ntu_"+ntu+"_benign_"+data_amount+light+"_no_label"+"_sigma_" + sigma + "_wrong_10_label"


ot = np.loadtxt(dir_target+r"\output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
ow = np.loadtxt(dir_target+r"\output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
ht = np.loadtxt(dir_target+r"\hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
hw = np.loadtxt(dir_target+r"\hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))

training_data_result = np.loadtxt(dir_target+r"\training_data_residual_predict_output_desire_output_desire_input.txt")

training_data_predict_output = training_data_result[:, 1]
training_data_desire_output = training_data_result[:, 2]
# print(training_data_predict_output)
# print(training_data_desire_output.shape[0])
not_outlier_data_amount = int(training_data_desire_output.shape[0] * 0.95)
outliers_predict_output = training_data_predict_output[not_outlier_data_amount:]
outliers_desire_output = training_data_desire_output[not_outlier_data_amount:]
# print(outliers_predict_output)
# print(outliers_desire_output)
outlier_benign_count = 0
outlier_malware_count = 0
for i in range(outliers_desire_output.shape[0]):
    if outliers_desire_output[i] == 1:
        outlier_benign_count += 1
    elif outliers_desire_output[i] == -1:
        outlier_malware_count += 1
# input("123")

training_data_benign_count = 0
training_data_malware_count = 0
training_data_benign_predict_correct_count = 0
training_data_malware_predict_correct_count = 0
for i in range(training_data_desire_output.shape[0]):
    if training_data_desire_output[i] == 1:
        training_data_benign_count += 1
        if training_data_predict_output[i] >= 0:
            training_data_benign_predict_correct_count += 1
    elif training_data_desire_output[i] == -1:
        training_data_malware_count += 1
        if training_data_predict_output[i] < 0:
            training_data_malware_predict_correct_count += 1

testing_data_dir = r"C:\Users\user\workspace\TextConvert"
malware_samples = np.loadtxt(testing_data_dir+r"\detection_rule_"+rule+"_samples"+light+"_no_label.txt", dtype=float, delimiter=" ")
malware_samples_training_part = np.loadtxt(testing_data_dir+r"\detection_rule_"+rule+"_samples"+light+"_no_label.txt", dtype=float, delimiter=" ")
if light == "":
    benign_samples = np.loadtxt(testing_data_dir+r"\ntu_30_5_"+ntu+"_result_converted_only_benign_samples_no_label.txt", dtype=float, delimiter=" ")
else:
    benign_samples = np.loadtxt(testing_data_dir + r"\ntu_30_5_"+ntu+"_result_converted_only_benign_samples_light_for_rule_"+rule+"_no_label.txt", dtype=float, delimiter=" ")

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
testing_data_malware_predict_wrong_count = predict_malware[np.where(predict_malware >= 0)].shape[0] - (training_data_malware_count - training_data_malware_predict_correct_count)

predict_benign = sess.run([output_layer], {x_placeholder: benign_samples})[0]
# print(predict_benign.shape)
# print(predict_benign[np.where(predict_benign < 0)].shape)
testing_data_benign_count = predict_benign.shape[0] - training_data_benign_count
testing_data_benign_predict_wrong_count = predict_benign[np.where(predict_benign < 0)].shape[0] - (training_data_benign_count - training_data_benign_predict_correct_count)

print(file_input)
print("False Positive of Benign Training Data: {0}/{1}".format(training_data_benign_count-training_data_benign_predict_correct_count, training_data_benign_count))
print("False Negative of Rule Training Data: {0}/{1}".format(training_data_malware_count-training_data_malware_predict_correct_count, training_data_malware_count))
print("False Positive of Benign Testing Data: {0}/{1}".format(testing_data_benign_predict_wrong_count, testing_data_benign_count))
print("False Negative of Rule Testing Data: {0}/{1}".format(testing_data_malware_predict_wrong_count, testing_data_malware_count))
print("There are {0} benign & {1} malware in outliers".format(outlier_benign_count, outlier_malware_count))
