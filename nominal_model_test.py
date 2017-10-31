import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

x_placeholder = tf.placeholder(tf.float64)

ntu = "4"
rule = "22"
label = 7
benign = True

if ntu == "3":
    rule_string = "2_4_6_7_10_11_12"
if ntu == "4":
    rule_string = "15_16_19_22"
file_input = r"nominal_data\ntu_rule_" + rule_string + "_100_and_benign_100"

if not benign:
    complete_data = np.loadtxt(r"nominal_data\ntu_"+ntu+"_detection_rule_"+rule+"_samples_light.txt", dtype=str)
else:
    complete_data = np.loadtxt(r"nominal_data\ntu_"+ntu+"_detection_rule_benign_samples_light.txt", dtype=str)
complete_data = complete_data[:, :complete_data.shape[1]-1]
complete_data = np.ndarray.astype(complete_data, float)
if not benign:
    training_data = np.loadtxt(r"nominal_data\ntu_rule_"+rule_string+"_100_and_benign_100_rule_"+rule+"_part.txt", dtype=str)
else:
    training_data = np.loadtxt(r"nominal_data\ntu_rule_"+rule_string+"_100_and_benign_100_benign_part.txt", dtype=str)
training_data = training_data[:, :training_data.shape[1]-1]
training_data = np.ndarray.astype(training_data, float)

dir_target = file_input + "_sigma_2"

ot = np.loadtxt(dir_target+r"\output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
ow = np.loadtxt(dir_target+r"\output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
ht = np.loadtxt(dir_target+r"\hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
hw = np.loadtxt(dir_target+r"\hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))

training_data_result = np.loadtxt(dir_target+r"\training_data_residual_predict_output_desire_output_desire_input.txt")

training_data_predict_output = training_data_result[:, 1]
training_data_desire_output = training_data_result[:, 2]

output_threshold = tf.Variable(ot, dtype=tf.float64)
output_weights = tf.Variable(ow, dtype=tf.float64)
hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
hidden_weights = tf.Variable(hw, dtype=tf.float64)

hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

predict_complete = sess.run([output_layer], {x_placeholder: complete_data})[0]
predict_training = sess.run([output_layer], {x_placeholder: training_data})[0]
if not benign:
    complete_wrong_predict = predict_complete[np.where(predict_complete < label - 0.5)].shape[0] + predict_complete[np.where(predict_complete > label + 0.5)].shape[0]
    training_wrong_predict = predict_training[np.where(predict_training < label - 0.5)].shape[0] + predict_training[np.where(predict_training > label + 0.5)].shape[0]
else:
    complete_wrong_predict = predict_complete[np.where(predict_complete > 0.5)].shape[0] - predict_complete[np.where(predict_complete > 4.5)].shape[0]
    training_wrong_predict = predict_training[np.where(predict_training > 0.5)].shape[0] - predict_training[np.where(predict_training > 4.5)].shape[0]

print("all data predict wrong count: "+str(complete_wrong_predict))
print("training data predict wrong count: "+str(training_wrong_predict))
print("testing data predict wrong count: "+str(complete_wrong_predict - training_wrong_predict))
