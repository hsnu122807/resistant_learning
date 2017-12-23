import tensorflow as tf
import numpy as np
import os
import sys

input_file_name = sys.argv[1]
# input_file_name = "owl範例輸入"

dir_path = os.path.dirname(os.path.realpath(__file__))

file_input = np.loadtxt(input_file_name+".txt", dtype=str, delimiter=" ")
file_input = file_input[:, :file_input.shape[1]-1]
file_input = np.ndarray.astype(file_input, float)

rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
data_amount_arr = np.array([65, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 39, 100, 40, 100, 100, 100, 100], dtype=str)

# match_arr = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=bool)
# match_count_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)

# for i in range(rule_arr.shape[0]):
#     rule = rule_arr[i]
#     data_amount = data_amount_arr[i]
#     sigma = "2"
#
#     dir_target = dir_path + r"\19_owl_rules"+r"\owl_rule_"+rule+"_"+data_amount+"_and_benign_"+data_amount+"_sigma_"+sigma
#
#     ot = np.loadtxt(dir_target+r"\output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
#     ow = np.loadtxt(dir_target+r"\output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
#     ht = np.loadtxt(dir_target+r"\hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
#     hw = np.loadtxt(dir_target+r"\hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))
#
#     with tf.Graph().as_default():
#         x_placeholder = tf.placeholder(tf.float64)
#
#         output_threshold = tf.Variable(ot, dtype=tf.float64)
#         output_weights = tf.Variable(ow, dtype=tf.float64)
#         hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
#         hidden_weights = tf.Variable(hw, dtype=tf.float64)
#
#         hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
#         output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)
#
#         init = tf.global_variables_initializer()
#         sess = tf.Session()
#         sess.run(init)
#
#         predict_values = sess.run([output_layer], {x_placeholder: file_input})[0]
#         malware_count = predict_values[np.where(predict_values < 0)].shape[0]
#         if malware_count > 0:
#             match_arr[i] = True
#             match_count_arr[i] = malware_count

dir_save_result = dir_path + r"\19_owl_rules\multiple_output_node_bp"
with tf.Graph().as_default():
    x = tf.placeholder(tf.float64, [file_input.shape[0], file_input.shape[1]], name='x_placeholder')
    W = tf.Variable(np.loadtxt(dir_save_result + r"\weight.txt"), dtype=tf.float64, name='weight')
    b = tf.Variable(np.loadtxt(dir_save_result + r"\bias.txt"), dtype=tf.float64, name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

    prediction = tf.argmax(y, 1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    result = sess.run([prediction], {x: file_input})[0]
    print(result[np.where(result != 0)])

# create file to save analyze result
analyze_result = open(dir_path + r"\_"+input_file_name+".txt", 'w', encoding="utf-8")
match_str = ""
for i in range(1, 20):
    if result[np.where(result == i)].shape[0] > 0:
        match_str = match_str + rule_arr[i] + " "
print("Malware "+input_file_name+" matches rule: "+match_str)
analyze_result.writelines("Malware "+input_file_name+" matches rule: "+match_str+"\n")
print("Malware "+input_file_name+" has "+str(file_input.shape[0])+" samples.")
analyze_result.writelines("Malware "+input_file_name+" has "+str(file_input.shape[0])+" samples.\n")
for i in range(20):
    match_count = result[np.where(result == i)].shape[0]
    if i == 0:
        print("Match benign sample amount: " + str(match_count))
        analyze_result.writelines("Match rule benign sample amount: " + str(match_count) + "\n")
    else:
        print("Match rule "+str(i)+" sample amount: "+str(match_count))
        analyze_result.writelines("Match rule "+str(i)+" sample amount: "+str(match_count)+"\n")
