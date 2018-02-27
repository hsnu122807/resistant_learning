import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# input_file_name = sys.argv[1]
# input_file_name = "owl範例輸入"
# input_file_name = "benign_samples"

# dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_binary"
dir_save_result = dir_path + r"\19_owl_rules\softmax_nn_all_envelope"
x = tf.placeholder(tf.float64, name='x_placeholder')
W = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
b = tf.Variable(np.loadtxt(dir_save_result + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')
y = tf.nn.softmax(tf.matmul(x, W) + b, name='predict_y')

prediction = tf.argmax(y, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

file_amount_count = 0
benign_amount_count = 0
malware_amount_count = 0
sample_amount = 0
classify_as_benign_sample_amount = 0
rule_match_amount_arr = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=int)

DATA_DIR = r"C:\Users\user\Desktop\chifeng_data\benign_testing_data"
# DATA_DIR = r"C:\Users\user\Desktop\chifeng_data\malware_testing_data"
for filename in os.listdir(DATA_DIR):
    file_amount_count += 1
    file_input = np.loadtxt(DATA_DIR+"/"+filename, dtype=str, delimiter=" ")
    file_input = file_input[:, :file_input.shape[1]-1]
    file_input = np.ndarray.astype(file_input, float)
    sample_amount += file_input.shape[0]

    rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
    result = sess.run([prediction], {x: file_input})[0]

    match_str = ""
    is_mal_process = False
    for i in range(1, 20):
        if result[np.where(result == i)].shape[0] > 0:
            match_str = match_str + rule_arr[i - 1] + " "
            rule_match_amount_arr[i-1] += 1
            is_mal_process = True

    if is_mal_process:
        malware_amount_count += 1
    else:
        benign_amount_count += 1

    classify_as_benign_sample_amount += result[np.where(result == 0)].shape[0]

    # print(filename + " matches rule: " + match_str)
    # print(filename + " has " + str(file_input.shape[0]) + " samples.")

    # analyze_result.writelines(filename + " matches rule: " + match_str+"\n")
    # analyze_result.writelines(filename + " has "+str(file_input.shape[0])+" samples.\n")
    # for i in range(0, 20):
    #     match_count = result[np.where(result == i)].shape[0]
    #     if i == 0:
    #         print("Match benign sample amount: " + str(match_count))
    #         # analyze_result.writelines("Match rule benign sample amount: " + str(match_count) + "\n")
    #     else:
    #         print("Match rule " + str(i) + " sample amount: " + str(match_count))
    #         # analyze_result.writelines("Match rule "+str(i)+" sample amount: "+str(match_count)+"\n")

print("total: "+str(file_amount_count)+" files.")
print("pure benign process count: "+str(benign_amount_count))
print("malicious process count: "+str(malware_amount_count))
print("benign rate: {0}/{1}".format(classify_as_benign_sample_amount, sample_amount))
for i in range(rule_match_amount_arr.shape[0]):
    print("Rule "+str(i+1)+" match "+str(rule_match_amount_arr[i])+" process.")

