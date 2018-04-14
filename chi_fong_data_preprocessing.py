import os
import pickle
import numpy as np

is_benign = False
cluster_arr = np.zeros(1000)
have_value_dic = {110: 0, 120: 1, 130: 2, 140: 3, 150: 4, 160: 5, 210: 6, 220: 7, 231: 8, 232: 9, 233: 10, 234: 11, 235: 12, 236: 13, 240: 14, 250: 15, 261: 16, 262: 17, 263: 18, 264: 19, 265: 20, 266: 21, 310: 22, 320: 23, 330: 24, 340: 25, 350: 26, 360: 27, 411: 28, 412: 29, 413: 30, 414: 31, 415: 32, 416: 33, 420: 34, 431: 35, 432: 36, 433: 37, 434: 38, 435: 39, 436: 40, 441: 41, 442: 42, 443: 43, 444: 44, 445: 45, 446: 46, 0: 47}

if is_benign:
    DATA_DIR = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\benign_testing_data"
    file = open("rnn_input_benign.pickle", "rb")
else:
    DATA_DIR = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\malware_testing_data"
    file = open("rnn_input.pickle", "rb")
diction = pickle.load(file)
# print(len(diction))
save_dict = {}
for filename in os.listdir(DATA_DIR):
    # temp_ls = np.zeros(1000)
    temp_ls = np.zeros(48)
    sample_cluster_ls = diction[filename][0]
    sample_cluster_ls = sample_cluster_ls[:900]
    # print(sample_cluster_ls.shape[0])
    for i in sample_cluster_ls:
        temp_index = int(i * 1000)
        # temp_ls[int(i*1000)] += 1
        temp_ls[have_value_dic[int(i * 1000)]] += 1
        # cluster_arr[temp_index] = int(temp_index)
    save_dict[filename] = temp_ls
    # print(temp_ls)
    # input(123)

file.close()

# # get have value list
# index = 0
# print("{", end='')
# for i in cluster_arr:
#     if i > 0:
#         print(str(int(i))+': '+str(index)+', ', end='')
#         index += 1
# print('}')

if is_benign:
    with open(r'C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\2851_benign_ghsom_distribution_dim_48_len_900.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
else:
    with open(r'C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\4763_mal_ghsom_distribution_dim_48_len_900.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
