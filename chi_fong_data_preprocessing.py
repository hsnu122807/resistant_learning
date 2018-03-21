import os
import pickle
import numpy as np

is_benign = False
if is_benign:
    DATA_DIR = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\benign_testing_data"
    # file = open("rnn_input.pickle", "rb")
else:
    DATA_DIR = r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\malware_testing_data"
    file = open("rnn_input.pickle", "rb")
diction = pickle.load(file)
# print(len(diction))
save_dict = {}
for filename in os.listdir(DATA_DIR):
    temp_ls = np.zeros(1000)
    sample_cluster_ls = diction[filename][0]
    # print(sample_cluster_ls.shape[0])
    for i in sample_cluster_ls:
        temp_ls[int(i*1000)] += 1
    # print(sample_cluster_ls)
    save_dict[filename] = temp_ls
    # print(save_dict[filename])
    # input(123)
file.close()
if is_benign:
    with open(r'C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\2851_benign_ghsom_distribution.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
else:
    with open(r'C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\4763_mal_ghsom_distribution.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
