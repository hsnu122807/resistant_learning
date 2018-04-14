import os
import pickle
import numpy as np
import random


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def gen_a_distribution():
    result_arr = np.zeros(48)
    for i in range(1700):
        index = random.randint(0, 46)
        result_arr[index] += 1
    return result_arr

file = open(r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\4763_mal_ghsom_distribution_dim_48.pickle", "rb")
# file = open(r"C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\2851_benign_ghsom_distribution_dim_48.pickle", "rb")
diction = pickle.load(file)
file.close()

distribution_set = set()
for k, v in diction.items():
    distribution_set.add(to_tuple(v))
# check if the distribution duplicated, answer: both B&M no any distribution pattern duplicated
print(len(distribution_set))  # 4763
input(1)

# the sum of distribution samples of a process is 1700
# maintain the 'have_value_dic' to decrease the dimension and learn faster
have_value_dic = {110: 0, 120: 1, 130: 2, 140: 3, 150: 4, 160: 5, 210: 6, 220: 7, 231: 8, 232: 9, 233: 10, 234: 11, 235: 12, 236: 13, 240: 14, 250: 15, 261: 16, 262: 17, 263: 18, 264: 19, 265: 20, 266: 21, 310: 22, 320: 23, 330: 24, 340: 25, 350: 26, 360: 27, 411: 28, 412: 29, 413: 30, 414: 31, 415: 32, 416: 33, 420: 34, 431: 35, 432: 36, 433: 37, 434: 38, 435: 39, 436: 40, 441: 41, 442: 42, 443: 43, 444: 44, 445: 45, 446: 46, 0: 47}

# random generate 4763 random sample, the distribution pattern would not duplicate
gen_counter = 0
temp_set_len = len(distribution_set)
random_gen_dic = {}
while gen_counter < 4763:
    add_fail = True
    while add_fail:
        test_distribution = gen_a_distribution()
        distribution_set.add(to_tuple(test_distribution))
        if len(distribution_set) > temp_set_len:
            add_fail = False
            temp_set_len = len(distribution_set)
            random_gen_dic[gen_counter] = test_distribution
    gen_counter += 1
    print(gen_counter)

with open(r'C:\Users\user\PycharmProjects\autoencoder\resistant_learning\chi_fong_2851_benign_4763_mal_process\4763_not_mal_random_ghsom_distribution_dim_48.pickle', 'wb') as handle:
    # gen distribution dictionary key would be int 0-4762
    pickle.dump(random_gen_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
