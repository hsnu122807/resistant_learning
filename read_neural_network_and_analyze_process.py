import tensorflow as tf
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

for z in range(1, 20):
    for s in range(3):
        if s == 0:
            testing_data_dir = dir_path + r"\998_malware_2_benign"
            need_to_clear_label = False
        elif s == 1:
            process_type = r"\benign"  # benign amount:2851   malware amount:4763
            testing_data_dir = dir_path + r"\19_owl_rules\testing_data" + process_type + "_process_testing_data"
            need_to_clear_label = True
        else:
            process_type = r"\malware"  # benign amount:2851   malware amount:4763
            testing_data_dir = dir_path + r"\19_owl_rules\testing_data" + process_type + "_process_testing_data"
            need_to_clear_label = True
    # process_type = r"\malware"  # benign amount:2851   malware amount:4763
    # testing_data_dir = dir_path+r"\19_owl_rules\testing_data"+process_type+"_process_testing_data"
    # # testing_data_dir = dir_path + r"\998_malware_2_benign"
    # need_to_clear_label = True
        with tf.Graph().as_default():
            run_slfn = True  # if False, run softmax
            if run_slfn:
                # model_data_dir = dir_path + r"\all_rules_data_sample_1_of_10_bml_separate_benign_and_malicious_2"
                # model_data_dir = dir_path + r"\all_rules_data_sample_100_bml_separate_benign_and_malicious_1"
                model_data_dir = dir_path + r"\19_owl_rules\owl_rule_{0}_all_training_data_bml".format(z)
            else:
                # model_data_dir = dir_path + r"\19_owl_rules\softmax_nn_sampling_bml"
                model_data_dir = dir_path + r"\19_owl_rules\softmax_nn_all_bml"
            print(model_data_dir)
            print(testing_data_dir)

            rule_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=str)
            data_amount_arr = np.array([65, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 39, 100, 40, 100, 100, 100, 100], dtype=str)

            process_name_arr = np.array(['17401', '7470', '13740', '12469', '9410', '477', '15199', '415', '525', '12846', '521', '17305', '563', '12179', '15136', '4847', '2831', '8397', '12937', '7236', '7388', '18099', '3701', '2480', '507', '5805', '13137', '6072', '10068', '3210', '1256', '9963', '14198', '17959', '1406', '9450', '1675', '16971', '684', '50494b0b0b6f3a8e578ddcd0f1b1c720', '16591', '15364', '3629', '4753', '480', '5950', '6926', '569', '7019', '8195', '533', '736', '12021', '10291', '5702', '1415', '5707', '1738', '1163', '8879', '5535', '711', '544', '9531', '6672', '508', '17017', '488', '9701', '7221', '9416', '17814', '2362', '403', '5464', '4829', '1603', '12472', '744', '14412', '4ea35ca7ed4270f416e20a64ce6f9430', '2322', '12890', '16397', '13911', '2890', '3989', '4600', '1714', 'chrome', '2103', '9503', '4feca244317eb906c3a7afe2ad5749d0', '11163', '5446', '2906', '6515', '9722', '16943', '2647', '3901', '8888', '4572', '5439', '9549', '1407', '3972', '11097', '13620', '6803', '12782', '695', '11010', '13211', '14932', '10409', '8148', '3821', '9708', '4315', '11003', '7861', '6059', '17539', '15564', '3853', '8602', '16851', '12724', '4aeade072460ca141f20c2f789990120', '6397', '3130', '7026', '16870', '6294', '18009', '4922', '17913', '7811', '6706', '15486', '407', '7489', '446', '1080', '12166', '15476', '12813', '18153', '4256', '17156', '17362', '15318', '5829', '8540', '9972', '1963', '18222', '731', '13799', '13851', '6901', '3217', '18253', '15356', '9076', '7256', '18187', '15261', '4992', '577', '6975', '3339', '2814', '9896', '5324', '4471', '497', '6911', '1780', '10700', '12732', '559', '1986', '16601', '9478', '2225', '10890', '6181', '15690', '11012', '8766', '8342', '15951', '18231', '15802', '8196', '2077', '3997', '14397', '518', '3108', '15835', '3557', '6610', '15459', '3781', '14611', '17070', '14963', '17461', '11749', '18234', '6956', '9945', '15166', '4781', '8569', '9954', '4780', '15104', '513', '14254', '12192', '405', '17946', '16294', '929', '4795', '15333', '6963', '14633', '13423', '9199', '685', '4933', '15929', '4db7b164c5595d97da7288962771b010', '13078', '11242', '13268', '6297', '94ff771ba3c779392fcfb643c165ed20', '596', '11156', '6954', '10931', '7408', '8991', '9028', '15232', '1135', '2381', '16279', '11984', '12106', '16565', '17325', '7042', '13889', '15012', '8946', '9494', '6961', '9006', '605', '10061', '13186', '9834', '16416', '6165', '9598', '13604', '3991', '5431', '14228', '428', '1367', '8813', '773', '18331', '10733', '801', '542', '5690', '9566', '17749', '613', '12155', '9104', '6815', '13922', '1734', '15269', '18540', '4b84837bb850f7871e3f0a3736022dc0', '18481', '5304', '4895', '4932', '576', '9060', '17001', '11201', '3175', '14712', '1224', '7982', '5436', '4130', '11648', '17308', '7286', '1120', '14515', '15606', '6135', '17988', '4963', '11887', '10624', '18471', '1926', '9866', '11437', '6554', '14671', '6729', '11662', '8867', '16716', '5343', '18044', '11246', '15280', '468', '6624', '4499', '17000', '7608', '18258', '7103', '4465', '7960', '12881', '9004', '10353', '4ad4ad416b7c16ff43b0d3012c6d35e0', '504', '4762', '824', '7572', '9899', '6277', '7793', '12952', '8522', '17845', '1029', '1924', '4312', '4942', '5243', '7997', '4eb6262331ca641b2ad821a76cc62810', '3520', '7988', '13310', '17772', '1565', '1203', '10400', '6759', '12815', '14496', '10942', '18520', '4023', '6735', '2565', '4268', '11539', '8644', '6304', '10239', '16999', '1240', '10440', '15436', '7951', '4826', '11140', '3019', '437', '8571', '561', '17645', '1420', '16417', '11284', '16739', '8130', '17451', '15538', '5430', '13340', '12308', '17524', '17489', '2236', '8908', '2780', '8072', '3697', '8822', '424', '17601', '2237', '13060', '6334', '453', '18219', '1263', '9106', '5888', '9801', '7288', '13052', '5911', '16685', '17590', '9034', '16577', '13316', '10096', '1008', '3858', '4738', '17698', '3571', '4958', '15654', '10934', '2815', '18090', '414', '1574', '11079', '8065', '7536', '8370', '526', '2717', '7444', '10560', '2396', '5490', '587', '10583', '460', '9848', '2646', '15990', '7945', '12902', '11317', '9383', '12712', '8081', '12653', '5276', '5194', '15175', '15821', '4680', '4318', '11149', '18458', '17873', '16335', '3163', '5997', '8221', '8533', '4019', '6906', '2524', '13956', '2639', '6606', '11678', '18552', '16200', '2199', '7731', '17234', '7601', '12235', '14878', '17351', '6295', '8339', '11849', '645', '17722', '10035', '10669', '833', '1ca95b8ae0d19ca7d9f22ca89ea92e10', '9894', '4459', '4345', '594', '1791', '7537', '1bd13e9dd95135e7f2700c8c942c3a60', '8740', '10151', '12448', '8560', '3912', '5839', '3115', '2794', '14238', '17952', '4072', '9245', '12583', '4399', '16000', '15337', '8426', '11771', '18306', '10660', '6025', '9310', '8217', '11585', '9915', '16717', '2129', '3065', '7587', '512', '9011', '7110', '12791', '2640', '1385', '15740', '12526', '4fec45dd99468089b8fc2e297894d830', '14097', '8119', '17224', '7186', '10213', '11314', '3745', '467', '444', '16539', '12185', '15779', '6458', '10907', '13405', '13638', '4ad3ba1abeaa1023cdab95ec43bc5540', '589', '16120', '17647', '7862', '11543', '5916', '12503', '4299', '2991', '13403', '3588', '16574', '9995', '8868', '583', '2113', '7515', '3958', '6194', '3658', '492', '17768', '6109', '13444', '1502', '7520', '3106', '10444', '11361', '7483', '1646', '11486', '16580', '2137', '15220', '5654', '7950', '7338', '668', '8981', '4856', '423', '501', '9975', '6539', '532', '2811', '4b40cac1e3291fa6f3c463264729bd21', '10679', '4605', '5995', '4e908daa6066f856b23023a234b886c1', '5826', '12690', '1047', '3938', '11755', '436', '2555', '16990', '11340', '13014', '8955', '16490', '8768', '10198', '1652', '7790', '2053', '12151', '12649', '11128', '8852', '12581', '14564', '10153', '11971', '4246', '8302', '15559', '17514', '498', '11786', '18198', '487', '11041', '15346', '3421', '5804', '6057', '15184', '572', '10139', '10935', '6745', '10238', '18502', '6238', '15097', '3816', '3966', '3848', '1321', '16262', '3375', '9126', '4691', '15105', '11532', '14151', '4316', '3623', '16291', '9465', '434', '14899', '1869', '12336', '4723', '4ad126c86e289b405cadad024bc9ec20', '4308', '2437', '6192', '1352', '14416', '4db956d8e1aa599b2c268dc113273ff0', '14346', '4837', '12884', '13056', '6597', '12545', '10570', '11153', '17954', '18216', '16133', '10768', '10264', '5503', '6347', '11382', '7352', '15597', '3174', '16364', '18347', '1afc0bebb7a87a7fc65f57061633eec0', '13320', '478', '12730', '11958', '11654', '547', '11215', '3800', '10401', '503', '9348', '17240', '17297', '11852', '9467', '7831', '10816', '15245', '519', '16093', '842', '4988', '4494', '17118', '12355', '1003', '12381', '8506', '11398', '3957', '4ad5351550eaced54fca34c8fd30f5c0', '3100', '2158', '9222', '1600', '6662', '9376', '3693', '443', '16701', '10770', '8647', '4719', '12427', '10881', '8999', '5962', '5391', '4321', '15893', '933', '14909', '9124', '14786', '7851', '9366', 'filezilla', '3599', '17079', '14927', '17857', '13247', '2728', '7617', '14035', '15327', '482', '1427', '458', '2642', '433', '17706', '10372', '12130', '18437', '12196', '9739', '16354', '9020', '484', '6899', '2670', '9792', '1479', '17409', '4967', '13561', '4044', '1098', '15540', '2710', '6785', '8177', '6411', '8330', '15323', '17883', '16233', '7050', '10478', '8142', '13782', '10655', '15173', '5633', '16424', '12402', '17267', '9160', '15937', '17058', '15101', '13870', '17316', '4229', '9433', '14527', '3620', '13667', '2264', '14762', '4032', '2417', '9219', '15341', '14629', '3631', '4722', '5037', '6440', '14907', '7183', '16285', '5584', '9819', '4636', '1178', '688', '15188', '10753', '7632', '570', '2176', '502', '6078', '11089', '15252', '538', '17780', '5953', '4412', '9470', '2729', '5588', '5302', '4735', '5077', '15532', '14678', '4007', '554', '791', '14218', '10962', '963', '8688', '2326', '5305', '6270', '562', '12964', '14376', '18131', '3923', '510', '13931', '15381', '5841', '14627', '1283', '8472', '3326', '12229', '9231', '10602', '14505', '15920', '10121', '1164', '17446', '16059', '2102', '2947', '15034', '6342', '16660', '17334', '1886', '13769', '8080', '1455', '12171', '9341', '7640', '9046', '9816', '5711', '438', '2953', '13903', '17104', '9407', '13526', '14565', '15687', '640', '16979', '9265', '4b79cc552737163cbc60e9a233892340', '13768', '12652', '15409', '1433', '4705', '6403', '11998', '11337', '5188', '7208', '17652', '8551', '8382', '3371', '8064', '13746', '6613', '9699', '13225', '7549', '11196', '432', '7102', '94ff28682758187936d81540cf095380', '6920', '13123', '5776', '2523', '12465', '8896', '11753', '2051', '6386', '528', '6898', '15291', '11624', '9186', '5523', '14925', '2112', '454', '529', '14392', '3785', '11983', '9619', '9032', '8357', '14316', '16110', '11613', '13971', '1768', '16998', '12343', '500', '5725', '4752', '15167', '3640', '1598'], dtype=str)
            process_count_arr = np.array([968, 1130, 1108, 1999, 526, 301, 1070, 1692, 3499, 420, 1771, 498, 3463, 338, 1602, 1029, 1006, 1860, 1011, 1533, 1236, 1705, 295, 1140, 3441, 377, 1049, 1053, 336, 989, 1105, 456, 1028, 941, 990, 1002, 1556, 1617, 814, 2169, 1006, 1684, 1502, 1566, 7437, 364, 511, 1573, 1566, 1027, 395, 340, 1051, 1547, 1121, 1032, 1409, 502, 1682, 1053, 1018, 1810, 529, 559, 1098, 3410, 1088, 1945, 989, 367, 1772, 1054, 977, 6880, 538, 1044, 1178, 1074, 1032, 1025, 1855, 526, 1125, 1597, 1569, 1018, 984, 421, 1765, 1799, 430, 1552, 1607, 1679, 1044, 2098, 988, 1097, 1001, 1117, 954, 1101, 1062, 1107, 1024, 393, 1162, 1052, 333, 948, 1012, 1063, 3596, 1078, 1223, 1287, 1816, 447, 1012, 1029, 1574, 1195, 980, 1013, 1218, 344, 1195, 1351, 1011, 602, 1135, 1047, 1543, 995, 948, 1045, 1147, 991, 1022, 1086, 406, 7401, 1466, 372, 1077, 1133, 1009, 1050, 1255, 1815, 1522, 959, 1560, 1796, 1300, 1114, 867, 433, 1130, 982, 1055, 1599, 1534, 1079, 1069, 392, 1124, 467, 1018, 1000, 3563, 1091, 1026, 2153, 1030, 1044, 564, 7019, 1185, 1535, 1075, 1065, 425, 1086, 1530, 355, 1803, 1056, 1000, 1018, 2057, 1059, 1761, 395, 492, 1039, 1272, 970, 1008, 1074, 303, 1023, 373, 1164, 1106, 1552, 516, 1038, 374, 1673, 988, 994, 1255, 1031, 1048, 987, 1881, 583, 1113, 396, 1151, 3392, 1060, 982, 1965, 1015, 992, 1263, 1012, 419, 1099, 1004, 1059, 381, 1027, 483, 1148, 2774, 1763, 1084, 501, 496, 1688, 1806, 1098, 1523, 1001, 1736, 1513, 1008, 1565, 401, 1041, 1261, 1234, 433, 1054, 1549, 1009, 1122, 1151, 337, 986, 1000, 310, 1871, 1017, 1271, 632, 423, 1025, 1066, 1013, 1203, 1731, 424, 2045, 1622, 1174, 547, 1170, 339, 988, 2124, 608, 589, 1630, 1636, 425, 1503, 1084, 1095, 1115, 1688, 2101, 1853, 1365, 979, 1082, 442, 289, 992, 1657, 991, 1006, 425, 1238, 1012, 1187, 403, 1073, 1006, 1508, 1584, 331, 1255, 1022, 996, 1087, 1266, 1063, 1720, 1173, 1066, 1093, 1984, 982, 976, 985, 444, 1045, 373, 1091, 1173, 587, 520, 966, 1072, 1535, 1063, 1032, 1783, 1170, 1533, 1714, 319, 979, 1661, 1369, 1499, 1532, 1156, 486, 1029, 1057, 982, 423, 344, 494, 581, 2200, 1202, 1303, 927, 1865, 1776, 1073, 949, 1040, 1122, 1008, 1111, 491, 354, 1266, 1113, 996, 411, 1085, 1009, 1524, 1065, 1556, 395, 595, 1043, 1015, 397, 531, 1736, 1421, 1535, 369, 386, 1704, 2011, 983, 1384, 1123, 1014, 1080, 478, 416, 1721, 294, 1000, 439, 1086, 1008, 1005, 1073, 1024, 1007, 382, 977, 1920, 1435, 1663, 313, 1528, 1936, 1781, 1156, 984, 496, 1018, 342, 758, 420, 1012, 1039, 302, 381, 1115, 2149, 1003, 3689, 430, 995, 1002, 517, 496, 572, 382, 1014, 414, 1249, 1078, 1019, 1042, 1023, 1604, 1535, 1092, 1024, 999, 937, 1740, 1756, 7163, 417, 1623, 457, 1023, 2022, 1513, 1119, 1181, 358, 2184, 977, 1739, 1035, 1093, 1015, 358, 1189, 474, 1047, 1680, 1640, 369, 507, 1613, 991, 1034, 1786, 493, 1723, 3487, 924, 1017, 1027, 1594, 588, 363, 1002, 1017, 385, 411, 2234, 1766, 1185, 1494, 2035, 1024, 388, 1048, 3398, 1329, 455, 1132, 1702, 1691, 1109, 1550, 1166, 1003, 1299, 1538, 331, 2120, 2056, 383, 1571, 487, 1076, 1028, 1010, 471, 1059, 1081, 628, 1205, 4047, 1510, 489, 1018, 1011, 542, 874, 664, 1740, 1071, 1051, 3429, 1108, 344, 1076, 1114, 1611, 1086, 992, 3516, 1032, 1108, 1201, 328, 1154, 380, 956, 550, 7320, 489, 587, 982, 363, 1023, 1063, 1585, 1722, 361, 1033, 1002, 1762, 439, 1061, 939, 2193, 1043, 332, 571, 1202, 1284, 1578, 1981, 322, 1108, 415, 458, 1570, 1597, 391, 1039, 1148, 979, 3507, 662, 493, 1007, 2161, 1895, 550, 1133, 1496, 538, 1536, 1144, 3620, 461, 1048, 338, 1677, 1944, 997, 3564, 268, 990, 288, 1006, 989, 1465, 1808, 1768, 392, 379, 1090, 1535, 7060, 1741, 1555, 1195, 1883, 1126, 358, 1557, 1016, 1042, 587, 1747, 1540, 1573, 1020, 1036, 1100, 435, 1123, 390, 1083, 1471, 1054, 1724, 7298, 3672, 1078, 7330, 1040, 2213, 1052, 1431, 996, 1261, 3512, 1107, 1608, 1540, 445, 486, 999, 1017, 1200, 1114, 500, 988, 1023, 353, 1573, 1019, 1665, 1046, 1181, 534, 1186, 479, 1022, 449, 1063, 527, 995, 1768, 1753, 390, 1015, 1718, 1028, 1034, 972, 1720, 1521, 577, 1111, 335, 296, 1971, 2637, 1032, 1009, 1143, 1176, 1913, 418, 1106, 1007, 1538, 354, 1023, 1014, 1033, 1720, 1486, 7354, 1535, 1031, 1125, 3585, 380, 373, 1109, 1697, 3677, 1231, 1018, 1068, 1068, 920, 475, 1531, 1661, 1015, 1738, 974, 1034, 3695, 871, 1937, 1154, 1020, 1113, 1087, 1926, 3707, 1019, 445, 1095, 988, 1075, 984, 7463, 1744, 423, 1032, 1015, 1150, 982, 357, 1120, 380, 1036, 3645, 1674, 867, 999, 420, 1178, 1060, 2270, 1042, 1069, 1047, 985, 1043, 997, 2054, 1018, 1256, 2039, 1914, 2011, 397, 1761, 1082, 1094, 1439, 442, 911, 1033, 1572, 1758, 315, 1614, 1039, 1063, 1568, 1095, 436, 1124, 377, 314, 533, 1012, 409, 1029, 1842, 1791, 1079, 1703, 1113, 2177, 1009, 1019, 1036, 986, 432, 429, 408, 3597, 1220, 1068, 1050, 1110, 1062, 1019, 475, 1026, 1629, 1070, 1009, 1120, 1765, 1283, 1096, 579, 654, 469, 991, 1715, 1105, 1147, 472, 1026, 366, 1063, 1090, 304, 1057, 1058, 1033, 2144, 1056, 1015, 391, 273, 421, 1701, 362, 979, 1723, 1130, 974, 1106, 369, 1053, 860, 1224, 1604, 1096, 388, 1604, 1027, 279, 1460, 1726, 1122, 1012, 1105, 2007, 340, 1025, 1766, 1105, 359, 1083, 1010, 224, 1000, 1105, 1546, 312, 379, 1076, 1098, 1043, 363, 1575, 1036, 1022, 1551, 1603, 1511, 1086, 1042, 1549, 1191, 466, 1754, 1542, 258, 1075, 1187, 1017, 1024, 1175, 1627, 1611, 1086, 1028, 425, 1122, 955, 1019, 510, 1013, 1017, 1200, 1528, 1018, 4256, 6681, 1031, 1020, 1079, 1507, 1268, 1059, 1720, 1049, 1152, 1722, 1093, 1124, 363, 1539, 1082, 965, 1770, 3553, 1120, 1545, 1310, 452, 1104, 1750, 1012, 522, 1154, 1083, 3671, 492, 1690, 1528, 1116, 1624, 1866, 976, 370, 1755, 1834, 1125, 1698, 2062, 1856, 1041, 1087, 1174, 1161, 1851, 1558, 1018, 1607, 1209, 1038, 1011, 1204, 526, 1557, 1664, 348, 1122, 1017, 1018], dtype=int)
            x_placeholder = tf.placeholder(tf.float64)
            if run_slfn:
                ot = np.loadtxt(model_data_dir + r"\two_class_output_threshold.txt", dtype=float, delimiter=" ").reshape(1)
                ow = np.loadtxt(model_data_dir + r"\two_class_output_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, 1))
                ht = np.loadtxt(model_data_dir + r"\two_class_hidden_threshold.txt", dtype=float, delimiter=" ").reshape((1, -1))
                hw = np.loadtxt(model_data_dir + r"\two_class_hidden_neuron_weight.txt", dtype=float, delimiter=" ").reshape((-1, ht.shape[1]))
                output_threshold = tf.Variable(ot, dtype=tf.float64)
                output_weights = tf.Variable(ow, dtype=tf.float64)
                hidden_thresholds = tf.Variable(ht, dtype=tf.float64)
                hidden_weights = tf.Variable(hw, dtype=tf.float64)
                hidden_layer = tf.tanh(tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds))
                output = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

                file = open(model_data_dir + r"\_two_class_training_detail.txt")

                line_index = 0
                while 1:
                    line_index += 1
                    line = file.readline()
                    if not line:
                        break
                    if "classify middle point((alpha+beta)/2): " in line:
                        middle_point = float(line.replace("classify middle point((alpha+beta)/2): ", ""))
                        break
            else:
                W = tf.Variable(np.loadtxt(model_data_dir + r"\two_class_weight.txt"), dtype=tf.float64, name='weight')
                b = tf.Variable(np.loadtxt(model_data_dir + r"\two_class_bias.txt"), dtype=tf.float64, name='bias')  # true benign rate: 0/2851
                output = tf.nn.softmax(tf.matmul(x_placeholder, W) + b, name='predict_y')
                y_ = tf.placeholder(tf.float64, name='y_placeholder')
                correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1)), tf.int32))

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            benign_process_count = 0
            mal_process_count = 0
            total_process_count = 0
            total_mal_sample_count = 0
            total_benign_sample_count = 0
            total_sample_count = 0
            for filename in os.listdir(testing_data_dir):
                total_process_count += 1
                if need_to_clear_label:
                    input_process = np.loadtxt(testing_data_dir + "/" + filename, dtype=str, delimiter=" ")
                    input_process = input_process[:, :input_process.shape[1]-1]
                    input_process = np.ndarray.astype(input_process, float)
                else:
                    input_process = np.loadtxt(testing_data_dir + "/" + filename, dtype=float, delimiter=" ")
                sample_amount = input_process.shape[0]
                predict_output = sess.run([output], feed_dict={x_placeholder: input_process})[0]
                mal_sample_amount = 0
                benign_sample_amount = 0
                if run_slfn:
                    mal_sample_amount = predict_output[np.where(predict_output < middle_point)[0]].shape[0]
                    benign_sample_amount = sample_amount - mal_sample_amount
                else:
                    y_benign = np.zeros(20)
                    y_benign[0] = 1
                    y_benign_arr = np.tile(y_benign, input_process.shape[0]).reshape(-1, 20)
                    benign_sample_amount = sess.run([correct_prediction], feed_dict={x_placeholder: input_process, y_: y_benign_arr})[0]
                    mal_sample_amount = sample_amount - benign_sample_amount
                # print("mal sample amount: {0}, benign sample amount: {1}, total sample amount:{2}".format(mal_sample_amount, benign_sample_amount, sample_amount))
                total_mal_sample_count += mal_sample_amount
                total_benign_sample_count += benign_sample_amount
                total_sample_count += sample_amount
                if mal_sample_amount == 0:
                    benign_process_count += 1
                else:
                    mal_process_count += 1
            print("mal process: {0}\nbenign process: {1}\ntotal process:{2}".format(mal_process_count, benign_process_count, total_process_count))
            print("total mal sample: {0}\ntotal benign sample: {1}\ntotal sample: {2}".format(total_mal_sample_count, total_benign_sample_count, total_sample_count))
            print("-"*15)
