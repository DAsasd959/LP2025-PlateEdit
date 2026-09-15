# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""

import numpy as np
import random

def count_license_plate_chars():

    with open('/home/avlab/scenetext/SRNet-Datagen_LP2022/datasets/imglist_test.txt','r',encoding='utf-8') as f:
        data_list = f.readlines()

    label = []

    for i in data_list:
        label_name = i.split('\t')[1].strip('\n')
        label.append(label_name)
        char_counts = {}

    for plate in label:
        plate_length = str(len(plate))
        if plate_length not in char_counts:
            char_counts[plate_length] = [{"characters": {}, "total": 0} for _ in range(len(plate))]
        
        for i, char in enumerate(plate):
            if char.isalnum():  # 统计字母和数字字符
                char = char.upper()  
                if char in char_counts[plate_length][i]["characters"]:
                    char_counts[plate_length][i]["characters"][char] += 1
                else:
                    char_counts[plate_length][i]["characters"][char] = 1
                char_counts[plate_length][i]["total"] += 1
    # 按照每个位置的字符数量进行排序
    for plate_length, char_count_list in char_counts.items():
        for char_count in char_count_list:
            char_count["characters"] = dict(sorted(char_count["characters"].items(), key=lambda x: x[1], reverse=True))


    # 计算每个位置每个字符的比例
    for plate_length, char_count_list in char_counts.items():
        for char_count in char_count_list:
            total_chars = char_count["total"]
            for char in char_count["characters"]:
                count = char_count["characters"][char]
                proportion = count / total_chars
                char_count["characters"][char] = {"count": count, "proportion": proportion}

    
    return char_counts

# 测试代码
# license_plates = ["2B4499", "AGS1303", "TDF5710", "BFZ5206", "PZ9726", "P28499", "MDA8913", "MBY7528", "AKR0317", "BFX6810", "39476", "F38676", "BVY332"]
# result = count_license_plate_chars(license_plates)


result = count_license_plate_chars()
license7 = result['7']
list7 = []
for loc in license7:
    loc_list = []
    character_list = []
    proportion = []
    for k,v in loc['characters'].items():
        character_list.append(k)
        proportion.append(v['proportion'])
    loc_list.append(character_list)
    loc_list.append(proportion)
    list7.append(loc_list)

license6 = result['6']
list6 = []
for loc in license6:
    loc_list = []
    character_list = []
    proportion = []
    for k,v in loc['characters'].items():
        character_list.append(k)
        proportion.append(v['proportion'])
    loc_list.append(character_list)
    loc_list.append(proportion)
    list6.append(loc_list)

license5 = result['5']
list5 = []
for loc in license5:
    loc_list = []
    character_list = []
    proportion = []
    for k,v in loc['characters'].items():
        character_list.append(k)
        proportion.append(v['proportion'])
    loc_list.append(character_list)
    loc_list.append(proportion)
    list5.append(loc_list)
        

def random_plate(text_len):

    plate_list = []
    letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'

    numbers = '0123456789'
    # numbers_wo0 = '123456789'
    def get_plate_7num():
        #generate 3 randomly chosen letters, L1, L2, L3
        
        plate = []
        #generate 4 randomly chosen numbers, N1, N2, N3, N4
        for i in range(7): 
            char,ratio = list7[i]     
            if i == 3:
                plate.append('·')
            
            text = np.random.choice(char, 1, p= ratio)[0]
            if text == 'I':
                text = '1' 
            elif text == 'O':
                text = '0' 
            plate.append(text)
            
        
        plate = ''.join(plate)

        return plate


    def get_plate_6num():
        #generate 3 randomly chosen letters, L1, L2, L3
        
        plate = []
        check_list = ['000·111','111·000','0000·11','0000·10','0000·01','11·0000','10·0000','01·0000']
        check_list = list(np.random.choice(check_list))
        insert_index = check_list.index('·')
        check_list.pop(insert_index)


        for i in range(6): 
            char,ratio = list6[i]   
            check_value = check_list[i]  
            
            text = np.random.choice(char, 1, p= ratio)[0] 
            if check_value == '1':
                while text.isdigit():
                    text = np.random.choice(char, 1, p= ratio)[0] 
            elif check_value == '0':
                while text.isalpha():
                    text = np.random.choice(char, 1, p= ratio)[0]
            if text == 'I':
                text = '1' 
            elif text == 'O':
                text = '0' 

            plate.append(text[0])
        
        plate.insert(insert_index,'·')
        plate = ''.join(plate)

        

        return plate

    def get_plate_5num(): # 針對舊式車牌(AB-001)
        #generate 3 randomly chosen letters, L1, L2, L3
        
        plate = []
        check_list = ['000-11','11-000','000-01','000-10','10-000','01-000']
        check_list = list(np.random.choice(check_list))
        insert_index = check_list.index('-')
        check_list.pop(insert_index)


        for i in range(5): 
            char,ratio = list5[i]   
            check_value = check_list[i]  
            
            text = np.random.choice(char, 1, p= ratio)[0] 
            if check_value == '1':
                while text.isdigit():
                    text = np.random.choice(char, 1, p= ratio)[0] 
            elif check_value == '0':
                while text.isalpha():
                    text = np.random.choice(char, 1, p= ratio)[0]
            if text == 'I':
                text = '1' 
            elif text == 'O':
                text = '0' 

            plate.append(text[0])
        
        plate.insert(insert_index,'-')
        plate = ''.join(plate)

        return plate

    def get_plate_4num(): # 針對舊式車牌(00-BA)
        #generate 3 randomly chosen letters, L1, L2, L3
        plate = []
        # insert = np.random.randint(2,3)
        for i in range(4):
            text_choice = random.choice(['char','num'])
            if text_choice == 'char':
                text = random.choice(letters)
            else:
                text = random.choice(numbers)
            plate.append(text)
            if i == 1:
                plate.append('·')
        plate = ''.join(plate)

        return plate

    # random_return_num = random.choice([4,5,6,7])  # 隨機選一個return

    # if   (text_len == 4):
    #     return  get_plate_4num()    #針對舊式車牌(01-BA)
    if (text_len == 5):  
        return  get_plate_5num()    #針對新式車牌(BA-001),(001-BA)
    elif (text_len == 6): 
        return  get_plate_6num()    #針對舊式車牌(AB0-001),(AB-0001),(0001-BA)
    elif (text_len == 7):  
        return  get_plate_7num()    #針對舊式車牌(ABC-0001)
