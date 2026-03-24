from data_dict import *
from data_dict_addition import *
import jieba
from processData import *

def process_sub(data_list):
    src_ids = []
    tgt_ids = []
    max_len = 0
    for en_t, cn_t in data_list:
        en_ts = en_t.split()
        cn_ts = list(cn_t)
        if len(cn_ts) > max_len: max_len = len(cn_ts)
        if len(en_ts) > max_len: max_len = len(en_ts)
        src_ids.extend(cn_ts)
        tgt_ids.extend(en_ts)
    cn_set = set(src_ids)
    en_set = set(tgt_ids)

    print(f"Sentence Max Length: {max_len}")
    return cn_set, en_set

def process_data():
    intersection_set = set(demo_data_train) & set(demo_data_test)
    if not intersection_set:
        print(f"demo_data_test has same item: {demo_data_test}")

    cn_set_train, en_set_train = process_sub(demo_data_train)
    cn_set_test, en_set_test = process_sub(demo_data_test)

    if cn_set_test.issubset(cn_set_train):
        print("cn_set_test OK")
    else:
        print("cn_set_test NOT OK")

    if en_set_test.issubset(en_set_train):
        print("en_set_test OK")
    else:
        print("en_set_test NOT OK")

    id = CUS_START_ID
    t_dict=dict()
    for i in cn_set_train:
        t_dict[i] = id
        id += 1

    for j in en_set_train:
        t_dict[j] = id
        id += 1

    print("CN set: ")
    print(cn_set_train)
    print("EN set: ")
    print(en_set_train)
    print("VOCAB Dict:")
    print(t_dict)


def process_data_addition():
    demo_data = []
    demo_data.extend(demo_data1)
    demo_data.extend(demo_data2)
    demo_data.extend(demo_data3)
    demo_data.extend(demo_data4)
    demo_data.extend(demo_data5)
    demo_data.extend(demo_data6)
    print(f'Demo length: {len(demo_data)}')
    src_ids = []
    tgt_ids = []
    max_len = 0
    for en_t, cn_t in demo_data:
        # cn_ts1 = jieba.lcut(cn_t, cut_all=False)
        cn_ts2 = list(cn_t)
        en_ts = re_eng(en_t)
        # if len(list(cn_ts1)) > max_len: max_len = len(list(cn_ts1))
        if len(cn_ts2) > max_len: max_len = len(cn_ts2)
        if len(en_ts) > max_len: max_len = len(en_ts)
        # src_ids.extend(cn_ts1)
        src_ids.extend(cn_ts2)
        tgt_ids.extend(en_ts)

    cn_set = set(src_ids)
    en_set = set(tgt_ids)

    id = CUS_START_ID
    t_dict = dict()
    for i in cn_set:
        t_dict[i] = id
        id += 1

    for j in en_set:
        t_dict[j] = id
        id += 1

    print("CN set: ")
    print(cn_set)
    print("EN set: ")
    print(en_set)
    print("VOCAB Dict:")
    print(t_dict)
    print(f"Sentence Max Length: {max_len}")

def process_data2():
    cn_list=list(demo_data)
    cn_set = set(cn_list)

    id = CUS_START_ID
    t_dict=dict()
    for i in cn_set:
        t_dict[i] = id
        id += 1

    print("CN set: ")
    print(cn_set)
    print("VOCAB Dict:")
    print(t_dict)

    id_list = []
    for i in cn_list:
        id_list.append(token2idx[i])

    print("CN list: ")
    print(cn_list)
    return id_list

def process_data3():
    t_list = []
    for t in demo_data_cn:
        for i in list(t):
            t_list.append(i)

    for e in demo_data_en:
        for j in e.split():
            t_list.append(j)

    print("t_list:")
    print(t_list)

    t_set = set(t_list)
    print("t_set:")
    print(t_set)

    id = CUS_START_ID
    t_dict = dict()
    for i in t_set:
        t_dict[i] = id
        id += 1

    print("VOCAB Dict:")
    print(t_dict)

def process_ori():
    ori_list = oritext.split('\n')
    print("ori_list:")
    print(ori_list)
    en_cn = True
    en_s = ""
    out_list = []
    for s in ori_list:
        if not s: continue
        if s.strip()[0].isdigit():
            print(f"例句：{s}")
            continue
        if en_cn:
            en_s = s
            en_cn=False
        else:
            cn_s = s
            en_cn_pair = (en_s, cn_s)
            print(f"en_cn_pair：{en_cn_pair}")
            en_cn = True
            out_list.append(en_cn_pair)

    print("out_list:")
    print(out_list)

def tmp_test():
    test_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(sum(test_list[-5:]) / 5)
    print(test_list[-6:-1])
    print(test_list[-2])
    pass

if __name__ == '__main__':
    #process_ori()
    process_data()
    #process_data_addtion()
    #tmp_test()