from data_dict import *
#from data_dict_addition import *
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import jieba
import re

def re_eng(en_t):
    return re.findall(r"\w+|[^\w\s]", en_t)

def generate_src_mask(src_token_ids, pad_id = PAD_ID):
    src_mask = (src_token_ids == pad_id)
    # [batch_size, 1, 1, src_seq_len] -> [batch_size, n_heads, seq_len_q, seq_len_k]
    src_mask_4d = src_mask.unsqueeze(1).unsqueeze(1)
    logging.debug(f"Mask: {src_mask_4d.shape} -> {src_mask_4d.device}")
    return src_mask_4d

def generate_tgt_mask(tgt_token_ids, pad_id = PAD_ID):
    # [batch_size, tgt_seq_len]
    batch_size, tgt_seq_len = tgt_token_ids.shape

    # ahead mask -> [batch_size, tgt_seq_len, tgt_seq_len]
    ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool), diagonal=1)
    ahead_mask = ahead_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    # pad mask -> [batch_size, tgt_seq_len, tgt_seq_len]
    tgt_pad_mask = (tgt_token_ids == pad_id)
    tgt_pad_mask_3d = tgt_pad_mask.unsqueeze(1).repeat(1, tgt_seq_len, 1)

    # combine
    tgt_mask = ahead_mask | tgt_pad_mask_3d
    # [batch_size, 1, tgt_seq_len, tgt_seq_len] -> [batch_size, n_heads, seq_len_q, seq_len_k]
    tgt_mask_4d = tgt_mask.unsqueeze(1)
    logging.debug(f"Mask: {tgt_mask_4d.shape} -> {tgt_mask_4d.device}")
    return tgt_mask_4d

def sce2id_fillpad(t, is_cn=False, is_cn_j=True):
    if is_cn:
        if is_cn_j:
            tokens = jieba.lcut(t)
        else:
            tokens = list(t)
    else:
        #tokens = re_eng(t)
        tokens = t.split()

    ids = [token2idx.get(token, UNK_ID) for token in tokens]
    ids = [BOS_ID] + ids + [EOS_ID]

    if len(tokens) > MAX_LEN:
        ids = ids[:MAX_LEN]
        ids[-1] = EOS_ID

    padding_length = MAX_LEN - len(ids)
    ids += [PAD_ID] * padding_length

    return ids

def sub_process_data(demo_data):
    src_ids = []
    tgt_ids = []
    for en_t, cn_t in demo_data:
        # cn use jieba
        #src_ids.append(sce2id_fillpad(en_t, False, True))
        #tgt_ids.append(sce2id_fillpad(cn_t, True, True))
        # cn use single word
        src_ids.append(sce2id_fillpad(en_t, False, False))
        tgt_ids.append(sce2id_fillpad(cn_t, True, False))

    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    # 先处理好 tgt change & mask，避免后继在训练逻辑中反复处理
    tgt_input = tgt_tensor[:, :-1]
    tgt_tgt = tgt_tensor[:, 1:]
    src_mask = generate_src_mask(src_tensor)
    tgt_mask = generate_tgt_mask(tgt_input)
    dataset = TensorDataset(src_tensor, tgt_input, src_mask, tgt_mask, tgt_tgt)
    return dataset

def process_data():
    dataset_train = sub_process_data(demo_data_train)
    dataset_test = sub_process_data(demo_data_test)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    test_dataloader = DataLoader(
        dataset_test,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle = False,
        drop_last = False
    )

    return train_dataloader, test_dataloader

def process_data_addition():
    demo_data = []
    demo_data.extend(demo_data1)
    demo_data.extend(demo_data2)
    demo_data.extend(demo_data3)
    dataset = sub_process_data(demo_data)

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])

    add_data=[]
    add_data.extend(demo_data4)
    add_data.extend(demo_data5)
    add_data.extend(demo_data6)
    add_dataset = sub_process_data(add_data)
    full_train_dataset = ConcatDataset([train_dataset, add_dataset])

    train_dataloader = DataLoader(
        full_train_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle = False,
        drop_last = False
    )

    return train_dataloader, test_dataloader