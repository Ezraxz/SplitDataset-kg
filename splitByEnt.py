# Author: liangwei
# Date: 2022-3-10

import os
import random
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle


data_path = './FB15k-237'
# data_path = './NELL-995'
# data_path = './WN18RR'
# data_path = './YAGO3-10'

#重合因子
factor_overlap = 0.9
#客户端数量
num_client = 3


ent2id_file = open(os.path.join(data_path, 'entity2id.txt'))
ent2id = dict()
num_ent = int(ent2id_file.readline())
for line in ent2id_file.readlines():
    ent, idx = line.split()
    ent2id[ent] = int(idx)
id2ent = {v: k for k, v in ent2id.items()}

rel2id = dict()
rel2id_file = open(os.path.join(data_path, 'relation2id.txt'))
num_rel = int(rel2id_file.readline())
for line in rel2id_file.readlines():
    rel, idx = line.split()
    rel2id[rel] = int(idx)
id2rel = {v: k for k, v in rel2id.items()}

triples = []

train2id_file = open(os.path.join(data_path, 'train2id.txt'))
num_train = int(train2id_file.readline())
train_triples = []
for line in train2id_file.readlines():
    line = map(lambda x: int(x), line.split())
    h, t, r = line
    triples.append([h, r, t])
    train_triples.append([h, r, t])

valid2id_file = open(os.path.join(data_path, 'valid2id.txt'))
num_valid = int(valid2id_file.readline())
valid_triples = []
for line in valid2id_file.readlines():
    line = map(lambda x: int(x), line.split())
    h, t, r = line
    triples.append([h, r, t])
    valid_triples.append([h, r, t])

test2id_file = open(os.path.join(data_path, 'test2id.txt'))
num_test = int(test2id_file.readline())
test_triples = []
for line in test2id_file.readlines():
    line = map(lambda x: int(x), line.split())
    h, t, r = line
    triples.append([h, r, t])
    test_triples.append([h, r, t])

triples = np.array(triples)

h_ent_pool = np.unique(triples[:,0])
t_ent_pool = np.unique(triples[:,2])

#实体集合
ent_pool = np.unique(np.hstack((h_ent_pool, t_ent_pool))) 

#客户端实体数量
num_client_ent = round(len(ent_pool) / num_client)
client_ent = []


#-------------------------------实验1-客户端重叠实体不同------------------------------------------
#分配数量
# give_ent = []
# num_give = round(num_client_ent * factor_overlap / (2 - (num_client - 1) * factor_overlap))

# print("每个客户端分配: ", num_give)


# for i in range(num_client):
#     give_ent.append([])
#     client_ent.append([])
#     if i != num_client - 1:
#         client_ent[i] = (np.random.choice(ent_pool, num_client_ent, replace=False))
#         ent_pool = np.setdiff1d(ent_pool, client_ent, assume_unique=True)
#     else:
#         client_ent[i] = ent_pool
#     for j in range(num_client):
#         give_ent[i].append([])
#         give_ent[i][j] = (np.random.choice(client_ent[i], num_give, replace=False))

# #抽取部分实体使不同重合因子实体数量相同
# for i in range(num_client):
#     client_ent[i] = (np.random.choice(client_ent[i], num_client_ent - num_give * (num_client - 1), replace=False))


# for i in range(num_client):
#     for j in range(num_client):
#         if j != i:
#             client_ent[i] = np.hstack((client_ent[i], give_ent[j][i]))

#-------------------------------实验1-客户端重叠实体不同------------------------------------------


#-------------------------------实验2-客户端重叠实体相同------------------------------------------
num_align = round(num_client_ent * factor_overlap)
num_alone = num_client_ent - num_align

ent_align = (np.random.choice(ent_pool, num_align, replace=False))
ent_pool = np.setdiff1d(ent_pool, ent_align, assume_unique=True)
for i in range(num_client):
    client_ent.append([])
    client_ent[i] = (np.random.choice(ent_pool, num_alone, replace=False))
    ent_pool = np.setdiff1d(ent_pool, client_ent, assume_unique=True)

for i in range(num_client):
    client_ent[i] = np.hstack((client_ent[i], ent_align))
#-------------------------------实验2-客户端重叠实体相同------------------------------------------


for i in range(num_client):
    client_ent[i] = set(client_ent[i])

rate_overlap = 0

for i in range(num_client - 1):
    for j in range(i+1,num_client):
        #计算两个客户端重合实体数
        tmp = [val for val in client_ent[i] if val in client_ent[j]]
        num_overlap = len(tmp)
        print(num_overlap," ",len(client_ent[i])," ",len(client_ent[j]))
        rate_overlap += 2 * num_overlap / (len(client_ent[i]) + len(client_ent[j]))
        
rate_overlap = rate_overlap / num_client
print("实体重合率", rate_overlap)

client_triples = [[] for i in range(num_client)]

ent_use = []

for tri in triples.tolist():
    h, r, t = tri
    for i in range(num_client):
        if h in client_ent[i] and t in client_ent[i]:
            client_triples[i].append(tri)
            ent_use.append(h)
            ent_use.append(t)
            #break

#排除未使用的实体 重新编号
ent_use = list(set(ent_use))

ent2newid = dict()
for i in range(len(ent_use)):
    ent2newid[ent_use[i]] = i
    
for i in range(num_client):
    for j in range(len(client_triples[i])):
        h, r, t = client_triples[i][j]
        client_triples[i][j] = [ent2newid[h], r, ent2newid[t]]
    print('第', i ,'个三元组数量: ', len(client_triples[i]))


#分配训练集 测试集 验证集
client_data = []

for client_idx in tqdm(range(num_client)):
    all_triples = client_triples[client_idx]

    triples_reidx = []
    ent_reidx = dict()
    rel_reidx = dict()
    entidx = 0
    relidx = 0

    ent_freq = ddict(int)
    rel_freq = ddict(int)

    for tri in all_triples:
        h, r, t = tri
        ent_freq[h] += 1
        ent_freq[t] += 1
        rel_freq[r] += 1
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        triples_reidx.append([h, r, t, ent_reidx[h], rel_reidx[r], ent_reidx[t]])

    client_train_triples = []
    client_valid_triples = []
    client_test_triples = []

    
    random.shuffle(triples_reidx)
    for idx, tri in enumerate(triples_reidx):
        h, r, t, _, _, _ = tri
        if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
            client_test_triples.append(tri)
            ent_freq[h] -= 1
            ent_freq[t] -= 1
            rel_freq[r] -= 1
        else:
            client_train_triples.append(tri)
        if len(client_test_triples) > int(len(triples_reidx) * 0.2):
            break
    client_train_triples.extend(triples_reidx[idx+1:])

    random.shuffle(client_test_triples)
    test_len = len(client_test_triples)
    client_valid_triples = client_test_triples[:int(test_len/2)]
    client_test_triples = client_test_triples[int(test_len/2):] 

    train_edge_index_ori = np.array(client_train_triples)[:, [0, 2]].T
    train_edge_type_ori = np.array(client_train_triples)[:, 1].T
    train_edge_index = np.array(client_train_triples)[:, [3, 5]].T
    train_edge_type = np.array(client_train_triples)[:, 4].T

    valid_edge_index_ori = np.array(client_valid_triples)[:, [0, 2]].T
    valid_edge_type_ori = np.array(client_valid_triples)[:, 1].T
    valid_edge_index = np.array(client_valid_triples)[:, [3, 5]].T
    valid_edge_type = np.array(client_valid_triples)[:, 4].T

    test_edge_index_ori = np.array(client_test_triples)[:, [0, 2]].T
    test_edge_type_ori = np.array(client_test_triples)[:, 1].T
    test_edge_index = np.array(client_test_triples)[:, [3, 5]].T
    test_edge_type = np.array(client_test_triples)[:, 4].T

    client_data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type, 
                          'edge_index_ori': train_edge_index_ori, 'edge_type_ori': train_edge_type_ori},
                'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type, 
                         'edge_index_ori': test_edge_index_ori, 'edge_type_ori': test_edge_type_ori},
                'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type, 
                      'edge_index_ori': valid_edge_index_ori, 'edge_type_ori': valid_edge_type_ori}}

    client_data.append(client_data_dict)

pickle.dump(client_data, open('./split_res/3th_res/fb15k237-3-entslt-0.9.pkl', 'wb'))
