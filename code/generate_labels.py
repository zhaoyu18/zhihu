import gensim
import pandas as pd
import numpy as np
from scipy import sparse, vstack
from utils import _save, _load, SaveData

train_y_df = pd.read_csv('./ieee_zhihu_cup/question_topic_train_set.txt', sep='\t', header=None)
train_y_df.columns = ['qid', 'lids']

topic_info_df = pd.read_csv('./ieee_zhihu_cup/topic_info.txt', sep='\t', header=None)
topic_info_df.columns = ['lid', 'pids', 'cn', 'wn', 'cd', 'wd']

# generate label ids
lids = []
for index, row in train_y_df.iterrows():
    lids.append([topic_info_df[topic_info_df.lid == np.int64(x)].index[0] for x in row.lids.split(',')])
print('finish generate label id array')

# generate label vectors
lid_vectors = []
for (index, item) in enumerate(lids):
    vector = [0] * 1999
    for lid in item:
        vector[lid] = 1
    lid_vectors.append(sparse.csr_matrix(vector))
print('finish generate label vectors')
    
lid_vectors = sparse.vstack(lid_vectors, format='csr')

save_data = SaveData()
save_data.lid_vectors = lid_vectors
_save('train_label_id_vectors.pkl', save_data)
print('finished')