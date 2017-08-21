import numpy as np
import pandas as pd
from utils import _save, _load, SaveData

test_pred = _load('model_cnn_0_pred.pkl') + _load('model_cnn_1_pred.pkl')

test_result = []
for y_pred in test_pred:
    pred_labels = np.argsort(y_pred)[::-1][:5]
    test_result.append(pred_labels)

print(test_result[0])

test_x_df = pd.read_csv('./ieee_zhihu_cup/question_eval_set.txt', sep='\t', header=None)
test_x_df.columns = ['qid', 'title_chars', 'title_words', 'des_chars', 'des_words']

result_labels = pd.DataFrame(test_result)
result_labels.columns = ['l1', 'l2', 'l3', 'l4', 'l5']

result = pd.DataFrame()
result['qid'] = test_x_df.qid
result['l1'] = result_labels.l1
result['l2'] = result_labels.l2
result['l3'] = result_labels.l3
result['l4'] = result_labels.l4
result['l5'] = result_labels.l5

topic_info_df = pd.read_csv('./ieee_zhihu_cup/topic_info.txt', sep='\t', header=None)
topic_info_df.columns = ['lid', 'pids', 'cn', 'wn', 'cd', 'wd']

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
for index, row in result.iterrows():
    l1.append(topic_info_df.lid[row.l1])
    l2.append(topic_info_df.lid[row.l2])
    l3.append(topic_info_df.lid[row.l3])
    l4.append(topic_info_df.lid[row.l4])
    l5.append(topic_info_df.lid[row.l5])

result['l1'] = l1
result['l2'] = l2
result['l3'] = l3
result['l4'] = l4
result['l5'] = l5

result.to_csv('model_cnn_average_5fold_submission.csv', sep=',', header=None, index=False)