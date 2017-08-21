from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras import metrics
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from scipy import sparse, vstack
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from utils import _save, _load, SaveData

np.random.seed(1)

text_saved_data = _load('./text_data_with_embedding_matrix.pkl')
data = text_saved_data.data
test_data = text_saved_data.test_data
embedding_matrix = text_saved_data.embedding_matrix
nb_words = text_saved_data.nb_words

label_saved_data = _load('./train_label_id_vectors.pkl')
label_vectors = label_saved_data.lid_vectors

print('load data finish')

import math
def eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    
    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0   #总问题数量
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / (all_marked_label_num + K.epsilon())

    return (precision * recall) / (precision + recall + K.epsilon())

def build_model(emb_matrix, max_sequence_length):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=max_sequence_length,
        trainable=True
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')
    conv7 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')
    conv8 = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')

    # Define inputs
    seq = Input(shape=(max_sequence_length,))

    # Run inputs through embedding
    emb = emb_layer(seq)

    # Run through CONV + GAP layers
    conv1 = conv1(emb)
    glob1 = GlobalAveragePooling1D()(conv1)

    conv2 = conv2(emb)
    glob2 = GlobalAveragePooling1D()(conv2)

    conv3 = conv3(emb)
    glob3 = GlobalAveragePooling1D()(conv3)

    conv4 = conv4(emb)
    glob4 = GlobalAveragePooling1D()(conv4)

    conv5 = conv5(emb)
    glob5 = GlobalAveragePooling1D()(conv5)

    conv6 = conv6(emb)
    glob6 = GlobalAveragePooling1D()(conv6)
    
    conv7 = conv7(emb)
    glob7 = GlobalAveragePooling1D()(conv7)
    
    conv8 = conv8(emb)
    glob8 = GlobalAveragePooling1D()(conv8)

    merge = concatenate([glob1, glob2, glob3, glob4, glob5, glob6, glob7, glob8])

    # The MLP that determines the outcome
    x = Dropout(0.1)(merge)
    x = BatchNormalization()(x)
    x = Dense(1500, activation='relu')(x)

    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    pred = Dense(1999, activation='softmax')(x)

    model = Model(inputs=seq, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.top_k_categorical_accuracy])

    return model

nfolds = 5
folds = KFold(data.shape[0], n_folds = nfolds, shuffle = True, random_state = 2017)
pred_results = []

def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index]
        y_batch = y[batch_index,:].toarray()
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


for curr_fold, (idx_train, idx_val) in enumerate(folds):

    data_train = data[idx_train]
    y_train = label_vectors[idx_train]

    data_val = data[idx_val]
    y_val = label_vectors[idx_val]
    
    model = build_model(embedding_matrix, data.shape[1])
    
    bst_model_path = 'model_cnn_1_fold{}.h5'.format(curr_fold)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_weights_only=True)
    
    print(data_train.shape, y_train.shape)
    print(bst_model_path, "curr_fold:", curr_fold)
    
    hist = model.fit_generator(generator=batch_generator(data_train, y_train, 1024, True),
                               steps_per_epoch=2344,
                               epochs=3,
                               callbacks=[model_checkpoint],
                               verbose = 1)

    model.load_weights(bst_model_path)
#    break
    preds = model.predict(test_data, batch_size=2048, verbose=1)
    pred_results.append(preds)

#pred = model.predict(data_val, batch_size=2048, verbose=1)
#eval_data = []
#for y_pred, y_true in zip(pred, y_val.todense()):
#    pred_labels = np.argsort(y_pred)[::-1][:5]
#    true_labels = np.where(y_true == 1)[1].tolist()
#    eval_data.append((pred_labels, true_labels))
#print(eval(eval_data))

test_pred = pred_results[0] + pred_results[1] + pred_results[2] + pred_results[3] + pred_results[4]
_save('model_cnn_1_pred.pkl', test_pred)
print('finished')