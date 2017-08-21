import gensim
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import _save, _load, SaveData

# load w2v embedding
word2vec = gensim.models.KeyedVectors.load_word2vec_format('./ieee_zhihu_cup/word_embedding.txt', binary=False)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

# load train text
train_x_df = pd.read_csv('./ieee_zhihu_cup/question_train_set.txt', sep='\t', header=None)
train_x_df.columns = ['qid', 'title_chars', 'title_words', 'des_chars', 'des_words']

MAX_TEXT_LENGTH = 200

texts = []
for index, row in train_x_df.iterrows():
    text = []
    if pd.isnull(row.title_words) == False:
        text += row.title_words.split(',')
    if pd.isnull(row.des_words) == False:
        text += row.des_words.split(',')
    texts.append(' '.join(text[:MAX_TEXT_LENGTH]))

print('finish loading train text')
    
# load test text
test_x_df = pd.read_csv('./ieee_zhihu_cup/question_eval_set.txt', sep='\t', header=None)
test_x_df.columns = ['qid', 'title_chars', 'title_words', 'des_chars', 'des_words']

test_texts = []
for index, row in test_x_df.iterrows():
    text = []
    if pd.isnull(row.title_words) == False:
        text += row.title_words.split(',')
    if pd.isnull(row.des_words) == False:
        text += row.des_words.split(',')
    test_texts.append(' '.join(text[:MAX_TEXT_LENGTH]))

print('finish loading test text')

# generate train/test data
MAX_NUM_WORDS = 200000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts + test_texts)

sequences = tokenizer.texts_to_sequences(texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_TEXT_LENGTH)

embedding_matrix = np.zeros((MAX_NUM_WORDS, 256))
for word, i in word_index.items():
    if word in word2vec.vocab and i < MAX_NUM_WORDS:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

save_data = SaveData()
save_data.data = data
save_data.test_data = test_data
save_data.embedding_matrix = embedding_matrix
save_data.nb_words = MAX_NUM_WORDS
_save('text_data_with_embedding_matrix.pkl', save_data)
print('finished')