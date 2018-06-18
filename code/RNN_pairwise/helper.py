# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import _pickle as pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import evaluation
import string
import jieba
from nltk import stem
from tqdm import tqdm
import chardet
import re
import config
from functools import wraps

FLAGS = config.flags.FLAGS
FLAGS._parse_flags()
dataset = FLAGS.data

UNKNOWN_WORD_IDX = 0
is_stemmed_needed = False
stopwords = { word for word in open(FLAGS.stopwords).read().split()}
# ner_dict = pickle.load(open('ner_dict'))
if is_stemmed_needed:
    stemmer = stem.lancaster.LancasterStemmer()

def cut(sentence):
    tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens

#print( tf.__version__)
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco


class Alphabet(dict):
    def __init__(self, start_feature_id = 1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))


@log_time_delta
def prepare(cropuses, is_embedding_needed=False, dim=300, fresh=False):
    vocab_file = FLAGS.vocab_file
    
    if os.path.exists(vocab_file) and not fresh:
        alphabet = pickle.load(open(vocab_file, 'rb'))
    else:   
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('[UNKNOW]')  
        alphabet.add('END') 
        count = 0
        for corpus in cropuses:
            for texts in [corpus["question"].unique(), corpus["answer"]]:
                for sentence in tqdm(texts):   
                    count += 1
                    if count % 10000 == 0:
                        print(count)
                    tokens = cut(sentence)
                    for token in set(tokens):
                        alphabet.add(token)
        print(len(alphabet.keys()))
        pickle.dump(alphabet, open(vocab_file, 'wb'))
    if is_embedding_needed:
        sub_vec_file = FLAGS.embedding
        if os.path.exists(sub_vec_file) and not fresh:
            embeddings = pickle.load(open(sub_vec_file, 'rb'))
        else:    
            fname = FLAGS.embedding_file
            embeddings = get_embedding(alphabet, fname, embedding_size=dim)
            pickle.dump(embeddings, open(sub_vec_file, 'wb'))
        return alphabet, embeddings
    else:
        return alphabet


def getSubVectors(vectors, vocab, dim = 50):
    print('embedding_size:',vectors.syn0.shape[1])
    embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]] = vectors.word_vec(word)
        else:
            embedding[vocab[word]] = np.random.uniform(-0.5,+0.5,vectors.syn0.shape[1])
    return embedding


@log_time_delta
def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            if topn != 0 and lines_num >= topn:
                break
    return vectors, dim


@log_time_delta
def get_embedding(alphabet, filename, embedding_size=300):
    vectors, dim = read_vectors(filename, 300000)
    assert dim == embedding_size
    cnt = 1
    embedding = np.zeros((len(alphabet), dim))
    file = open(FLAGS.missword_file, 'w')
    for word in alphabet:
        if word in vectors:
            embedding[alphabet[word]] = vectors[word]
            cnt += 1
        else:
            file.write(word + '\n')
            embedding[alphabet[word]] = np.random.uniform(-0.5, +0.5, dim)
    file.close()
    print('embedding_size: %s\n Word number %d, \n Done!', (dim, cnt))
    return embedding


@log_time_delta
def get_overlap_dict(df, alphabet, q_len=40, a_len=40):
    d = dict()
    for question in df['question'].unique():
        group = df[df['question'] == question]
        answers = group['answer']
        for ans in answers:
            q_overlap,a_overlap = overlap_index(question,ans,q_len,a_len)
            d[(question,ans)] = (q_overlap,a_overlap)
    return d


@log_time_delta
def batch_gen_with_pair_overlap(df, alphabet, batch_size=10, q_len=40, a_len=40, fresh = True, overlap_dict=None):
    pairs = []
    start = time.time()
    for question in df["question"].unique():
        group = df[df["question"]==question]
        pos_answers = group[df["flag"] == 1]["answer"]
        neg_answers = group[df["flag"] == 0]["answer"].reset_index()
        question_indices = encode_to_split(question,alphabet,max_sentence = q_len)
        for pos in pos_answers:
            if len(neg_answers.index) > 0:
                neg_index = np.random.choice(neg_answers.index)
                neg = neg_answers.loc[neg_index,]["answer"]
                if overlap_dict:
                    q_pos_overlap, a_pos_overlap = overlap_index(question,pos,q_len,a_len)
                    q_neg_overlap, a_neg_overlap = overlap_index(question,neg,q_len,a_len)
                else:  
                    q_pos_overlap, a_pos_overlap = overlap_dict[(question,pos)]
                    q_neg_overlap, a_neg_overlap = overlap_dict[(question,neg)]
                q_position = position_index(question,q_len)
                a_pos_position = position_index(pos,a_len)
                a_neg_position = position_index(neg,a_len)
                pairs.append((question_indices,encode_to_split(pos,alphabet,max_sentence = a_len),encode_to_split(neg,alphabet,max_sentence = a_len),\
                    q_pos_overlap,q_neg_overlap,a_pos_overlap,a_neg_overlap,\
                    q_position,a_pos_position,a_neg_position))
    print('pairs:{}'.format(len(pairs)))
    end = time.time()
    delta = end - start
    print("batch_gen_with_pair_overlap_runed %.2f seconds" % (delta))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches= int(len(pairs)*1.0/batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state =132)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield [[pair[i] for pair in batch] for i in range(10)]

# calculate the overlap_index
def overlap_index(question, answer, q_len, a_len, stopwords = []):
    qset = set(cut(question))
    aset = set(cut(answer))

    q_index = np.zeros(q_len)
    a_index = np.zeros(a_len)

    overlap = qset.intersection(aset)
    for i,q in enumerate(cut(question)[:q_len]):
        value = 1
        if q in overlap:
            value = 2
        q_index[i] = value
    for i,a in enumerate(cut(answer)[:a_len]):
        value = 1
        if a in overlap:
            value = 2
        a_index[i] = value
    return q_index,a_index


def position_index(sentence,length):
    index = np.zeros(length)

    raw_len = len(cut(sentence))
    index[:min(raw_len, length)] = range(1,min(raw_len + 1,length + 1))
    # print index
    return index


def encode_to_split(sentence,alphabet,max_sentence = 40):
    indices = []    
    tokens = cut(sentence)
    for word in tokens:
        if word in alphabet:
            indices.append(alphabet[word])
        else:
            continue
    results=indices+[alphabet["END"]]*(max_sentence-len(indices))
    return results[:max_sentence]


def transform(flag):
    if flag == 1:
        return [0,1]
    else:
        return [1,0]


@log_time_delta
def batch_gen_with_single(df, alphabet, batch_size=10, q_len=33, a_len=40, overlap_dict=None):
    pairs = []
    for index, row in df.iterrows():
        quetion = encode_to_split(row["question"], alphabet,max_sentence=q_len)
        answer = encode_to_split(row["answer"], alphabet,max_sentence=a_len)
        if overlap_dict:
            q_pos_overlap, a_pos_overlap = overlap_index(row["question"], row["answer"],q_len,a_len)
        else:
            q_pos_overlap, a_pos_overlap = overlap_dict[(row["question"], row["answer"])]

        q_position = position_index(row['question'], q_len)
        a_pos_position = position_index(row['answer'], a_len)
        pairs.append((quetion,answer,q_pos_overlap, a_pos_overlap,q_position,a_pos_position))
    num_batches_per_epoch = int((len(pairs)-1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(pairs))
            batch = pairs[start_index:end_index]
            yield [[pair[j] for pair in batch] for j in range(6)]


def load(dataset=dataset, filter=False):
    data_dir = "data/" + dataset
    datas = []
    for data_name in ['train.txt', 'test.txt', 'dev.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=["question", "answer", "flag"], quoting=3).fillna('')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)


def removeUnanswerdQuestion(df):
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct=counter[counter>0].index
    counter= df.groupby("question").apply(lambda group: sum(group["flag"]==0))
    questions_have_uncorrect=counter[counter>0].index
    counter=df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi=counter[counter>1].index
    return df[df["question"].isin(questions_have_correct) &  df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()


def data_processing():
    train, test, dev = load('nlpcc',filter = False)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    q_len = map(lambda x:len(x),train['question'].str.split())
    a_len = map(lambda x:len(x),train['answer'].str.split())
    print(np.max(q_len))
    print(np.max(a_len))
    print('Total number of unique question:{}'.format(len(train['question'].unique())))
    print('Total number of question pairs for training: {}'.format(len(train)))
    print('Total number of question pairs for test: {}'.format(len(test)))
    print('Total number of question pairs for dev: {}'.format(len(dev)))
    print('Duplicate pairs: {}%'.format(round(train['flag'].mean()*100, 2)))
    print(len(train['question'].unique()))

    #text analysis
    train_qs = pd.Series(train['answer'].tolist())
    test_qs = pd.Series(test['answer'].tolist())
    dev_qs = pd.Series(dev['answer'].tolist())

    dist_train = train_qs.apply(lambda x:len(x.split(' ')))
    dist_test = test_qs.apply(lambda x:len(x.split(' ')))
    dist_dev = dev_qs.apply(lambda x:len(x.split(' ')))
    pal = sns.color_palette()
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins = 200, range=[0, 200], color=pal[2], normed = True, label='train')
    plt.hist(dist_dev, bins = 200, range=[0, 200], color=pal[3], normed = True, alpha = 0.5, label='test1')
    plt.hist(dist_test, bins = 200, range=[0, 200], color=pal[1], normed = True, alpha = 0.5, label='test2')
    
    plt.title('Normalised histogram of tokens count in answers', fontsize = 15)
    plt.legend()
    plt.xlabel('Number of words', fontsize = 15)
    plt.ylabel('Probability', fontsize = 15)

    print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
    plt.show('hard')

    qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
    who = np.mean(train_qs.apply(lambda x:'Who' in x))
    where = np.mean(train_qs.apply(lambda x:'Where' in x))
    how_many = np.mean(train_qs.apply(lambda x:'How many' in x))
    fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
    capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
    capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
    numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))
    print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
    print('Questions with [Who] tags: {:.2f}%'.format(who * 100))
    print('Questions with [where] tags: {:.2f}%'.format(where * 100))
    print('Questions with [How many] tags:{:.2f}%'.format(how_many * 100))
    print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
    print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
    print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
    print('Questions with numbers: {:.2f}%'.format(numbers * 100))


def batch_single(df,alphabet,batch_size = 10,q_len = 33,a_len = 40,overlap_dict = None):
    question = encode_to_split(df["question"],alphabet,max_sentence = q_len)
    answer = encode_to_split(df["answer"],alphabet,max_sentence = a_len)
    return question, answer

