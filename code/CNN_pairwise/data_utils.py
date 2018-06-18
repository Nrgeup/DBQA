import csv
from tensorflow.python.platform import gfile
import re
import random
import numpy as np
title = []
story = []
dic = []
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_DIGIT_RE = re.compile(r"\d")

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
SOS_ID = 1


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def create_vocabulary(vocab_path, vocab_size, file_path1, file_path2=None):
    dict = {}
    title = []
    story = []
    sent = []
    ct = 0
    with gfile.GFile(file_path1, mode="r") as f1:
        with gfile.GFile(file_path2, mode="r") as f2:
            q, a = f1.readline().rstrip("\n"), f2.readline().rstrip("\n")
            while q and a:
                q, a = q.split(" "), a.split(" ")
                for w in q:
                    if w == "":
                        continue
                    if w not in dict:
                        dict[w] = 1
                    else:
                        dict[w] += 1
                for w in a:
                    if w == "":
                        continue
                    if w not in dict:
                        dict[w] = 1
                    else:
                        dict[w] += 1
                q, a = f1.readline().rstrip("\n"), f2.readline().rstrip("\n")
    # dict["..."] = 0
    vocab = _START_VOCAB + sorted(dict, key=dict.get, reverse=True)
    print(len(vocab))
    print(dict[vocab[60000]])
    if len(vocab) > vocab_size:
        vocab = vocab[:vocab_size]
    with gfile.GFile(vocab_path, mode="w") as vocab_file:
        for w in vocab:
            vocab_file.write(w + "\n")
    return vocab, (title, story, sent)

# def sentence_to_token_ids(sentence, vocabulary, normalize_digits=False):
#   if not normalize_digits:
#     return [vocabulary.get(w.lower(), UNK_ID) for w in sentence]
#   return [vocabulary.get(_DIGIT_RE.sub("0", w.lower()), UNK_ID) for w in sentence]
#
# def story_to_ids(title_set, story_set, title_path, story_path, vocab_path):
#     if not gfile.Exists(story_path):
#         print("Tokenizing data in %s" % title_path)
#         vocab, _ = initialize_vocabulary(vocab_path)
#         with gfile.GFile(story_path, mode="w") as tokens_file:
#             for story in story_set:
#                 story_token = []
#                 for line in story:
#                     token_ids = sentence_to_token_ids(line, vocab)
#                     story_token.extend(token_ids)
#                     story_token.append(EOS_ID)
#                 tokens_file.write(" ".join([str(tok) for tok in story_token]) + "\n")
#     if not gfile.Exists(title_path):
#         print("Tokenizing data in %s" % title_path)
#         vocab, _ = initialize_vocabulary(vocab_path)
#         with gfile.GFile(title_path, mode="w") as tokens_file:
#             for title in title_set:
#                 token_ids = sentence_to_token_ids(title, vocab)
#                 tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_data(data_dir, file1, file2, vocab_size):
    story_set = []
    title_set = []
    vocab_path = data_dir + "/vocab_" + str(vocab_size)
    train_story_path = data_dir + "/train_story_" + str(vocab_size)
    train_title_path = data_dir + "/train_title_" + str(vocab_size)
    valid_story_path = data_dir + "/valid_story_" + str(vocab_size)
    valid_title_path = data_dir + "/valid_title_" + str(vocab_size)
    if not False:
        vocab, data = create_vocabulary(vocab_path, vocab_size, file1, file2)
        title_set, story_set, sent_set = data


    return train_title_path, train_story_path, valid_title_path, valid_story_path, vocab_path

if __name__ == "__main__":
    prepare_data("/home/wangtm/code/DBQA","train_q", "train_a", 60000)