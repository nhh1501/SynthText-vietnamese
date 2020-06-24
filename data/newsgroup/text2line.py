from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import glob

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '.', data)

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        # self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def normalizeString(s):

    if (s[:5] == 'test_') or (s[:6] =='train_'):
        s = ''
    if (len(s)<= 7):
        s = ''

    s = re.sub(r"K\s",r"không ",s,flags=re.IGNORECASE)
    s = re.sub(r"Ko\s",r"không ",s,flags=re.IGNORECASE)
    s = re.sub(r"\svs\s",r" với ",s,flags=re.IGNORECASE)
    s = re.sub(r"\s+", r" ", s).strip() #nhiều space -> 1 space

    # s = s[:4] + re.sub(r"((\b)[A-Z])", r".\1", s[4:])
    # s = remove_emojis(s)

    s = re.sub(r'["”“]+', r"", s) # bỏ "
    s = re.sub(r'\ufeff',r'',s)
    s = re.sub(r'[“]+', r"", s) # bỏ "
    s = re.sub(r"[']+", r"", s)
    if (s[:1] == " "):
        s = s[1:]
    s = re.sub(r'[:!?,@()_]+', r".", s)
    s = re.sub(r' - ',r'.',s)
    s = re.sub(r'- ',r'',s)
    s = re.sub(r"\. ", r".", s,flags=re.IGNORECASE)
    s = re.sub(r" \.", r".", s)
    s = re.sub(r'[.]+', r".", s)
    s = re.sub(r'\.$', r"", s) #bỏ chấm cuối câu
    return s

def trim_write(f,stri):
    if (len(stri.split(" ")) < MAX_LENGTH and len(stri.split(" ")) > MIN_LENGTH):
     f.write(stri + '\n')

def write_to_text(list,f):

    for i in list:
        stri = str(i)
        stri = re.sub(r"\['", r"", stri)
        stri = re.sub(r"\']", r"", stri)
        # print(stri)
        while (True):
            a = re.search("\.", stri)
            if a != None:
                i1 = stri[:a.regs[0][0]]
                # print(i1)

                trim_write(f,i1)
                stri = stri[a.regs[0][1]:]
                # print(i1)
                # print(stri)
            else:
                trim_write(f,stri)
                break

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf_16_le').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # voc = Voc(corpus_name)
    # return voc, pairs
    return pairs

def loadPrepareData( corpus_name, datafile):
    # voc, pairs = readVocs(datafile, corpus_name)
    pairs = readVocs(datafile, corpus_name)
    # print("Read {!s} sentence pairs".format(len(pairs)))
    # pairs = filterPairs(pairs)
    # print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    # print("Counting words...")
    # for pair in pairs:
    #     voc.addSentence(pair[0])
    #     voc.addSentence(pair[1])
    # print("Counted words:", voc.num_words)
    # return voc, pairs
    return pairs



MAX_LENGTH = 7  # Maximum sentence length to consider
MIN_LENGTH = 1  # Maximum sentence length to consider
corpus_name = 'SA_demo'
# datafile = 'SA_demo\\test1.txt'
datafiles = glob.glob('dt/**/*.txt', recursive=True)
out_data = 'SA_demo\\output.txt'
f = open(out_data, 'a+', encoding='utf-8')

for datafile in datafiles:
    pairs = loadPrepareData( corpus_name, datafile)
    list2 = [x for x in pairs if x != ['']]
    # print('s')
    write_to_text(list2,f)
