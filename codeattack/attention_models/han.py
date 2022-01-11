import torch
from torch import nn
from .utils import preprocess, rev_label_map
import json
import os
import gzip
import os
import sys
import io
import re
import random
import csv
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from .han_model import HierarchialAttentionNetwork
#from PIL import Image, ImageDraw, ImageFont
csv.field_size_limit(sys.maxsize)


n_classes = 2
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?

class HAN():

    def __init__(self, path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_folder = os.path.dirname(path)
        with open(os.path.join(self.data_folder, 'word_map.json'), 'r') as j:
            self.word_map = json.load(j)
        self.model = HierarchialAttentionNetwork(n_classes=n_classes,
                                            vocab_size=len(self.word_map),
                                            emb_size=200,
                                            word_rnn_size=word_rnn_size,
                                            sentence_rnn_size=sentence_rnn_size,
                                            word_rnn_layers=word_rnn_layers,
                                            sentence_rnn_layers=sentence_rnn_layers,
                                            word_att_size=word_att_size,
                                            sentence_att_size=sentence_att_size,
                                            dropout=dropout)
        mode_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(mode_dict)
        #self.model = self.model.to(self.device)
        self.model.eval()

    def classify(self,document):

        """
        Classify a document with the Hierarchial Attention Network (HAN).

        :param document: a document in text form
        :return: pre-processed tokenized document, class scores, attention weights for words, attention weights for sentences, sentence lengths
        """

        sentence_limit = 1000
        word_limit =  1000

        word_map = self.word_map
        sent_tokenizer = PunktSentenceTokenizer()
        word_tokenizer = TreebankWordTokenizer()
        # A list to store the document tokenized into words
        model = self.model
        device = self.device
        doc = list()

        # Tokenize document into sentences
        sentences = list()
        for paragraph in preprocess(document).splitlines():
            sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

        # Tokenize sentences into words
        for s in sentences[:sentence_limit]:
            w = s.split(" ")
            w = w[:word_limit]
            #w = word_tokenizer.tokenize(s)[:word_limit]
            if len(w) == 0:
                continue
            doc.append(w)

        # Number of sentences in the document
        sentences_in_doc = len(doc)
        sentences_in_doc = torch.LongTensor([sentences_in_doc]).to(device)  # (1)

        # Number of words in each sentence
        words_in_each_sentence = list(map(lambda s: len(s), doc))
        words_in_each_sentence = torch.LongTensor(words_in_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

        # Encode document with indices from the word map
        encoded_doc = list(
            map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
                doc)) + [[0] * word_limit] * (sentence_limit - len(doc))
        encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

        # Apply the HAN model
        scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc,
                                                    words_in_each_sentence)  # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
        scores = scores.squeeze(0)  # (n_classes)
        scores = nn.functional.softmax(scores, dim=0)  # (n_classes)
        word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
        sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
        words_in_each_sentence = words_in_each_sentence.squeeze(0)  # (n_sentences)

        return doc, scores, word_alphas, sentence_alphas