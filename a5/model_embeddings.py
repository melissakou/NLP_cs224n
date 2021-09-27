#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from torch.nn.functional import dropout
from cnn import CNN
from highway import Highway

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.embed_size = embed_size
        char_embed_size = 50
        dropout_rate = 0.3

        self.char_embedding = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim=char_embed_size, padding_idx=vocab.char2id["<pad>"])
        self.cnn = CNN(in_channels=char_embed_size, out_channels=embed_size)
        self.highway = Highway(input_size=embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        sentence_length, batch_size, _ = input.shape
        
        x_emb = self.char_embedding(input)
        x_reshape = x_emb.reshape(x_emb.shape[0] * x_emb.shape[1], x_emb.shape[2], x_emb.shape[3]).permute([0, 2, 1])
        x_conv_out = self.cnn(x_reshape)
        x_highway = self.highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)

        return x_word_emb.reshape(sentence_length, batch_size, x_word_emb.shape[-1])
        ### END YOUR CODE

