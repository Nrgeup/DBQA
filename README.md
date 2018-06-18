# DBQA: pairwise-based deep model


## Introduction

This task is mainly based on the NLPCC 2017 DBQA task. Given a question and its corresponding documentation, Build a DBQA system to select one or more sentences from the document as a suitable answer.

## Motholds

In this project, we have implemented two models:

1. The first is a pairwise model based on the CNN representation (CNN_pairwise):

![cnn](./pic/cnn.png)

2. The first is a pairwise model based on the RNN representation, combining position and overlap information (RNN_pairwise):

![rnn](./pic/rnn.png)

## Results


We ensemble results of two models(just averaging), and our result ranks 3-th amoung the 18 submission in our class.