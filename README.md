# DBQA: pairwise-based deep model


## Introduction

This task is mainly based on the DQQA task of NLPCC 2017. Given a question and its corresponding document, a DBQA system is built to select one or more sentences from the document as the appropriate answer.

## Mothods

In this project, we implemented two models:

1. The first is a pairwise model which based on the CNN representation (CNN_pairwise):

![cnn](./pic/cnn.png)

2. The second is a pairwise model which based on the RNN representation, combining position and overlap information (RNN_pairwise):

![rnn](./pic/rnn.png)

## Results


We ensemble results of two models(just averaging), ranking 3-rd among the 18 submission of our class.
