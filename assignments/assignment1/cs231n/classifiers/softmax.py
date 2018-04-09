import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  sum_prob = 0.0
  sum_prob_single_class = np.zeros((num_classes, 1))
  for i in xrange(num_train):
    score = X[i].dot(W)
    exp_score = np.exp(score)
    sum_exp_score = exp_score.sum()
    loss += - np.log(exp_score[y[i]] / sum_exp_score)
    dW[:, y[i]] -= X[i]
    for j in xrange(num_classes):
      dW[:, j] += X[i] * exp_score[j] / sum_exp_score
  pass

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  sum_exp_scores = exp_scores.sum(axis = 1)
  loss_sum = (- np.log(exp_scores[np.arange(scores.shape[0]), y] / sum_exp_scores)).sum()
  loss = loss_sum / num_train + 0.5 * reg * np.sum(W * W)
  correct_class = np.zeros((num_train, num_classes))
  correct_class[xrange(num_train),y] = -1
  dW += X.T.dot(correct_class)
  dW += (X.T / sum_exp_scores).dot(exp_scores) 
  dW /= num_train
  dW += reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

