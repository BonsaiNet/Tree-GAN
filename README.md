## BonsaiNet

[License](https://github.com/RoozbehFarhoodi/McNeuron/blob/master/LICENSE)

## Goal

The intricate morphology of neurons has fascinated scientists since the dawn of neuroscience. Here we use recent techniques of deep learning to build a generative model for 3-d neural structures.This generative model can be used in the simulation of realistic neural structures and in the inference of neuronal structure from imaging  techniques.

## Approach

Neuronal structures can be approximated as trees in 3-d space. Each neuron is uniquely specified by the adjacency matrix of its tree (morphology) and the location of the tree's vertices in 3-d space. We train generative adversarial networks (GANs) to synthesize the the morphology and geometry of realistic neurons, whose joint statistics are obtained from a neuroanatomy database: [neuromorpho.org](http://neuromorpho.org).

![alt tag](https://raw.githubusercontent.com/tree-gan/BonsaiNet/master/network.png)

## Prerequisites

- Python 2.7
- [Keras](https://github.com/fchollet/keras/tree/master/keras)
- [SciPy](http://www.scipy.org/install.html)

## installation

Clone the repository.
```
$ git clone https://github.com/tree-gan/BonsaiNet
```
