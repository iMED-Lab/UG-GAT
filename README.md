# UG-GAT
This repository holds the Pytorch implementation of UG-GAT. 

## Introduction
We utilize all the CT images containing uncertainty information of a patient rather than a single 2D slice, and propose a graph-based framework for UPPE and CPPE classification.

## Training BayesianCNN
BayesianCNN Training  can be done:
```
python trainCNN.py
```
## Obtaining feature and uncertainty for graph
After training the Bayesian, you can generate the image representations and uncertainty by running:
```
python test.py
```
## Trainging UG-GAT
UG-GAT can be trained and tested by running:
```
python trainGraph.py
```

## Citing This Paper
<span id="jump">
If you use this code,please use the following BibTeX entry.
