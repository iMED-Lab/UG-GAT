# UG-GAT
This repository holds the Pytorch implementation of **Uncertainty-guided Graph Attention Network for Parapneumonic Effusion Diagnosis**
 (UG-GAT). 

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
 
'''
  @article{hao2021uncertainty,
  title={Uncertainty-guided Graph Attention Network for Parapneumonic Effusion Diagnosis},
  author={Hao, Jinkui and Liu, Jiang and Pereira, Ella and Liu, Ri and Zhang, Jiong and Zhang, Yangfan and Yan, Kun and Gong, Yan and Zheng, Jianjun and Zhang, Jingfeng and others},
  journal={Medical Image Analysis},
  pages={102217},
  year={2021},
  publisher={Elsevier}
}
'''
  </span>
  
