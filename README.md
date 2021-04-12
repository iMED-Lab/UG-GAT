#UG-GAT
This repository holds the Pytorch implementation of UG-GAT. 
;If you find our code useful in your research, please consider citing:

##Introduction
We utilize all the CT images containing uncertainty information of a patient rather than a single 2D slice, and propose a graph-based framework for UPPE and CPPE classification.

## Training BayesianCNN
BayesianCNN Training  can be done:
```
python trainCNN.py
```
## Obtaining feature and uncertainty for graph
After training the Bayesian, you can generate the image representations and uncertainty by running:
BayesianCNN Training  can be done:
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

@article{hao2020reconstruction,
  title={Reconstruction and Quantification of 3D Iris Surface for Angle-Closure Glaucoma Detection in Anterior Segment OCT},
  author={Hao, Jinkui and Fu, Huazhu and Xu, Yanwu and Hu, Yan and Li, Fei and Zhang, Xiulan and Liu, Jiang and Zhao, Yitian},
  journal={arXiv preprint arXiv:2006.05179},
  year={2020}
}
```
</span>