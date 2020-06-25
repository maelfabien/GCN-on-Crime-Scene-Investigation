Graph Convolutional Networks (GCNs) for Crime Scene Investigation (CSI) Data
====

CSI is a dataset that reports characters and detailed transcripts of the CSI TV-show. This work is based on the original implementation of GCNs in Pytorch, available [here](https://github.com/tkipf/pygcn). 

The aim of the work is:
- explore GCNs for visualization of embeddings in 3-dimensions for nodes in a criminal network (can we benefit from NLP features when visualizing the embeddings?)
- explore model performance for classification (predicting the community, i.e. the episode of CSI during which the person appeared based on TF-IDF features of the text pronounced) 

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or higher
  * Python 2.7 or 3.6 + higher

## Usage

In PyGCN, go to:

`CSI_GCN.ipynb`

![](images/emb.png)

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)