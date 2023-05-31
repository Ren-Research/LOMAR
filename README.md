# Learning for Edge-Weighted Online Bipartite Matching with Robustness Guarantees

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

[Pengfei Li](https://www.cs.ucr.edu/~pli081/), [Jianyi Yang](https://jyang-ai.github.io/) and [Shaolei Ren](https://intra.ece.ucr.edu/~sren/)

**Note**

This is the official implementation of the ICML 2023 paper 

## Requirements

* python>=3.6

## Installation
* Clone this repo:
```bash
git clone git@github.com:Ren-Research/LOMAR.git
cd LOMAR
```
Then please refer to [the install guide](INSTALL.md) for more details about installation

## Usage 
1. Generate graph dataset
2. Train the RL model
3. Evaluate the policy

## Evaluation

In our experiment, we set $u_0 = 10$ and $v_0 = 60$ to generate the training and testing datasets. The number of graph instances in the training and testing datasets are 20000 and 1000, respectively. For the sake of reproducibility and fair comparision, our settings follows the same setup of our [baseline](https://github.com/lyeskhalil/CORL). 


## Citation
```BibTex
    @inproceedings{Li2023LOMAR,
        title={Learning for Edge-Weighted Online Bipartite Matching with Robustness Guarantees},
        author={Li, Pengfei and Yang, Jianyi and Ren, Shaolei},
        booktitle={International Conference on Machine Learning},
        year={2023},
        organization={PMLR}
    }
```


## Codebase
Thanks for the code base from Mohammad Ali Alomrani, Reza Moravej, Elias B. Khalil. The public repository of their code is available at [https://github.com/lyeskhalil/CORL](https://github.com/lyeskhalil/CORL)








