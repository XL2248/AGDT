# Code and data for [A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/D19-1559.pdf)

## Introduction

The implementation is based on [THUMT](https://github.com/thumt/THUMT). Download [Glove](http://nlp.stanford.edu/data/glove.840B.300d.zip) file and change the path in 'AGDT/thumt/thumt/bin/trainer.py' correspondingly. The dataset we used is from [GCAE](https://github.com/wxue004cs/GCAE).

## Usage

Training with the following scripts: 

+ ACSA

```
bash run_train_14.sh
bash run_train_large.sh
```

+ ATSA

```
bash run_train_r.sh
bash run_train_l.sh
```

The result can be found in the path like '/14_agdt-result-0/eval/record'.

## Requirements

+ tensorflow 1.8.0 
+ python 2.7

## Citation

If you find this project helps, please cite our paper :)

```
@inproceedings{liang-etal-2019-novel-aspect,
    title = "A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis",
    author = "Liang, Yunlong  and
      Meng, Fandong  and
      Zhang, Jinchao  and
      Xu, Jinan  and
      Chen, Yufeng  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1559",
    doi = "10.18653/v1/D19-1559",
    pages = "5568--5579",
}
```
