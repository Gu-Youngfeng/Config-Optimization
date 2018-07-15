# Config-Optimization
This project includes implementations of different configuration optimization methods in prior papers. 
Note that due to the randomness of each method, we can roughly replicate their experimental results in folder `/experiments/`. 

### 1. CART method

```java
@inproceedings{DBLP:conf/kbse/GuoCASW13,
  author    = {Jianmei Guo and
               Krzysztof Czarnecki and
               Sven Apel and
               Norbert Siegmund and
               Andrzej Wasowski},
  title     = {Variability-aware performance prediction: {A} statistical learning
               approach},
  booktitle = {2013 28th {IEEE/ACM} International Conference on Automated Software
               Engineering, {ASE} 2013, Silicon Valley, CA, USA, November 11-15,
               2013},
  pages     = {301--311},
  year      = {2013},
  url       = {https://doi.org/10.1109/ASE.2013.6693089},
  doi       = {10.1109/ASE.2013.6693089},
  timestamp = {Tue, 23 May 2017 01:06:50 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/kbse/GuoCASW13},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### 2. Progressive & Projective methods

```java
@inproceedings{DBLP:conf/kbse/SarkarGSAC15,
  author    = {Atri Sarkar and
               Jianmei Guo and
               Norbert Siegmund and
               Sven Apel and
               Krzysztof Czarnecki},
  title     = {Cost-Efficient Sampling for Performance Prediction of Configurable
               Systems {(T)}},
  booktitle = {30th {IEEE/ACM} International Conference on Automated Software Engineering,
               {ASE} 2015, Lincoln, NE, USA, November 9-13, 2015},
  pages     = {342--352},
  year      = {2015},
  url       = {https://doi.org/10.1109/ASE.2015.45},
  doi       = {10.1109/ASE.2015.45},
  timestamp = {Tue, 23 May 2017 01:06:49 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/kbse/SarkarGSAC15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### 3. Rank-based method

```java
@inproceedings{DBLP:conf/sigsoft/NairMSA17,
  author    = {Vivek Nair and
               Tim Menzies and
               Norbert Siegmund and
               Sven Apel},
  title     = {Using bad learners to find good configurations},
  booktitle = {Proceedings of the 2017 11th Joint Meeting on Foundations of Software
               Engineering, {ESEC/FSE} 2017, Paderborn, Germany, September 4-8, 2017},
  pages     = {257--267},
  year      = {2017},
  crossref  = {DBLP:conf/sigsoft/2017},
  url       = {http://doi.acm.org/10.1145/3106237.3106238},
  doi       = {10.1145/3106237.3106238},
  timestamp = {Wed, 16 Aug 2017 08:10:24 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/sigsoft/NairMSA17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### 4. FLASH method

```java
@article{DBLP:journals/corr/abs-1801-02175,
  author    = {Vivek Nair and
               Zhe Yu and
               Tim Menzies and
               Norbert Siegmund and
               Sven Apel},
  title     = {Finding Faster Configurations using {FLASH}},
  journal   = {CoRR},
  volume    = {abs/1801.02175},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.02175},
  archivePrefix = {arXiv},
  eprint    = {1801.02175},
  timestamp = {Thu, 01 Feb 2018 19:52:26 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1801-02175},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
