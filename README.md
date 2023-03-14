# A Context-Constrained Sentence Modeling for Deception Detection in Real Interrogation Official Implementation

This repository is the official implementation of A Context-Constrained Sentence Modeling for Deception Detection in Real Interrogation. 

## 1. How to start?
Extract emobase, BERT and the CTD feature by the tracing the following script.


#### 1.1 Emobase
You need to install opensmile to extract emobase feature
```
feature_extract/emobase
```

#### 1.2 BERT
Please install the dependency of ''pytorch-pretrained-bert 0.6.2'' then use this script to extract the BERT
```
feature_extract/bert
```

#### 1.3 CTD
```
feature_extract/ctd
```

## 2. Training code
Trace ``run_multitask.py`` and ``models.py`` to get more information.


## 3. Testing code
Trace ``test_only.py`` to get more information.  
We also provide our best model in the paper, please see ``BEST_MODEL``.  
We can provide some of our feature for testing only, please mail me to get more information.
