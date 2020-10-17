# AIDL - EEG Based Emotion Recognition

## Summary
This project seeks to apply deep learning techniques to electroencephalography 
(EEG) data collected in the context of subject emotion recognition. 

## Instructions

### Requirements
Docker :whale: `19.03+` is required. A container is used to manage other requirements.

### Data
Datasets currently used in this project are 
[SEED-IV](http://bcmi.sjtu.edu.cn/~seed/seed-iv.html) and
[OpenNEURO](https://openneuro.org/datasets/ds003004/versions/1.0.0). 
In order to replicate the results of the code in this repository, acquiring permission 
and downloading the datasets are linking them to the respective `./data` repository of 
each dataset is necessary.

### Running Code
Replication of results can be acheived by running the following docker 
[script](run_docker.sh). This script will build a docker container containing project 
dependencies if not already built locally. It will then launch a Jupyter Notebook in 
the container which contains replicable project code and results for each dataset in 
respective `.ipynb` notebook files.

### Current Status
The [SEED-IV](./SEED_IV) dataset is the dataset being primarily studied. Previous, 
but currently incomplete work on the [ds003004](./ds003004-download) dataset is also 
contained. Further datasets may be added.

## Credit

### Authors
This project is the work of [Atneya Nair](https://github.com/atneya) 
and [Akum Kang](https://github.com/kangakum36).

### Institution 
This repository is a subproject of the 
[AI-based Discovery and Innovation](https://www.vip.gatech.edu/teams/ai-based-discovery-and-innovation)
vertically integrated project at [Georgia Tech](https://gatech.edu), led by 
Prof. [Ali Abidi](https://www.ece.gatech.edu/faculty-staff-directory/ali-adibi).

