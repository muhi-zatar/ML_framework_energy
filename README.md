# PV-monitor

This repository includes the training scripts for the application of fault classification of a PV plant performance. Many architectures were implemented which are; Fully Connected Neural Networks, Long Short Term Memory, Convolutional Neural Networks, alongside 4 machine learning classification algorithms.

**Requirements**
To install teh required packages

```bash
pip install -r requirements.txt
```

**Execution**

In order to train the network at first you should fill the config.yml with the architecture of your choice and its hyper-parameters. Also, if you want to run multiple training with different hyper-parameters, edit tuning.yml file and it will train all the possible combinations.

To run the code:
```bash
python train.py --config config.yml --tuning tuning.yml
```
The tuning is optional.
