# Network Alignment with Transferable Graph Autoencoders

This repository contains the implementation for the T-GAE framework for graph matching proposed in the paper "Network Alignment with Transferable Graph Autoencoders". 

Please change parameters in the args.py file, the current settings are default to be specific experiment on Celegans. To run the experiments in Section 5, the probability_model should be "uniform", and to run the experiments in the Appendix, please change it to "degree". 

To train the model according to equation (12), setting should be "transfer" and training perturbation level should be 0. To train according to (13), the training perturbation level should be greater than 0. For specific training in Section 5.5, change the setting to "specific" in args.py.

The required libraries can be installed by:
```
pip install -r requirements.txt
```
Unzip the dataset directory by:
```
unzip data.zip
```
Then run the experiment by:
```
python robust_GAE.py
```
