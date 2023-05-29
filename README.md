# Network Alignment with Trasnferable Graph Autoencoders

This repository contains the implementation for the T-GAE framework for graph matching proposed in the paper "Network Alignment with Transferable Graph Autoencoders". 

Please change parameters in the args.py file, the current settings are default to be specific experiment on Celegans. To run the experiments in Section 5, the probability_model should be "uniform", and to run the experiments in the Appendix, please change it to "degree"

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
