# Network Alignment with Transferable Graph Autoencoders

This repository is a temporary code base to reproduce the experiment for the T-GAE graph matching framework proposed in the paper "Network Alignment with Transferable Graph Autoencoders". 

Please change parameters in the args.py file, the current settings are default to be specific experiment on Celegans. To run the experiments in Section 5.3, the probability_model should be "uniform", and to run the experiments in the Appendix, please change it to "degree". 

To train the model according to equation (12), setting should be "transfer" and training perturbation level should be 0. To train according to (13), the training perturbation level should be greater than 0. For specific training in Section 5.5, change the setting to "specific" in args.py.

For subgraph matching experiments in section 5.4, please change the subgraph_dataset variable in args.py, to either "ACM_DBLP" or "douban_real".

The required libraries can be installed by:
```
pip install -r requirements.txt
```
Unzip the dataset directory by:
```
unzip data.zip
```
Then run the experiment in Section 5.3 by:
```
python robust_GAE.py
```
And experiment in Section 5.4 by:
```
python subgraph.py
```
