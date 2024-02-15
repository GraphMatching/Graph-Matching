"""
graph-matching experiment settings
"""
setting = "transfer" #choose from specific or transfer
dataset = "cheat" #choose training and testing dataset for specific setting
probability_model = "uniform"
training_perturbation_level = 0.01 #choose from 0, 0.01, 0.05
testing_perturbation_levels = [0.01]
no_training_samples_per_graph = 10#No. of perturbed S used in training
no_testing_samples_each_level = 10#No. testing samples per perturbation level
NUM_HIDDEN_LAYERS = 6
HIDDEN_DIM = 6
output_feature_size = 4
lr = 0.001
epoch = 20
encoder = "GIN" #choose from GIN, GCN, GNN

"""
sub-graph experiment setting
"""
subgraph_dataset = "douban_real"