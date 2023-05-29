from algorithm import *
from model import *
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pickle
from netrd.distance import netsimile
import networkx as nx
import os.path as osp
from scipy.sparse import coo_matrix
from tqdm import tqdm
import random
import warnings
from torch.optim import Adam
import args
warnings.filterwarnings("ignore")
def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()

            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")

def gen_netsmile(S):
    np_S = S.numpy()
    G = nx.from_numpy_array(np_S)
    feat = netsimile.feature_extraction(G)
    feat = torch.tensor(feat, dtype=torch.float)
    return feat

def generate_purturbations(device, S, perturbation_level, no_samples, method):
    purturbated_samples = []
    if(method == "uniform"):
        for i in tqdm(range(no_samples)):
            num_edges = int(torch.count_nonzero(S).item()/2)
            total_purturbations = int(perturbation_level * num_edges)
            add_edge = random.randint(0,total_purturbations)
            delete_edge = total_purturbations - add_edge
            S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
            purturbated_samples.append(S_prime)
    elif(method == "degree"):
        print("Preprocessing probability distribution")
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(perturbation_level * num_edges)
        S = torch.triu(S, diagonal=0)
        ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
        ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
        ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
        try:
            D = S @ ones_long
        except:
            try:
                D = S @ ones_int
            except:
                D = S @ ones_float

        sum = torch.sum(torch.mul(D@D.T,S))
        edge_index = S.nonzero().t().contiguous()
        edge_index = np.array(edge_index)
        prob = []
        for i in range(edge_index.shape[1]):
            d1 = edge_index[0,i]
            d2 = edge_index[1,i]
            prob.append(D[d1]*D[d2]/sum)
        prob = np.array(prob,dtype='float64')
        print("Generating samples")
        for i in tqdm(range(no_samples)):
            edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations,False,prob)
            edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
            edges_index = edge_index[:,edges_remain]
            S_prime = torch.zeros_like(S)
            for j in range(edges_index.shape[1]):
                n1 = edges_index[:,j][0]
                n2 = edges_index[:,j][1]
                S_prime[n1][n2] = 1
                if(S_prime[n2][n1]==0):
                    S_prime[n2][n1] = 1
            purturbated_samples.append(S_prime)
    else:
        print("Probability model not defined.")
        exit()
    return purturbated_samples

def generate_features(purturbated_S):
    features = []
    for S in purturbated_S:
        feature = gen_netsmile(S)
        features.append(feature)
    return features

def gen_dataset(S, NUM_TO_ADD, NUM_TO_DELETE):
    SIZE = S.shape[0]
    num_added = 0
    num_deleted = 0
    E = torch.zeros(S.shape[0], S.shape[0])
    edge_indexes = (S == 1).nonzero(as_tuple=False).cpu()
    blank_indexes = (S == 0).nonzero(as_tuple=False).cpu()
    """
    delete edges
    """
    while(num_deleted < NUM_TO_DELETE):

        delete_index = random.randint(0, edge_indexes.shape[0]-1)
        index = edge_indexes[delete_index]
        E[index[0]][index[1]] = -1
        E[index[1]][index[0]] = -1
        num_deleted += 1

    """
    add edges
    """
    while (num_added < NUM_TO_ADD):

        add_index = random.randint(0, blank_indexes.shape[0] - 1)
        index = blank_indexes[add_index]
        E[index[0]][index[1]] = 1
        E[index[1]][index[0]] = 1
        num_added += 1

    S_prime = torch.add(S.cpu(),E.cpu())
    permutator = torch.randperm(SIZE)
    S_hat = S_prime[permutator]
    S_hat = S_hat.t()[permutator].t()
    P = torch.zeros(SIZE, SIZE)
    for i in range(permutator.shape[0]):
        P[i, permutator[i]] = 1
    return S, S_prime, S_hat, P

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def fit_GAE(no_samples, GAE, epoch, train_loader, train_features, device, lr):

    optimizer = Adam(GAE.parameters(), lr=lr,weight_decay=5e-4)
    for step in range(epoch):
        loss = 0
        for dataset in train_loader.keys():
            S = train_loader[dataset][0]
            initial_features = train_features[dataset]
            for i in range(len(train_loader[dataset])):
                adj_tensor = train_loader[dataset][i]
                adj = coo_matrix(adj_tensor.numpy())
                adj_norm = preprocess_graph(adj)
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

                adj_label = coo_matrix(S.numpy())
                adj_label = sparse_to_tuple(adj_label)

                adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                    torch.FloatTensor(adj_norm[1]),
                                                    torch.Size(adj_norm[2])).to(device)
                adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                    torch.FloatTensor(adj_label[1]),
                                                    torch.Size(adj_label[2])).to(device)

                initial_feature = initial_features[i].to(device)

                features = csr_matrix(initial_features[i])
                features = sparse_to_tuple(features.tocoo())
                features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                                    torch.FloatTensor(features[1]),
                                                    torch.Size(features[2])).to(device)

                weight_mask = adj_label.to_dense().view(-1) == 1
                weight_tensor = torch.ones(weight_mask.size(0))
                weight_tensor[weight_mask] = pos_weight
                weight_tensor = weight_tensor.to(device)
                z = GAE(features, initial_feature, adj_norm)
                A_pred = torch.sigmoid(torch.matmul(z,z.t()))
                loss += norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                           weight=weight_tensor)
        optimizer.zero_grad()
        loss = loss / no_samples
        loss.backward()
        optimizer.step()
        print("Epoch:", '%04d' % (step + 1), "train_loss= {0:.5f}".format(loss.item()))


def gen_test_set(device,S, no_samples_each_level, perturbation_levels,method):
    S_hat_samples = {}
    S_prime_samples = {}
    p_samples = {}
    for level in perturbation_levels:
        S_hat_samples[str(level)] = []
        S_prime_samples[str(level)] = []
        p_samples[str(level)] = []
    for level in perturbation_levels:
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(num_edges*level)
        if(method == "degree"):
            print("Preprocessing degree probability distribution")
            S = torch.triu(S, diagonal=0)
            ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
            ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
            ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
            try:
                D = S @ ones_long
            except:
                try:
                    D = S @ ones_int
                except:
                    D = S @ ones_float
            sum = torch.sum(torch.mul(D @ D.T, S))
            edge_index = S.nonzero().t().contiguous()
            edge_index = np.array(edge_index)
            prob = []
            for i in range(edge_index.shape[1]):
                d1 = edge_index[0, i]
                d2 = edge_index[1, i]
                prob.append(D[d1] * D[d2] / sum)
            prob = np.array(prob, dtype='float64')
        for i in tqdm(range(no_samples_each_level)):
            if(method == "uniform"):
                add_edge = random.randint(0, total_purturbations)
                delete_edge = total_purturbations - add_edge
                S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
            elif(method == "degree"):
                edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations, False, prob)
                edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
                edges_index = edge_index[:, edges_remain]
                S_prime = torch.zeros_like(S)
                for j in range(edges_index.shape[1]):
                    n1 = edges_index[:, j][0]
                    n2 = edges_index[:, j][1]
                    S_prime[n1][n2] = 1
                    if (S_prime[n2][n1] == 0):
                        S_prime[n2][n1] = 1
                SIZE = S_prime.shape[0]
                permutator = torch.randperm(SIZE)
                S_hat = S_prime[permutator]
                S_hat = S_hat.t()[permutator].t()
                P = torch.zeros(SIZE, SIZE)
                for i in range(permutator.shape[0]):
                    P[i, permutator[i]] = 1
            else:
                print("Probability model not defined")
                exit()
            S_hat_samples[str(level)].append(S_hat)
            p_samples[str(level)].append(P)
            S_prime_samples[str(level)].append(S_prime)
    return S_hat_samples, S_prime_samples, p_samples

def test_matching(GAE, S_hat_samples, p_samples, S_hat_features, S_emb, device):
        print("Testing over all perturbations")
        results = []
        for i in tqdm(range(len(S_hat_samples))):
            adj = coo_matrix(S_hat_samples[i].numpy())
            adj_norm = preprocess_graph(adj)
            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                torch.FloatTensor(adj_norm[1]),
                                                torch.Size(adj_norm[2])).to(device)
            initial_feature = S_hat_features[i].to(device)

            features = csr_matrix(S_hat_features[i])
            features = sparse_to_tuple(features.tocoo())
            features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                                torch.FloatTensor(features[1]),
                                                torch.Size(features[2])).to(device)
            z = GAE(features, initial_feature, adj_norm).detach()
            D = torch.cdist(S_emb, z, p=2)
            P_HG = get_match(D,device)
            c = 0
            P = p_samples[i]
            for j in range(P_HG.size(0)):
                r1 = P_HG[j].cpu()
                r2 = P[j].cpu()
                if (r1.equal(r2)): c += 1
            results.append(c/S_emb.shape[0])

        results = np.array(results)
        avg = np.average(results)
        std = np.std(results)
        print("Correct number of matchings is " + str(avg)[:6] + "+-" +str(std)[:6])
        print()

def load_adj(dataset):
    if (dataset == "celegans"):
        #size = 453
        S = torch.load("data/celegans.pt")
    elif (dataset == "arena"):
        #size = 1135
        S = torch.load("data/arena.pt")
    elif (dataset == "douban"):
        #size = 3906
        S = torch.load("data/douban.pt")
    else:
        filepath = "data/" + dataset + ".npz"
        loader = load_npz(filepath)
        data = loader["adj_matrix"]
        samples = data.shape[0]
        features = data.shape[1]
        values = data.data
        coo_data = data.tocoo()
        indices = torch.LongTensor([coo_data.row, coo_data.col])
        S = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features]).to_dense()
        if (not torch.all(S.transpose(0, 1) == S)):
            S = torch.add(S, S.transpose(0, 1))
        S = S.int()
        ones = torch.ones_like(S)
        S = torch.where(S > 1, ones, S)
    return S

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if(args.setting == "transfer"):
        train_set = ["celegans","arena","douban","cora"]
        test_set = ["celegans","arena","douban","cora","dblp","coauthor_cs"]
    else:
        train_set = [args.dataset]
        test_set = [args.dataset]
    print("Using training sets: ", end = "")
    print(train_set)
    probability_model = args.probability_model
    training_perturbation_level = args.training_perturbation_level
    testing_perturbation_levels = args.testing_perturbation_levels
    no_training_samples_per_graph = args.no_training_samples_per_graph
    no_testing_samples_each_level = args.no_testing_samples_each_level
    NUM_HIDDEN_LAYERS = args.NUM_HIDDEN_LAYERS
    HIDDEN_DIM = args.HIDDEN_DIM
    output_feature_size = args.output_feature_size
    lr = args.lr
    epoch = args.epoch
    encoder = args.encoder
    if(encoder == "GIN"):
        use_input_augmentation = True
        use_output_augmentation = False
    else:
        use_input_augmentation = True
        use_output_augmentation = True
    print("Loading training datasets")

    train_loader = {}
    original_graph_loader = {}
    for dataset in [*set(train_set+test_set)]:
        original_graph_loader[dataset] = load_adj(dataset)
    print("Generating Training Perturbations")
    for dataset in train_set:
        train_loader[dataset] = generate_purturbations(device, original_graph_loader[dataset],
                                                        perturbation_level = training_perturbation_level,
                                                        no_samples=no_training_samples_per_graph,
                                                       method = probability_model)
    model = GAE(NUM_HIDDEN_LAYERS,
               7,
               HIDDEN_DIM,
               output_feature_size, activation=F.relu,
                use_input_augmentation=use_input_augmentation,
                use_output_augmentation=use_output_augmentation,
                encoder=encoder).to(device)

    print("Generating training features")
    train_features = {}
    for dataset in train_loader.keys():
        train_features[dataset] = generate_features(train_loader[dataset])

    print("Fitting model")
    fit_GAE(len(train_set)*(no_training_samples_per_graph+1),model,epoch, train_loader, train_features, device, lr)
    print("Fitting finished")
    model.eval()

    print("Starting testing")
    print("Data set used for training is: ", end = "")
    for dataset in train_set:
        print(dataset, end = "\t")
    print()
    print("Data set used only for testing is: ", end = "")
    for dataset in test_set:
        print(dataset, end="\t")
    print()
    print("Training perturbation level: ",end = "")
    print(training_perturbation_level)

    for dataset in test_set:
        print("Results for " + dataset)
        S = original_graph_loader[dataset]

        adj = coo_matrix(S.numpy())
        adj_norm = preprocess_graph(adj)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2])).to(device)
        print("Generating S embedding")
        S_feat = generate_features([S] )[0]
        features = csr_matrix(S_feat)
        features = sparse_to_tuple(features.tocoo())
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).to(device)
        S_emb = model(features, S_feat.to(device), adj_norm).detach()
        print("Generating testing perturbations")
        S_hat_samples, S_prime_samples, p_samples = gen_test_set(device, S, no_testing_samples_each_level,
                                            testing_perturbation_levels, method=probability_model)
        for level in testing_perturbation_levels:
            print("Level of perturbation: " + str(level))
            print("Generating testing features")
            S_hat_features = generate_features(S_hat_samples[str(level)])
            test_matching(model, S_hat_samples[str(level)], p_samples[str(level)], S_hat_features, S_emb, device)