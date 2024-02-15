from robust_GAE import *
from networkx import read_edgelist
from scipy.io import loadmat
from model import *
import args

def load_douban():
	x = loadmat("/data/douban.mat")
	return (x['online_edge_label'][0][1],
			x['online_node_label'],
			x['offline_edge_label'][0][1],
			x['offline_node_label'],
			x['ground_truth'].T)

def fit_GAE_real(data, no_samples, GAE, epoch, train_loader, train_features, device, lr, test_pairs):
    optimizer = Adam(GAE.parameters(), lr=lr,weight_decay=5e-4)
    for step in range(epoch):
        print("Hit rate for training epoch "+ str(step+1))
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

        keys = list(train_loader.keys())
        S1 = train_loader[keys[0]][0]
        S2 = train_loader[keys[1]][0]
        adj_S1 = coo_matrix(S1.numpy())
        adj_norm_1 = preprocess_graph(adj_S1)
        adj_norm_1 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_1[0].T),
                                              torch.FloatTensor(adj_norm_1[1]),
                                              torch.Size(adj_norm_1[2])).to(device)
        adj_S2 = coo_matrix(S2.numpy())
        adj_norm_2 = preprocess_graph(adj_S2)
        adj_norm_2 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_2[0].T),
                                              torch.FloatTensor(adj_norm_2[1]),
                                              torch.Size(adj_norm_2[2])).to(device)
        if (data == "ACM_DBLP"):
            S1_feat = train_features["ACM"][0]
            S2_feat = train_features["DBLP"][0]
        elif (data == "douban_real"):
            S1_feat = train_features["Online"][0]
            S2_feat = train_features["Offline"][0]
        S1_features = csr_matrix(S1_feat)
        S1_features = sparse_to_tuple(S1_features.tocoo())
        S1_features = torch.sparse.FloatTensor(torch.LongTensor(S1_features[0].T),
                                               torch.FloatTensor(S1_features[1]),
                                               torch.Size(S1_features[2])).to(device)
        S2_features = csr_matrix(S2_feat)
        S2_features = sparse_to_tuple(S2_features.tocoo())
        S2_features = torch.sparse.FloatTensor(torch.LongTensor(S2_features[0].T),
                                               torch.FloatTensor(S2_features[1]),
                                               torch.Size(S2_features[2])).to(device)

        S1_emb = GAE(S1_features, S1_feat.to(device), adj_norm_1).detach()
        S2_emb = GAE(S2_features, S2_feat.to(device), adj_norm_2).detach()

        D = torch.cdist(S1_emb, S2_emb, 2)
        if (data == "ACM_DBLP"):
            test_idx = test_pairs[:, 0].astype(np.int32)
            labels = test_pairs[:, 1].astype(np.int32)
        elif (data == "douban_real"):
            test_idx = test_pairs[0, :].astype(np.int32)
            labels = test_pairs[1, :].astype(np.int32)
        hitAtOne = 0
        hitAtFive = 0
        hitAtTen = 0
        hitAtFifty = 0
        hitAtHundred = 0
        for i in tqdm(range(len(test_idx))):
            dist_list = D[test_idx[i]]
            sorted_neighbors = torch.argsort(dist_list).cpu()
            label = labels[i]
            for j in range(100):
                if (sorted_neighbors[j].item() == label):
                    if (j == 0):
                        hitAtOne += 1
                        hitAtFive += 1
                        hitAtTen += 1
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 4):
                        hitAtFive += 1
                        hitAtTen += 1
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 9):
                        hitAtTen += 1
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 49):
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 100):
                        hitAtHundred += 1
                        break
        print("Hit@1: ", end="")
        print(hitAtOne / len(test_idx))
        print("Hit@5: ", end="")
        print(hitAtFive / len(test_idx))
        print("Hit@10: ", end="")
        print(hitAtTen / len(test_idx))
        print("Hit@50: ", end="")
        print(hitAtFifty / len(test_idx))
        print("Hit@100: ", end="")
        print(hitAtHundred / len(test_idx))
        print()

if __name__ =="__main__":
    data = args.subgraph_dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    train_features = {}
    if (data == "ACM_DBLP"):
        train_set = ["ACM", "DBLP"]
        test_set = ["ACM_DBLP"]
        input_dim = 17
        b = np.load('/data/ACM-DBLP.npz')
        train_features["ACM"] = [torch.from_numpy(b["x1"]).float()]
        train_features["DBLP"] = [torch.from_numpy(b["x2"]).float()]
        test_pairs = b['test_pairs'].astype(np.int32)
    elif (data == "douban_real"):
        a1, f1, a2, f2, test_pairs = load_douban()
        a1 = a1.A
        a2 = a2.A
        f1 = f1.A
        f2 = f2.A
        train_set = ["Online", "Offline"]
        test_set = ["douban_real"]
        input_dim = 538
        test_pairs = torch.tensor(np.array(test_pairs, dtype=int)) - 1
        test_pairs = test_pairs.numpy()
        train_features["Online"] = [torch.from_numpy(f1).float()]
        train_features["Offline"] = [torch.from_numpy(f2).float()]
    print("Using training sets: ", end="")
    print(train_set)
    NUM_HIDDEN_LAYERS = 6
    HIDDEN_DIM = 512
    output_feature_size = 512
    lr = 0.0001
    epoch = 20
    encoder = "GIN"
    if (encoder == "GIN"):
        use_input_augmentation = True
        use_output_augmentation = True
    else:
        use_input_augmentation = False
        use_output_augmentation = False
    print("Loading training datasets")
    train_loader = {}
    for dataset in train_set:
        train_loader[dataset] = [load_adj(dataset)]
    model = GAE(NUM_HIDDEN_LAYERS,
                input_dim,
                HIDDEN_DIM,
                output_feature_size, activation=F.relu,
                use_input_augmentation=use_input_augmentation,
                use_output_augmentation=use_output_augmentation,
                encoder=encoder).to(device)
    print("Generating training features")
    print("Fitting model")
    fit_GAE_real(data, len(train_set) * (1 + 1), model, epoch, train_loader, train_features, device,
            lr,test_pairs)
    print("Fitting finished")
    model.eval()

    print("Starting testing")
    print("Data set used for training is: ", end="")
    for dataset in train_set:
        print(dataset, end="\t")
    print()
    print("Data set used only for testing is: ", end="")
    for dataset in test_set:
        print(dataset, end="\t")
    print()

    print("Results for " + data)
    if(data == "ACM_DBLP"):
        S1 = torch.load("/data/ACM.pt")
        S2 = torch.load("/data/DBLP.pt")
    elif(data == "douban_real"):
        S1 = torch.load("/data/online.pt")
        S2 = torch.load("/data/offline.pt")
    print("Generating S1,S2 embedding")

    adj_S1 = coo_matrix(S1.numpy())
    adj_norm_1 = preprocess_graph(adj_S1)
    adj_norm_1 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_1[0].T),
                                    torch.FloatTensor(adj_norm_1[1]),
                                    torch.Size(adj_norm_1[2])).to(device)
    adj_S2 = coo_matrix(S2.numpy())
    adj_norm_2 = preprocess_graph(adj_S2)
    adj_norm_2 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_2[0].T),
                                      torch.FloatTensor(adj_norm_2[1]),
                                      torch.Size(adj_norm_2[2])).to(device)
    if(data == "ACM_DBLP"):
        S1_feat = torch.from_numpy(b["x1"]).float()
        S2_feat = torch.from_numpy(b["x2"]).float()
    elif(data == "douban_real"):
        S1_feat = train_features["Online"][0]
        S2_feat = train_features["Offline"][0]
    S1_features = csr_matrix(S1_feat)
    S1_features = sparse_to_tuple(S1_features.tocoo())
    S1_features = torch.sparse.FloatTensor(torch.LongTensor(S1_features[0].T),
                                    torch.FloatTensor(S1_features[1]),
                                    torch.Size(S1_features[2])).to(device)
    S2_features = csr_matrix(S2_feat)
    S2_features = sparse_to_tuple(S2_features.tocoo())
    S2_features = torch.sparse.FloatTensor(torch.LongTensor(S2_features[0].T),
                                           torch.FloatTensor(S2_features[1]),
                                           torch.Size(S2_features[2])).to(device)

    S1_emb = model(S1_features, S1_feat.to(device), adj_norm_1).detach()
    S2_emb = model(S2_features, S2_feat.to(device), adj_norm_2).detach()
    print("Calculating Matching")
    D = torch.cdist(S1_emb, S2_emb, p=2)
    if(data == "ACM_DBLP"):
        test_idx = test_pairs[:, 0].astype(np.int32)
        labels = test_pairs[:, 1].astype(np.int32)
    elif(data == "douban_real"):
        test_idx = test_pairs[0, :].astype(np.int32)
        labels = test_pairs[1, :].astype(np.int32)
    print()
    print("Final hit rate:")
    hitAtOne = 0
    hitAtFive = 0
    hitAtTen = 0
    hitAtFifty = 0
    hitAtHundred = 0
    for i in tqdm(range(len(test_idx))):
        dist_list = D[test_idx[i]]
        sorted_neighbors = torch.argsort(dist_list).cpu()
        label = labels[i]
        for j in range(100):
            if(sorted_neighbors[j].item()==label):
                if(j == 0):
                    hitAtOne += 1
                    hitAtFive += 1
                    hitAtTen += 1
                    hitAtFifty += 1
                    hitAtHundred += 1
                    break
                elif(j <= 4):
                    hitAtFive += 1
                    hitAtTen += 1
                    hitAtFifty += 1
                    hitAtHundred += 1
                    break
                elif(j <= 9):
                    hitAtTen += 1
                    hitAtFifty += 1
                    hitAtHundred += 1
                    break
                elif(j<=49):
                    hitAtFifty += 1
                    hitAtHundred += 1
                    break
                elif(j <= 100):
                    hitAtHundred += 1
                    break


    print("Hit@1: ",end="")
    print(hitAtOne/len(test_idx))
    print("Hit@5: ", end="")
    print(hitAtFive/len(test_idx))
    print("Hit@10: ", end="")
    print(hitAtTen/len(test_idx))
    print("Hit@50: ", end="")
    print(hitAtFifty/len(test_idx))
    print("Hit@100: ", end="")
    print(hitAtHundred/len(test_idx))