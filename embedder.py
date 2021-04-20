import torch
from utils import process


class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_dblp(args.sc)
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_acm_mat()
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_imdb(args.sc)
        if args.dataset == "amazon":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_amazon(args.sc)

        features = process.preprocess_features(features)

        args.nb_nodes = adj_list[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = labels.shape[1]

        adj_list = [process.normalize_adj(adj) for adj in adj_list]
        self.adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.args = args
