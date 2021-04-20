import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from embedder import embedder
from evaluate import evaluate
from layers import GCN, InterDiscriminator


class HDI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.coef_l = self.args.coef_layers
        self.criteria = nn.BCEWithLogitsLoss()

    def training(self):
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]

        print("Started training...")
        print("The number of layers: {}".format(len(adj_list)))
        final_embs = []
        for n_adj, adj in enumerate(adj_list):
            print("Layer {}".format(n_adj))
            model = modeler(self.args.ft_size, self.args.hid_units).to(self.args.device)
            optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            cnt_wait = 0
            best = 1e9
            for _ in tqdm(range(self.args.nb_epochs)):
                model.train()
                optimiser.zero_grad()

                # corruption function
                idx = np.random.permutation(self.args.nb_nodes)
                shuf_fts = features[idx, :].to(self.args.device)

                logits_e, logits_i, logits_j = model(features, shuf_fts, adj, self.args.sparse)

                # loss
                loss_e = self.get_loss(logits_e)
                loss_i = self.get_loss(logits_i)
                loss_j = self.get_loss(logits_j)
                loss = self.coef_l[0] * loss_e + self.coef_l[1] * loss_i + self.coef_l[2] * loss_j

                # early stop
                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(
                        self.args.dataset, self.args.embedder, n_adj))
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    print("Early stopped!")
                    break

                loss.backward()
                optimiser.step()

            # get embedding
            model.eval()
            embeds = model.embed(features, adj, self.args.sparse)
            final_embs.append(embeds)

        # final evaluation
        print("Evaluating...")
        final_embs = torch.mean(torch.stack(final_embs), 0)   # average pooling
        macro_f1s, micro_f1s, k1 = evaluate(final_embs, self.idx_train, self.idx_val, self.idx_test, self.labels)
        return macro_f1s, micro_f1s, k1

    def get_loss(self, logits):
        """
        :param logits: [2, n_nodes]
        """
        n_nodes = logits.shape[1]
        lbl_1 = torch.ones(n_nodes)
        lbl_2 = torch.zeros(n_nodes)
        lbl = torch.stack((lbl_1, lbl_2))

        lbl = lbl.to(self.args.device)
        loss = self.criteria(logits, lbl)
        return loss


class modeler(nn.Module):
    def __init__(self, ft_size, hid_units):
        super(modeler, self).__init__()
        self.gcn = GCN(ft_size, hid_units)
        self.disc = InterDiscriminator(hid_units, ft_size)

    def forward(self, seq1, seq2, adj, sparse):
        # real
        h_1 = torch.squeeze(self.gcn(seq1, adj, sparse))
        c = torch.squeeze(torch.mean(h_1, 0))  # readout

        # negative
        h_2 = torch.squeeze(self.gcn(seq2, adj, sparse))

        # discriminator
        ret = self.disc(c, h_1, h_2, seq1, seq2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = torch.squeeze(self.gcn(seq, adj, sparse))
        return h_1.detach()
