import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from evaluate import evaluate
from embedder import embedder
from layers import GCN, InterDiscriminator


class HDMI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.coef_l = self.args.coef_layers
        self.coef_f = self.args.coef_fusion
        self.criteria = nn.BCEWithLogitsLoss()

        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]

        print("Started training...")
        model = modeler(self.args.ft_size, self.args.hid_units, len(adj_list), self.args.same_discriminator)\
            .to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        cnt_wait = 0
        best = 1e9
        model.train()
        for _ in tqdm(range(self.args.nb_epochs)):
            optimiser.zero_grad()

            # corruption function
            idx = np.random.permutation(self.args.nb_nodes)
            shuf_fts = features[idx, :].to(self.args.device)

            logits_e_list, logits_i_list, logits_j_list, logits_e_fusion, logits_i_fusion, logits_j_fusion \
                = model(features, shuf_fts, adj_list, self.args.sparse)

            # loss
            # layers
            loss_e = loss_i = loss_j = 0
            for i in range(len(logits_e_list)):
                loss_e += self.get_loss(logits_e_list[i])
                loss_i += self.get_loss(logits_i_list[i])
                loss_j += self.get_loss(logits_j_list[i])

            # fusion
            loss_e_fusion = self.get_loss(logits_e_fusion)
            loss_i_fusion = self.get_loss(logits_i_fusion)
            loss_j_fusion = self.get_loss(logits_j_fusion)

            loss = self.coef_l[0] * loss_e + self.coef_l[1] * loss_i + self.coef_l[2] * loss_j + \
                self.coef_f[0] * loss_e_fusion + self.coef_f[1] * loss_i_fusion + self.coef_f[2] * loss_j_fusion

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset,
                                                                                   self.args.embedder))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                break

            loss.backward()
            optimiser.step()

        print(loss_e, loss_i, loss_j, loss_e_fusion, loss_i_fusion, loss_j_fusion)

        # save
        model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.embedder)))

        # evaluation
        print("Evaluating...")
        model.eval()
        embeds = model.embed(features, adj_list, self.args.sparse)
        macro_f1s, micro_f1s, k1 = evaluate(embeds, self.idx_train, self.idx_val, self.idx_test, self.labels,)
        return macro_f1s, micro_f1s, k1

    def get_loss(self, logits):
        """
        :param logits: [2, n_nodes]
        """
        n_nodes = logits.shape[1]
        lbl_1 = torch.ones(n_nodes)
        lbl_2 = torch.zeros(n_nodes)
        lbl = torch.stack((lbl_1, lbl_2), 0)

        lbl = lbl.to(self.args.device)
        loss = self.criteria(logits, lbl)
        return loss


class modeler(nn.Module):
    def __init__(self, ft_size, hid_units, n_networks, same_discriminator=True):
        super(modeler, self).__init__()
        self.gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
        self.w_list = nn.ModuleList([nn.Linear(hid_units, hid_units, bias=False) for _ in range(n_networks)])
        self.y_list = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(n_networks)])

        self.disc_layers = InterDiscriminator(hid_units, ft_size)
        if same_discriminator:
            self.disc_fusion = self.disc_layers
        else:
            self.disc_fusion = InterDiscriminator(hid_units, ft_size)

        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def forward(self, seq1, seq2, adj_list, sparse):
        h_1_list = []
        h_2_list = []
        c_list = []

        logits_e_list = []
        logits_i_list = []
        logits_j_list = []
        for i, adj in enumerate(adj_list):
            # real samples
            h_1 = torch.squeeze(self.gcn_list[i](seq1, adj, sparse))
            h_1_list.append(h_1)
            c = torch.squeeze(torch.mean(h_1, 0))   # readout
            c_list.append(c)

            # negative samples
            h_2 = torch.squeeze(self.gcn_list[i](seq2, adj, sparse))
            h_2_list.append(h_2)

            # discriminator
            logits_e, logits_i, logits_j = self.disc_layers(c, h_1, h_2, seq1, seq2)
            logits_e_list.append(logits_e)
            logits_i_list.append(logits_i)
            logits_j_list.append(logits_j)

        # fusion
        h1 = self.combine_att(h_1_list)
        h2 = self.combine_att(h_2_list)
        c = torch.mean(h1, 0)   # readout
        logits_e_fusion, logits_i_fusion, logits_j_fusion = self.disc_fusion(c, h1, h2, seq1, seq2)

        return logits_e_list, logits_i_list, logits_j_list, logits_e_fusion, logits_i_fusion, logits_j_fusion

    # Detach the return variables
    def embed(self, seq, adj_list, sparse):
        h_1_list = []
        for i, adj in enumerate(adj_list):
            h_1 = torch.squeeze(self.gcn_list[i](seq, adj, sparse))
            h_1_list.append(h_1)
        h = self.combine_att(h_1_list)
        return h.detach()
