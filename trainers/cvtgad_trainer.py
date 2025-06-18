import torch
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, average_precision_score
from .base_trainer import BaseTrainer

class CVTGAD_Trainer(BaseTrainer):    
    def __init__(self, model, device, semi=False):
        super().__init__(model, device)
        self.model = self.model.to(self.device)
        self.semi = semi
    
    def train(self, train_loader, optimizer, **kwargs):
        self.model.train()
        loss_all = 0
        num_sample = 0
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            g_f, g_s, n_f, n_s = self.model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
            loss_g, loss_n = self.compute_loss(g_f, g_s, n_f, n_s, data.batch)
            loss = loss_g.mean() + loss_n.mean()
            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
            loss.backward()
            optimizer.step()
        return loss_all / num_sample
    
    def evaluate(self, data_loader, normal_class=0):
        self.model.eval()
        score_list, label_list = [], []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                g_f, g_s, n_f, n_s = self.model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                y_score_g, y_score_n = self.compute_loss(g_f, g_s, n_f, n_s, data.batch)
                y_score = y_score_g + y_score_n
                
                # Append losses and labels
                score_list.append(y_score)
                label_list.append(data.y)

        # Concatenate all scores and labels
        score_list = torch.cat(score_list).cpu().numpy()
        label_list = torch.cat(label_list).cpu().numpy()

        # Create binary labels: 0 for normal, 1 for anomaly
        binary_labels = (label_list != normal_class).astype(int)

        # Calculate ROC-AUC and PR-AUC scores
        roc_auc = roc_auc_score(binary_labels, score_list)
        pr_auc = average_precision_score(binary_labels, score_list)

        return roc_auc, pr_auc
    
    def compute_loss(self, g_f, g_s, n_f, n_s, batch, *args, **kwargs):
        loss_g = self.calc_loss_g(g_f, g_s)
        loss_n = self.calc_loss_n(n_f, n_s, batch)
        return loss_g, loss_n
        
    def calc_loss_n(self, x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        node_belonging_mask = batch.repeat(batch_size, 1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss

    def calc_loss_g(self, x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss