import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from .base_trainer import BaseTrainer

class SIGNET_Trainer(BaseTrainer):
    def __init__(self, model, device, semi=False):
        super().__init__(model, device)
        self.model = self.model.to(self.device)
        self.semi = semi

    def compute_loss(self, x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    def train(self, train_loader, optimizer, **kwargs):
        self.model.train()
        loss_all = 0
        num_sample = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(self.device)
            y, y_hyper, _, _ = self.model(data)
            loss = self.compute_loss(y, y_hyper).mean()
            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
            loss.backward()
            optimizer.step()
        return loss_all / num_sample

    def evaluate(self, data_loader, normal_class, **kwargs):
        self.model.eval()
        score_list, label_list = [], []
        for data in data_loader:
            data = data.to(self.device)
            with torch.no_grad():
                y, y_hyper, _, _ = self.model(data)
                ano_score = self.compute_loss(y, y_hyper)
            label_list.append(data.y)
            score_list.append(ano_score)

        label_list = torch.cat(label_list).cpu().numpy()
        score_list = torch.cat(score_list).cpu().numpy()
        
        binary_labels = (label_list != normal_class).astype(int)
        
        roc_auc = roc_auc_score(binary_labels, score_list)
        pr_auc = average_precision_score(binary_labels, score_list)
        return roc_auc, pr_auc

    def evaluate_explanation(self, explain_loader):
        self.model.eval()
        all_node_explain_true = []
        all_node_explain_score = []
        all_edge_explain_true = []
        all_edge_explain_score = []
        for data in explain_loader:
            data = data.to(self.device)
            with torch.no_grad():
                node_score = self.model.explainer(data.x, data.edge_index, data.batch)
                edge_score = self.model.lift_node_score_to_edge_score(node_score, data.edge_index)
            all_node_explain_true.append(data.node_label.cpu())
            all_node_explain_score.append(node_score.cpu())
            all_edge_explain_true.append(data.edge_label.cpu())
            all_edge_explain_score.append(edge_score.cpu())

        x_node_true = torch.cat(all_node_explain_true)
        x_node_score = torch.cat(all_node_explain_score)
        x_node_auc = roc_auc_score(x_node_true, x_node_score)

        x_edge_true = torch.cat(all_edge_explain_true)
        x_edge_score = torch.cat(all_edge_explain_score)
        x_edge_auc = roc_auc_score(x_edge_true, x_edge_score)

        return x_node_auc, x_edge_auc
