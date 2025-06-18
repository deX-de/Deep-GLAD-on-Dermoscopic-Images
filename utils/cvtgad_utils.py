import torch
import scipy.sparse as sp
from torch_geometric.utils import degree, to_scipy_sparse_matrix

def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)         
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()    

        Dinv = sp.diags(D)                                                      
        RW = A * Dinv                                                           
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]                         
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE, dim=-1)                                        

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1
 
        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)                               

    return gs