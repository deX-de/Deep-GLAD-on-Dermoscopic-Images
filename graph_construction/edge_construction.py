import numpy as np
import torch
from skimage import graph
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import time

def construct_edges(segments, mean_color, node_pos, segment_map, config):
    """Constructs edges based on configuration (RAG, kNN)."""
    edge_config = config.get('edges', {})
    num_nodes = node_pos.shape[0]
    num_real_nodes = len(np.unique(segments[segments!=0]))
    num_virtual_nodes = num_nodes - num_real_nodes
    edge_indices = {}
    edge_attrs = {}
    times = {}
    # RAG Edges
    if edge_config.get('rag', {}).get('enabled', True):
        start_time = time.time()
        
        rag_edge_index = _construct_rag(segments, edge_config.get('rag', {}).get('connectivity', 1))
        rag_edge_index_mapped = _map_rag_indices(rag_edge_index, segment_map)
        if rag_edge_index_mapped is not None:
            if num_virtual_nodes > 0:
                rag_edge_index_mapped = _add_virtual_node_edges(rag_edge_index_mapped, num_real_nodes, num_nodes)
            edge_indices['rag'] = rag_edge_index_mapped
            edge_attrs['rag'] = _calculate_relative_coords(node_pos, rag_edge_index_mapped)
        
        end_time = time.time()
        times['rag'] = end_time - start_time
    # kNN Edges
    if edge_config.get('knn_spatial', {}).get('enabled', True):
        start_time = time.time()
        
        pos_tensor = torch.from_numpy(node_pos).float()
        k = min(edge_config.get('knn_spatial', {}).get('k', 8), num_nodes - 1)
        if k > 0:
            # Only apply k-NN on segmentation nodes (skip virtual nodes)
            knn_edge_index = knn_graph(pos_tensor[:-num_virtual_nodes or None], k=k, loop=False)
            knn_edge_index = to_undirected(knn_edge_index, num_nodes=num_nodes)
            if num_virtual_nodes > 0:
                knn_edge_index = _add_virtual_node_edges(knn_edge_index, num_real_nodes, num_nodes)
            edge_indices['knn_spatial'] = knn_edge_index
            edge_attrs['knn_spatial'] = _calculate_relative_coords(node_pos, knn_edge_index)
            
        end_time = time.time()
        times['knn_spatial'] = end_time - start_time

    if edge_config.get('knn_spatial_color', {}).get('enabled', True):
        start_time = time.time()
        
        # Turn distance measure in KNN to:
        # d_spatial_rgb = sqrt( (x_i - x_j)^2 / 2 + (y_i - y_j)^2 / 2 + (r_i - r_j)^2 / 3 + (g_i - g_j)^2 / 3 + (b_i - b_j)^2 / 3 )
        # d_spatial_grayscale = sqrt( (x_i - x_j)^2 / 2 + (y_i - y_j)^2 / 2 + (I_i - I_j)^2 )
        if mean_color.max() <= 1.0:
            mean_color = mean_color * 255
        color_tensor = torch.from_numpy(mean_color / np.sqrt(mean_color.shape[1])).float() 
        pos_tensor = torch.from_numpy(node_pos / np.sqrt(node_pos.shape[1])).float() 
        feature_tensor = torch.cat([pos_tensor, color_tensor], dim=1)
        k = min(edge_config.get('knn_spatial_color', {}).get('k', 8), num_nodes - 1)
        if k > 0:
            # Only apply k-NN on segmentation nodes (skip virtual nodes)
            knn_edge_index = knn_graph(feature_tensor[:-num_virtual_nodes or None], k=k, loop=False)
            knn_edge_index = to_undirected(knn_edge_index, num_nodes=num_nodes)
            if num_virtual_nodes > 0:
                knn_edge_index = _add_virtual_node_edges(knn_edge_index, num_real_nodes, num_nodes)
            edge_indices['knn_spatial_color'] = knn_edge_index
            edge_attrs['knn_spatial_color'] = _calculate_relative_coords(node_pos, knn_edge_index)
            
        end_time = time.time()
        times['knn_spatial_color'] = end_time - start_time
        
    return edge_indices, edge_attrs, times

def _add_virtual_node_edges(edge_index, num_real_nodes, num_nodes):
    # Global virtual node (last node) connected to all other nodes
    global_to_all = torch.cat([
        torch.full((1, num_nodes - 1), num_nodes - 1),
        torch.arange(num_nodes - 1).unsqueeze(0)
    ], dim=0)
    all_to_global = torch.stack([global_to_all[1], global_to_all[0]], dim=0)

    # Background virtual node (second to last node) connected only to global node
    if num_nodes - num_real_nodes > 1:
        background_to_global = torch.tensor([[num_nodes - 2], [num_nodes - 1]])
        global_to_background = torch.tensor([[num_nodes - 1], [num_nodes - 2]])
        virtual_edges = torch.cat([global_to_all, all_to_global, background_to_global, global_to_background], dim=1)
    else:
        virtual_edges = torch.cat([global_to_all, all_to_global], dim=1)

    # Combine with existing edges
    combined_edges = torch.cat([edge_index, virtual_edges], dim=1)

    # Remove duplicates
    combined_edges = torch.unique(combined_edges, dim=1)

    return combined_edges

def _construct_rag(segments, connectivity, cumstom_rag=False):
    # Create a RAG from the segmentation and get the edges
    if cumstom_rag:
        edges = custom_rag_boundary(segments, connectivity=connectivity)
        edges = np.array(edges)
    else:
        rag_graph = graph.rag_boundary(segments, np.ones_like(segments, dtype=float), connectivity=connectivity)
        edges = np.array(list(rag_graph.edges))
    
    # Filter out edges involving background (0)
    valid_edge_mask = (edges[:, 0] > 0) & (edges[:, 1] > 0)
    edges = edges[valid_edge_mask]
    
    # Create bidirectional edges by adding reverse direction
    reverse_edges = edges[:, [1, 0]]  # Swap columns
    bidirectional_edges = np.vstack([edges, reverse_edges])
    
    # Convert to PyTorch tensor
    return torch.tensor(bidirectional_edges.T, dtype=torch.long)

def custom_rag_boundary(segments, connectivity=1):
    rows, cols = segments.shape
    
    max_distance = np.sqrt(connectivity)
    
    # Generate all possible pixel coordinate pairs within max_distance
    offsets = []
    for dr in range(-int(max_distance)-1, int(max_distance)+2):
        for dc in range(-int(max_distance)-1, int(max_distance)+2):
            if dr == 0 and dc == 0:
                continue
            distance = np.sqrt(dr*dr + dc*dc)
            if distance <= max_distance + 1e-10: 
                offsets.append((dr, dc))
    edges = set()
    
    # For each pixel, check its neighbors within the connectivity distance
    for r in range(rows):
        for c in range(cols):
            current_segment = segments[r, c]
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_segment = segments[nr, nc]
                    
                    # If different segments, add edge
                    if current_segment != neighbor_segment:
                        edge = tuple(sorted([current_segment, neighbor_segment]))
                        edges.add(edge)
    return sorted(edges)


def _map_rag_indices(rag_edge_index_seg_ids, segment_map):
    """Maps edge index from segment IDs to 0-based node indices."""
    if rag_edge_index_seg_ids.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Create a mapping from segment ID to node index (0 to N-1)
    # segment_map is assumed to be the array of unique_segment_ids
    # E.g., if unique_segments = [1, 3, 4], then seg_id 1 -> idx 0, 3 -> 1, 4 -> 2
    seg_id_to_node_idx = {seg_id: i for i, seg_id in enumerate(segment_map)}

    mapped_rows = []
    mapped_cols = []
    valid_edge = []

    for i in range(rag_edge_index_seg_ids.shape[1]):
        src_seg_id = rag_edge_index_seg_ids[0, i].item()
        dst_seg_id = rag_edge_index_seg_ids[1, i].item()

        if src_seg_id in seg_id_to_node_idx and dst_seg_id in seg_id_to_node_idx:
             mapped_rows.append(seg_id_to_node_idx[src_seg_id])
             mapped_cols.append(seg_id_to_node_idx[dst_seg_id])
             valid_edge.append(True)
        else:
             # This should not happen if segment_map is correct, but handle defensively
             valid_edge.append(False)
             print(f"Warning: RAG edge segment ID not found in map: ({src_seg_id}, {dst_seg_id})")


    if not mapped_rows:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([mapped_rows, mapped_cols], dtype=torch.long)


def _calculate_relative_coords(node_pos, edge_index):
    """Calculates relative coordinates (dx, dy) for edge attributes."""
    if edge_index.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float)

    pos_tensor = torch.from_numpy(node_pos).float()
    row, col = edge_index # row = source, col = target
    pos_row = pos_tensor[row]
    pos_col = pos_tensor[col]
    cart = pos_row - pos_col
    
    # normalize to [0, 1]
    max_val = float(cart.abs().max())
    interval = (0.0, 1.0)
    length = interval[1] - interval[0]
    center = (interval[0] + interval[1]) / 2
    cart = length * cart / (2 * max_val) + center
    
    return cart.float()