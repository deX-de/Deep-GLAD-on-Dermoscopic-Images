import yaml
import argparse
from datasets import ImageGraphDataset
from collections import Counter
import numpy as np
from datasets import create_semi_supervised_dataloaders

def print_label_distribution(dataset, name):
    labels = dataset.y.tolist()
    label_counts = Counter(labels)
    total = len(labels)
    
    print(f"\n{name} Label Distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")
        
def print_dataloader_label_distribution(dataloader, name):
    label_counts = Counter()
    for batch in dataloader:
        labels = batch.y.tolist()
        label_counts.update(labels)
    
    total = sum(label_counts.values())
    print(f"\n{name} Label Distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")
        
def print_node_statistics(dataset, name):
    num_nodes = [data.num_nodes for data in dataset]
    min_nodes = min(num_nodes)
    avg_nodes = np.mean(num_nodes)
    max_nodes = max(num_nodes)

    print(f"\n{name} Node Statistics:")
    print(f"Minimum number of nodes: {min_nodes}")
    print(f"Average number of nodes: {avg_nodes:.2f}")
    print(f"Maximum number of nodes: {max_nodes}")
    return avg_nodes
    
def print_edge_statistics(dataset, name):
    print(f"\n{name} Edge Statistics:")
    if getattr(dataset.get(0), "rag_edge_index", None) is not None:
        rag_edges = [data.rag_edge_index.shape[1] for data in dataset]
        print("RAG Edges:")
        print(f"  Minimum: {min(rag_edges)}")
        print(f"  Average: {np.mean(rag_edges):.2f}")
        print(f"  Maximum: {max(rag_edges)}")
    if getattr(dataset.get(0), "knn_spatial_edge_index", None) is not None:
        knn_edges = [data.knn_spatial_edge_index.shape[1] for data in dataset]
        print("KNN Spatial Edges:")
        print(f"  Minimum: {min(knn_edges)}")
        print(f"  Average: {np.mean(knn_edges):.2f}")
        print(f"  Maximum: {max(knn_edges)}")        
    if getattr(dataset.get(0), "knn_spatial_color_edge_index", None) is not None:
        knn_edges = [data.knn_spatial_color_edge_index.shape[1] for data in dataset]
        print("KNN Spatial-Color Edges:")
        print(f"  Minimum: {min(knn_edges)}")
        print(f"  Average: {np.mean(knn_edges):.2f}")
        print(f"  Maximum: {max(knn_edges)}") 
        
def print_time_statistics(dataset, name):
    if not hasattr(dataset, "times") or dataset.times is None:
        return
    
    times = dataset.times
    num_images = len(dataset)

    print(f"\n{name} Time Statistics:")

    seg_times = times.get('segments', 0)
    feat_times = times.get('features', 0)
    edge_times = times.get('edges', 0)

    print("Segmentation Times")
    print(f"  {'Total:':<12} {seg_times:>10.4f}s")
    print(f"  {'Avg / img:':<12} {seg_times / num_images:>10.4f}s\n")

    print("Feature Extraction Times")
    print(f"  {'Total:':<12} {feat_times:>10.4f}s")
    print(f"  {'Avg / img:':<12} {feat_times / num_images:>10.4f}s")
    indiv_features = times.get('individual_features', {})
    if indiv_features:
        print(f"  {'By Feature Type:':<12}")
        max_name_len = max(len(name) for name in indiv_features)
        for edge_type, total_time in indiv_features.items():
            avg_time = total_time / num_images
            print(f"    {edge_type:<{max_name_len}} : Total {total_time:>10.4f}s,   Avg {avg_time:>10.4f}s")
    
    print("\nEdge Construction Times")
    print(f"  {'Total:':<12} {edge_times:>10.4f}s")
    print(f"  {'Avg / img:':<12} {edge_times / num_images:>10.4f}s")

    indiv_edges = times.get('individual_edges', {})
    if indiv_edges:
        print(f"  {'By Edge Type:':<12}")
        max_name_len = max(len(name) for name in indiv_edges)
        for edge_type, total_time in indiv_edges.items():
            avg_time = total_time / num_images
            print(f"    {edge_type:<{max_name_len}} : Total {total_time:>10.4f}s,   Avg {avg_time:>10.4f}s")

    print(f"\n{'Total Time:':<14} {times.get('total', seg_times + feat_times + edge_times):.4f}s")

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = config['dataset']['root']

    print("Creating/Loading Training Dataset...")
    train_dataset = ImageGraphDataset(root=dataset_root, config=config, train=True, timing=True)
    print(f"Training dataset loaded: {train_dataset}")
    print(f"Example train data point: {train_dataset.get(0)}")
    print_label_distribution(train_dataset, "Training")
    print_node_statistics(train_dataset, "Training")
    print_edge_statistics(train_dataset, "Training")
    print_time_statistics(train_dataset, "Training")

    print("\nCreating/Loading Test Dataset...")
    test_dataset = ImageGraphDataset(root=dataset_root, config=config, train=False, timing=True)
    print(f"Test dataset loaded: {test_dataset}")
    print(f"Example test data point: {test_dataset.get(0)}")
    print_label_distribution(test_dataset, "Test")
    print_node_statistics(test_dataset, "Test")
    print_edge_statistics(test_dataset, "Test")
    print_time_statistics(test_dataset, "Test")

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total number of samples: {len(train_dataset) + len(test_dataset)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    if config['dataset']['name'] == 'Custom':
        train_files = set([data.file_name for data in train_dataset])
        test_files = set([data.file_name for data in test_dataset])
        overlap = train_files.intersection(test_files)
        if len(overlap) > 0:
            print(f"Error: Found {len(overlap)} overlapping samples between train and test sets.")
            print("Overlapping samples:", overlap)
        else:
            print("Success: No overlap found between train and test sets.")
    
    loader_config = {
        'batch_size': 128,
        'shuffle_train': True,
        'num_workers': 4,
        'persistent_workers': False,
        'pin_memory': False,
        'classes': [0, 1, 2, 3, 4, 5, 6],
        # For semi-supervised setup
        'semi_supervised': {
            'enabled': True,
            'labeled_fraction': 0.05, # Fraction of the *training* set to be labeled
            'normal_label': 1,
            'anomaly_label': -1,
            'unlabeled_label': 0
        }
            
    }        
    train_loader, _ = create_semi_supervised_dataloaders(train_dataset, test_dataset, loader_config, normal_class=5)
    print_dataloader_label_distribution(train_loader, "Train Loader")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display information about image graph datasets.")
    parser.add_argument('config_path', type=str, help="Path to the configuration YAML file (e.g., 'configs/ham10000.yaml'")
    
    args = parser.parse_args()
    
    main(args.config_path)