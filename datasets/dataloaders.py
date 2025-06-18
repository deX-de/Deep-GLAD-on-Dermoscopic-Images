import torch
import numpy as np
from torch_geometric.loader import DataLoader

def create_unsupervised_dataloaders(dataset_train, dataset_test, loader_config, normal_class=0):
    """Creates DataLoaders for unsupervised learning."""

    # Train loader: only normal class samples
    train_data = []
    idx = 0
    for data in dataset_train:
        if data.y.item() == normal_class:
            train_data.append(data)
            data.idx = idx
            idx += 1
    train_loader = DataLoader(
        train_data,
        batch_size=loader_config['batch_size'],
        shuffle=loader_config['shuffle_train'],
        num_workers=loader_config['num_workers'],
        pin_memory=loader_config.get('pin_memory', False),
        persistent_workers=loader_config.get('persistent_workers', False),
    )

    # Test loader: normal class (label 0) and all others (label 1)
    test_data = []
    for data in dataset_test:
        new_data = data.clone()
        new_data.y = torch.tensor([0 if data.y.item() == normal_class else 1], dtype=torch.long)
        test_data.append(new_data)

    test_loader = DataLoader(
        test_data,
        batch_size=loader_config['batch_size'],
        shuffle=False,
        num_workers=loader_config['num_workers'],
        pin_memory=loader_config.get('pin_memory', False),
        persistent_workers=loader_config.get('persistent_workers', False),
    )

    return train_loader, test_loader

def create_semi_supervised_dataloaders(dataset_train, dataset_test, loader_config, normal_class=0):
    """Creates DataLoaders for semi-supervised learning."""
    semi_sup_config = loader_config.get('semi_supervised', {})

    if not semi_sup_config or not semi_sup_config.get('enabled', False):
        print("Semi-supervised mode not enabled in config. Returning standard loaders.")
        return create_unsupervised_dataloaders(dataset_train, dataset_test, loader_config)

    labeled_fraction = semi_sup_config.get('labeled_fraction', 0.0)
    normal_label = semi_sup_config.get('normal_label', 1)
    anomaly_label = semi_sup_config.get('anomaly_label', -1)
    unlabeled_label = semi_sup_config.get('unlabeled_label', 0)
    
    classes = loader_config.get('classes', list(range(10))).copy()
    classes.remove(normal_class)
    known_anomaly_classes = classes
    
    # Prepare training data
    train_data = []
    normal_indices = []
    anomaly_indices = []

    for i, data in enumerate(dataset_train):
        if data.y.item() == normal_class:
            normal_indices.append(i)
        elif data.y.item() in known_anomaly_classes:
            anomaly_indices.append(i)

    N, P = len(normal_indices), len(anomaly_indices)
    target_labeled_samples = int((N * labeled_fraction) / (1 - labeled_fraction))
    n_labeled_anomaly = min(P, target_labeled_samples) # Take as many labeled anomalies as possible
    
    # Maintain ratio while filling the rest with labeled normal data: 
    #     labeled_fraction = labeled_samples / total_samples 
    #                      = (P + labeled_normal) / (P + N) 
    # <=> labeled_normal   = labeled_fraction * (N + P) - P
    # with 0 <= labeled_normal <= N
    n_labeled_normal = max(0, int(min(labeled_fraction * (N + P) - P, N)))

    labeled_normal_idx = np.random.choice(normal_indices, n_labeled_normal, replace=False)
    labeled_anomaly_idx = np.random.choice(anomaly_indices, n_labeled_anomaly, replace=False)
    idx = 0
    for i, data in enumerate(dataset_train):
        new_data = data.clone()
        if i in labeled_normal_idx:
            new_data.y = torch.tensor([normal_label], dtype=torch.long)
        elif i in labeled_anomaly_idx:
            new_data.y = torch.tensor([anomaly_label], dtype=torch.long)
        elif data.y.item() == normal_class:
            new_data.y = torch.tensor([unlabeled_label], dtype=torch.long)
        else:
            continue 
        train_data.append(new_data)
        new_data.idx = idx
        idx += 1
    train_loader = DataLoader(
        train_data,
        batch_size=loader_config['batch_size'],
        shuffle=loader_config['shuffle_train'],
        num_workers=loader_config['num_workers'],
        pin_memory=loader_config.get('pin_memory', False),
        persistent_workers=loader_config.get('persistent_workers', False),
    )

    # Prepare test data
    test_data = []
    for data in dataset_test:
        new_data = data.clone()
        new_data.y = torch.tensor([normal_label if data.y.item() == normal_class else anomaly_label], dtype=torch.long)
        test_data.append(new_data)

    test_loader = DataLoader(
        test_data,
        batch_size=loader_config['batch_size'],
        shuffle=False,
        num_workers=loader_config['num_workers'],
        pin_memory=loader_config.get('pin_memory', False),
        persistent_workers=loader_config.get('persistent_workers', False),
    )

    return train_loader, test_loader