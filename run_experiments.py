import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
torch.backends.cudnn.benchmark = True

import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import yaml
import copy
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import traceback
from tqdm import tqdm
from itertools import product
from datasets import ImageGraphDataset, create_unsupervised_dataloaders, create_semi_supervised_dataloaders
from models import OCGTL, SIGNET, CVTGAD
from trainers import OCGTL_Trainer, SIGNET_Trainer, CVTGAD_Trainer
from utils import OCGTL_loss, Semi_OCGTL_loss, Logger, init_structural_encoding

DEFAULT_MODE = "unsupervised"
BASE_CONFIG_PATH = 'configs/experiments/ham10000.yaml'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _modify_config(graph_construction_config, segmentation_config, dataset_name, use_mask, use_virtual_nodes, segmentation_method):
    config = copy.deepcopy(graph_construction_config)

    use_virtual_nodes = use_virtual_nodes and segmentation_method != 'patch'
    config['dataset']['use_mask'] = use_mask
    config['segmentation']['method'] = segmentation_method
    config['features']['virtual_nodes']['background'] = use_virtual_nodes
    config['features']['virtual_nodes']['global'] = use_virtual_nodes
    
    if segmentation_method in segmentation_config and dataset_name in segmentation_config[segmentation_method]:
        params = segmentation_config[segmentation_method][dataset_name]
        for param_name, param_value in params.items():
            config['segmentation']['params'][param_name] = param_value
    return config

def _fit_pca_scaler(data_list, feature_selection, variance_ratio=0.95):
    scaler = StandardScaler()
    pca = PCA(n_components=variance_ratio)
    collected_features = []

    for data in data_list:
        if feature_selection == 'mean_rgb':
            features = data.x
        elif feature_selection == 'all':
            features = [getattr(data, f, None) for f in ['color', 'texture', 'shape']]
            if any(feat is None for feat in features):
                continue 
            features = torch.cat(features, dim=-1).float()
        elif feature_selection == 'pos':
            features = torch.cat([data.x, data.pos], dim=-1).float()
        else:
            features = getattr(data, feature_selection, None)
        
        if features is not None:
            collected_features.append(features.cpu().numpy())

    # Count number of features
    number_features = collected_features[0].shape[1] if collected_features else 0

    all_features_np = np.concatenate(collected_features, axis=0)
    all_features_np = np.nan_to_num(all_features_np)
    scaled_features = scaler.fit_transform(all_features_np)
    pca.fit(scaled_features)

    # Count new number of features after PCA
    new_number_features = pca.n_components_

    return pca, scaler, number_features, new_number_features

def _apply_pca_scaler_transform(data, pca, scaler):
    if pca is None or scaler is None:
        return data.x
    features_np = np.nan_to_num(data.x.cpu().numpy())
    scaled_features = scaler.transform(features_np)
    pca_features = pca.transform(scaled_features)
    
    return torch.from_numpy(pca_features).to(dtype=torch.float32, device=data.x.device)


def _select_runtime_attributes(data_list, feature_selection, edge_selection, pca=None, scaler=None):
    processed_list = []
    for data in data_list:
        if data is None:
            continue
        new_data = data.clone()
        
        # Only store necessary features (for better memory efficiency)
        for f in ['color', 'texture', 'shape']:
            setattr(new_data, f, None)
        for e in ['rag', 'knn_spatial', 'knn_spatial_color']:
            setattr(new_data, f"{e}_edge_index", None)
            setattr(new_data, f"{e}_edge_attr", None)
            
        if feature_selection == 'mean_rgb':
            new_data.x = data.x  # Already contains mean RGB values
        elif feature_selection == 'all':
            if getattr(data, 'shape', None) is None:
                features = [getattr(data, f, None) for f in ['color', 'texture']]
            else:
                features = [getattr(data, f, None) for f in ['color', 'texture', 'shape']]
            if any(feat is None for feat in features):
                continue 
            new_data.x = torch.cat(features, dim=-1).float()                
        elif feature_selection == 'pos':
            new_data.x = torch.cat([data.x, data.pos], dim=-1).float()
        else:
            features = getattr(data, feature_selection, None)
            if features is None:
                continue
            new_data.x = features
        if pca is not None and scaler is not None:
            new_data.x = _apply_pca_scaler_transform(new_data, pca, scaler)
    
        edge_index_key = f"{edge_selection}_edge_index"
        edge_attr_key = f"{edge_selection}_edge_attr"
        new_data.edge_index = getattr(data, edge_index_key, torch.empty((2, 0), dtype=torch.long, device=new_data.x.device))
        new_data.edge_attr = getattr(data, edge_attr_key, None)
        processed_list.append(new_data)
    return processed_list

def _get_model_trainer(model_type, model_config, in_channels, semi):
    model, trainer = None, None
    config = model_config.get(model_type)
    if model_type == 'OCGTL':
        model = OCGTL(dim_features=in_channels, config=config).to(DEVICE)
        if semi:
            loss_fn = Semi_OCGTL_loss().to(DEVICE)
        else:
            loss_fn = OCGTL_loss().to(DEVICE)
        trainer = OCGTL_Trainer(model, loss_fn, device=DEVICE, semi=semi)
    elif model_type == 'SIGNET':
        model = SIGNET(input_dim=in_channels, input_dim_edge=None, config=config).to(DEVICE)
        trainer = SIGNET_Trainer(model, device=DEVICE, semi=semi)
    elif model_type == 'CVTGAD':
        model = CVTGAD(feat_dim=in_channels, config=config).to(DEVICE)
        trainer = CVTGAD_Trainer(model, device=DEVICE, semi=semi)
    return model, trainer

def run_experiment_unsupervised(train_loader, test_loader, model_type, model_config, param_grid, logger=None):
    start_time = time.time()
    sample_data = next(iter(train_loader))
    in_channels = sample_data.num_node_features
    model, trainer = _get_model_trainer(model_type, model_config, in_channels, semi=False)

    optimizer = optim.Adam(model.parameters(), lr=param_grid.get('training', {}).get('lr', 1e-3), weight_decay=param_grid.get('training', {}).get('weight_decay', 0))
    epochs = param_grid.get('epochs', 20)
    final_roc, final_pr = trainer.run_training(train_loader, optimizer, epochs, validation_loader=test_loader,
                     test_loader=test_loader, normal_class=0, scheduler=None, early_stopper=None,
                     logger=logger, log_every=1)
    end_time = time.time()
    time_expended = end_time - start_time
    return final_roc, final_pr, time_expended

def run_experiment_semi_supervised(train_loader, test_loader, model_type, model_config, param_grid, logger=None):
    start_time = time.time()
    sample_data = next(iter(train_loader))
    in_channels = sample_data.num_node_features
    model, trainer = _get_model_trainer(model_type, model_config, in_channels, semi=True)
    optimizer = optim.Adam(model.parameters(), lr=param_grid.get('training', {}).get('lr', 1e-3), weight_decay=param_grid.get('training', {}).get('weight_decay', 0))
    epochs = param_grid.get('epochs', 20)
    final_roc, final_pr = trainer.run_training(train_loader, optimizer, epochs, validation_loader=test_loader,
                     test_loader=test_loader, normal_class=1, scheduler=None, early_stopper=None,
                     logger=logger, log_every=1)
    
    end_time = time.time()
    time_expended = end_time - start_time
    return final_roc, final_pr, time_expended

DATALOADER_FN = {
    'unsupervised': create_unsupervised_dataloaders,
    'semi_supervised': create_semi_supervised_dataloaders
}
RUN_EXPERIMENT_FN = {
    'unsupervised': run_experiment_unsupervised,
    'semi_supervised': run_experiment_semi_supervised
}

from sklearn.model_selection import StratifiedKFold #, KFold

def main(mode=DEFAULT_MODE, config=BASE_CONFIG_PATH, override_model=None):
    with open(config, 'r') as f:
        base_config = yaml.safe_load(f)
        
    param_grid = base_config['param_grid'][mode]
    # If user passed a specific model_type via CLI, replace list in param_grid
    if override_model is not None:
        param_grid['model_type'] = [override_model]

    run_experiment = RUN_EXPERIMENT_FN[mode]
    dataloader_fn = DATALOADER_FN[mode]

    # Build a suffix for filenames that includes mode, seed(s), and model_type(s)
    seed_suffix = str(param_grid['seed'][0]) if len(param_grid['seed']) == 1 else "allseeds"
    model_suffix = str(param_grid['model_type'][0]) if len(param_grid['model_type']) == 1 else "allmodels"
    run_prefix = f"{mode}_seed={seed_suffix}_model={model_suffix}"

    # Insert the run_suffix before the original filenames
    orig_results = param_grid['results_file']
    orig_summary = param_grid['summary_file']
    results_filename = os.path.join(RESULTS_DIR, f"{run_prefix}_{orig_results}")
    summary_filename = os.path.join(RESULTS_DIR, f"{run_prefix}_{orig_summary}")

    normal_classes = param_grid['normal_classes']

    graph_construction_config = base_config['graph_construction']
    segmentation_config = base_config['segmentation_config']
    model_config = base_config['model_config']
    loader_config = base_config['dataloader_config']
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = None
    if param_grid.get('log_file', None) is not None:
        orig_log = param_grid['log_file']
        log_filename = os.path.join(LOGS_DIR, f"{run_prefix}_{orig_log}")
        logger = Logger(log_filename, 'a')

    # Load existing results if file exists
    if os.path.exists(results_filename):
        existing_results = pd.read_csv(results_filename)
    else:
        existing_results = pd.DataFrame(columns=[
            'dataset', 'model_type', 'segmentation', 'use_mask', 'use_virtual_nodes', 
            'features', 'edges', 'normal_class', 'cv_fold', 'ROC_AUC', 'PR_AUC', 'Time', 'Error'])
    
    # Sort existing csv file to correct order
    dataset_order = {'MNIST': 0, 'FashionMNIST': 1, 'CIFAR10': 2, 'HAM10000': 3}
    feature_order = {'mean_rgb': 0, 'color': 1, 'texture': 2, 'shape': 3, 'all': 4}
    existing_results = existing_results.sort_values(
        by=['dataset', 'segmentation', 'use_mask', 'use_virtual_nodes', 'model_type', 'features', 'edges', 'cv_fold', 'normal_class'],
        key=lambda x: x.map(dataset_order) if x.name == 'dataset' 
        else x.map(feature_order) if x.name == 'features' 
        else x,
        ascending=[True, True, False, False, True, True, True, True, True]
    )
    existing_results.to_csv(results_filename, index=False)

    # Calculate total experiments
    total_experiments = (sum(len(normal_classes[dataset_name]) for dataset_name in param_grid['dataset']) *
        sum((len(param_grid['features']) - 1) if 'shape' in param_grid['features'] and seg_method == 'patch' # do not count patch + shape combination (as all patches have the same shape)
            else len(param_grid['features']) for seg_method in param_grid['segmentation_method']) *
        len(param_grid['edges']) *
        len(param_grid['seed']) *
        len(param_grid['mask']) * len(param_grid['virtual_nodes']) * 5)

    # Fix number of experiments because we skip (False, True) combinations of mask and virtual nodes
    if len(param_grid['mask']) * len(param_grid['virtual_nodes']) == 4:
        total_experiments = int(total_experiments * 3/4)
    elif len(param_grid['mask']) == 2  and param_grid['virtual_nodes'] == [True]:
        total_experiments = int(total_experiments * 1/2)
    
    # Main progress bar
    pbar = tqdm(total=total_experiments, desc="Experiment Progress", smoothing=1)
    for dataset_name, segmentation_method, use_mask, use_virtual_nodes in product(param_grid['dataset'], param_grid['segmentation_method'], param_grid['mask'], param_grid['virtual_nodes']):
        if not use_mask and use_virtual_nodes:
            continue
        current_config = _modify_config(graph_construction_config, segmentation_config, dataset_name, use_mask, use_virtual_nodes, segmentation_method)
        current_config['epochs'] = param_grid['epochs']
        try:
            full_dataset = ImageGraphDataset(root=current_config['dataset']['root'], config=current_config, train=True, timing=True)
            loader_config['classes'] = set(full_dataset.y.tolist())

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            dataset_indices = list(range(len(full_dataset)))
            cv_splits = list(kfold.split(dataset_indices, full_dataset.y))

            for cv_fold, (train_indices, test_indices) in enumerate(cv_splits):
                train_dataset = Subset(full_dataset, train_indices)
                test_dataset = Subset(full_dataset, test_indices)
                for feature_selection in param_grid['features']:
                    # All patches have the same patch_size x patch_size shape
                    if segmentation_method == 'patch' and feature_selection == 'shape':
                        continue
                    for model_type in param_grid['model_type']:
                        for edge_selection, normal_class in product(
                            param_grid['edges'], normal_classes[dataset_name]):

                            # Check if experiment already exists
                            existing_exp = existing_results[
                                (existing_results['dataset'] == dataset_name) &
                                (existing_results['model_type'] == model_type) &
                                (existing_results['segmentation'] == segmentation_method) &
                                (existing_results['use_mask'] == use_mask) &
                                (existing_results['use_virtual_nodes'] == use_virtual_nodes) &
                                (existing_results['features'] == feature_selection) &
                                (existing_results['edges'] == edge_selection) &
                                (existing_results['normal_class'] == normal_class) &
                                (existing_results['cv_fold'] == cv_fold)
                            ]

                            if not existing_exp.empty:
                                pbar.update(1)
                                continue

                            # Use PCA for features other than mean rgb value
                            pca, scaler, number_features, number_features_pca = None, None, 3, 3
                            if feature_selection != 'mean_rgb':
                                pca, scaler, number_features, number_features_pca = _fit_pca_scaler(train_dataset, feature_selection)

                            selected_train = _select_runtime_attributes(train_dataset,
                                                                    feature_selection, edge_selection, pca, scaler)
                            selected_test = _select_runtime_attributes(test_dataset,
                                                                    feature_selection, edge_selection, pca, scaler)

                            if model_type == 'CVTGAD':
                                cvtgad_config = model_config.get('CVTGAD', {})
                                rw_dim = cvtgad_config.get('rw_dim', 16)
                                dg_dim = cvtgad_config.get('dg_dim', 16)
                                selected_train = init_structural_encoding(selected_train, rw_dim, dg_dim)
                                selected_test = init_structural_encoding(selected_test, rw_dim, dg_dim)

                            # unsupervised: train_loader labels: [normal_class], test_loader labels: [0, 1]
                            # # semi-supervised: train_loader labels: [-1, 0, 1] (anomaly, unlabeled, normal), test_loader labels: [-1, 1] (anomaly, normal)
                            train_loader, test_loader = dataloader_fn(
                                selected_train, selected_test, loader_config, normal_class)

                            result = {
                                'dataset': dataset_name,
                                'model_type': model_type,
                                'segmentation': segmentation_method,
                                'use_mask': use_mask,
                                'use_virtual_nodes': use_virtual_nodes,
                                'features': feature_selection,
                                'edges': edge_selection,
                                'normal_class': normal_class,
                                'cv_fold': cv_fold,
                                'ROC_AUC': None,
                                'PR_AUC': None,
                                'Time': None,
                                'Error': None,
                                'number_features': number_features,
                                'number_features_pca': number_features_pca
                            }

                            try:
                                if logger:
                                    logger.log(f"\nDataset: {dataset_name}\nCVFold: {cv_fold}\nModel: {model_type}\nSegmentation: {segmentation_method}\nEdge Construction: {edge_selection}\nFeatures: {feature_selection}\nNormal Class: {normal_class}\n")
                                    logger.log("--------------------Training Start--------------------")
                                final_auc, final_pr, time_expended = run_experiment(
                                    train_loader, test_loader, model_type, model_config, param_grid, logger)
                                if logger:
                                    logger.log("--------------------Training End----------------------\n")

                                result['ROC_AUC'] = final_auc
                                result['PR_AUC'] = final_pr
                                result['Time'] = time_expended
                            except Exception as e:
                                print(f"Error in experiment: {e}")
                                result['ROC_AUC'] = 'Error'
                                result['PR_AUC'] = 'Error'
                                result['Time'] = 'Error'
                                result['Error'] = str(e)
                                traceback.print_exc()

                            # Append new result to existing results using concat
                            new_results_df = pd.DataFrame([result])
                            if existing_results.empty:
                                existing_results = new_results_df
                            else:
                                existing_results = pd.concat([existing_results, new_results_df], ignore_index=True)
                            existing_results.to_csv(results_filename, index=False)

                            pbar.update(1)

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            traceback.print_exc()

    pbar.close()  # Close the progress bar

    # Final aggregation
    agg_results = existing_results[
        (existing_results['ROC_AUC'].apply(lambda x: x != 'Error')) & 
        (existing_results['PR_AUC'].apply(lambda x: x != 'Error')) & 
        (existing_results['Time'].apply(lambda x: x != 'Error'))
    ].copy()

    agg_results['ROC_AUC'] = pd.to_numeric(agg_results['ROC_AUC'])
    agg_results['PR_AUC'] = pd.to_numeric(agg_results['PR_AUC'])
    agg_results['Time'] = pd.to_numeric(agg_results['Time'])
    agg_results['number_features'] = pd.to_numeric(agg_results['number_features'])
    agg_results['number_features_pca'] = pd.to_numeric(agg_results['number_features_pca'])

    summary = agg_results.groupby([
        'dataset', 'model_type', 'segmentation', 'use_mask', 'use_virtual_nodes', 
        'features', 'edges', 'normal_class'
    ]).agg({
        'ROC_AUC': ['mean', 'std', 'count'],
        'PR_AUC': ['mean', 'std'],
        'Time': ['mean', 'std'],
        'number_features': ['mean', 'std'],
        'number_features_pca': ['mean', 'std']
    }).reset_index()

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.columns = summary.columns.str.rstrip('_')
    summary = summary.rename(columns={'ROC_AUC_count': 'count'})

    column_order = [
        'dataset', 'model_type', 'segmentation', 'use_mask', 'use_virtual_nodes', 
        'features', 'edges', 'normal_class',
        'ROC_AUC_mean', 'ROC_AUC_std', 'PR_AUC_mean', 'PR_AUC_std', 'Time_mean', 'Time_std', 'count'
    ]
    summary = summary[column_order]
    summary = summary.sort_values(
        by=['dataset', 'segmentation', 'use_mask', 'use_virtual_nodes', 'model_type', 'features', 'edges', 'normal_class'],
        key=lambda x: x.map(dataset_order) if x.name == 'dataset' 
        else x.map(feature_order) if x.name == 'features' 
        else x,
        ascending=[True, True, False, False, True, True, True, True]
    )
    summary.to_csv(summary_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE, choices=['unsupervised', 'semi_supervised'])
    parser.add_argument('--configs', nargs='+', default=[BASE_CONFIG_PATH])
    parser.add_argument('--model_type', type=str, default=None, help="If provided, override the list of model types in the config and run only this one.")
    args = parser.parse_args()
    for config in args.configs:
        main(mode=args.mode,
            config=config,
            override_model=args.model_type)
