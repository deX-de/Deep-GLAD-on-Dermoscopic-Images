import os
import yaml
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch_geometric.data import InMemoryDataset, Data
from graph_construction import segment_image, calculate_node_features, construct_edges
from multiprocessing import Pool, cpu_count
import random
import time

class ImageGraphDataset(InMemoryDataset):
    def __init__(self, root, config, train=True, transform=None, pre_transform=None, pre_filter=None, force_reload=False, timing=False):
        self.config = config
        self.train = train
        self.timing = timing
        self._processed_dir_name = self._get_processed_dir_name()
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        if self.timing and os.path.exists(self.times_file_name):
            with open(self.times_file_name, 'r') as f:
                self.times = yaml.safe_load(f)
        self.load(self.processed_paths[0])

    def _get_processed_dir_name(self):
        edge_abbr = {
            'rag': 'r',
            'knn_spatial': 'ks',
            'knn_spatial_color': 'ksc'
        }
        config = self.config
        seg_method = config['segmentation']['method']
        feat_str = "".join(k[0] for k, v in config['features'].items() if isinstance(v, dict) and v.get('enabled'))
        edge_str = "".join(edge_abbr.get(k, k[0]) for k, v in config['edges'].items() if isinstance(v, dict) and v.get('enabled'))
        virt_str = "".join(k[0] for k, v in config['features']['virtual_nodes'].items() if v)
        mask_str = 'masked' if config['dataset'].get('use_mask', False) else ''
        full_str = 'full' if self.config['dataset'].get('train_split', 0.8) == 1.0 else ""
        return f"processed_{seg_method}_{feat_str}_{edge_str}_{virt_str}_{config['primary_edge_type']}_{mask_str}_{full_str}"

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.config['dataset']['path']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.config['dataset']['name'], 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.config['dataset']['name'], self._processed_dir_name)
    
    @property
    def times_file_name(self):
        return os.path.join(self.processed_dir, f"{'train' if self.train else 'test'}_times.yaml")

    @property
    def processed_file_names(self):
        if self.config['dataset'].get('train_split', 0.8) == 1.0 and self.train:
            prefix = 'full'
        elif self.train:
            prefix = 'train'
        else:
            prefix = 'test'
        return [f"{prefix}_data.pt"]

    def download(self):
        print(f"{self.config['dataset']['name']} dataset assumed to be at: {self.config['dataset']['path']}")

    def process(self):
        images, labels, f_names, masks = self._load_raw_data()
        
        # Convert string labels to numeric
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [self.label_encoder[label] for label in labels]
        
        indices = range(len(images))
        num_workers = self.config['dataset'].get('num_workers', 1)
        num_workers = min(num_workers, cpu_count())
        
        if self.timing:
            self.times = {}
            start_time = time.time()
        
        if num_workers > 1:
            with Pool(num_workers) as pool:
                data_list = list(tqdm(pool.imap(self._process_single_image, 
                                                [(idx, images[idx], numeric_labels[idx], f_names[idx] if f_names else None,
                                                  masks[idx] if masks else None) for idx in indices]), 
                                      total=len(indices)))
        else:
            data_list = [self._process_single_image((idx, images[idx], numeric_labels[idx], f_names[idx] if f_names else None,
                                                     masks[idx] if masks else None)) 
                         for idx in tqdm(indices)]
        
        data_list = [data for data in data_list if data is not None]
        if self.timing:
            end_time = time.time()
            self.times['total'] = end_time - start_time
            with open(self.times_file_name, 'w') as f:
                yaml.dump(self.times, f)
        self.save(data_list, self.processed_paths[0])

    def _process_single_image(self, args):
        idx, image, label, f_name, mask = args
        
        if self.timing:
            start_time = time.time()
            segments = segment_image(image, self.config, mask)
            end_time = time.time()
            self.times['segments'] = self.times.get('segments', 0) + (end_time - start_time)
            
            start_time = time.time()
            feature_dict, segment_map, feature_times = calculate_node_features(image, segments, self.config, mask)
            end_time = time.time()    
            self.times['features'] = self.times.get('features', 0) + (end_time - start_time)
            self.times['individual_features'] = {k: self.times.get('individual_features', {}).get(k, 0) + feature_times.get(k, 0) for k in set(feature_times)}
            
            start_time = time.time()
            edge_indices, edge_attrs, edge_times = construct_edges(segments, feature_dict['x'], feature_dict['pos'], segment_map, self.config)
            end_time = time.time() 
            self.times['edges'] = self.times.get('edges', 0) + (end_time - start_time)
            self.times['individual_edges'] = {k: self.times.get('individual_edges', {}).get(k, 0) + edge_times.get(k, 0) for k in set(edge_times)}
        else:
            segments = segment_image(image, self.config, mask)
            feature_dict, segment_map, _ = calculate_node_features(image, segments, self.config, mask)
            edge_indices, edge_attrs, _ = construct_edges(segments, feature_dict['x'], feature_dict['pos'], segment_map, self.config)
        
        data = Data(y=torch.tensor([label], dtype=torch.long))
        if f_name is not None:
            setattr(data, 'file_name', f_name)
        for key, value in feature_dict.items():
            setattr(data, key, torch.from_numpy(value).float())
        for key, value in edge_indices.items():
            setattr(data, f"{key}_edge_index", value)
        for key, value in edge_attrs.items():
            setattr(data, f"{key}_edge_attr", value)
        
        primary_edge_type = self.config['primary_edge_type']
        if primary_edge_type in edge_indices:
            data.edge_index = edge_indices[primary_edge_type]
            data.edge_attr = edge_attrs.get(primary_edge_type, None)
        else:
            print(f"Warning: Primary edge type '{primary_edge_type}' not found for sample {idx}.")
            data.edge_index = data.edge_attr = None
        
        if self.pre_filter is not None and not self.pre_filter(data):
            return None
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    def _load_raw_data(self):
        train_split = self.config['dataset']['train_split']
        if train_split < 0 or train_split > 1:
            raise ValueError(f'Training split ratio must be between 0 and 1')
        img_dir = self.config['dataset']['path']
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        data = []
        
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                mask = None
                if self.config['dataset'].get('use_mask', False):
                    mask_dir = self.config['dataset']['mask_dir']
                    mask_suffix = self.config['dataset'].get('mask_suffix', '')
                    mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + mask_suffix + '.png')
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                        # Ignore images, where mask is just a white image (taking the whole image)
                        if np.all(mask == 255):
                            print(f"{fname} has only 255s in the mask - Skipping.")
                            continue
                data.append((fname, img_rgb, mask))
        
        # Shuffle the data deterministically to avoid test data contamination
        seed = 42
        rng = random.Random(seed)
        rng.shuffle(data)
        
        # Calculate split points
        num_samples = len(data)
        num_train = int(num_samples * self.config['dataset']['train_split'])
        
        # Only keep the samples for the requested split
        if self.train:
            split_data = data[:num_train]
        else:
            split_data = data[num_train:]
        
        # Separate the shuffled data
        img_files, images, masks = zip(*split_data)
        
        if self.config['dataset'].get('label_source') == 'csv':
            import pandas as pd
            label_df = pd.read_csv(self.config['dataset']['label_file'])
            image_column = self.config['dataset'].get('image_column', 'filename')
            label_column = self.config['dataset'].get('label_column', 'label')
            labels = [label_df.loc[label_df[image_column] == os.path.splitext(fname)[0], label_column].item() for fname in img_files]
        else:  # Assume labels are from parent folders
            labels = [os.path.basename(os.path.dirname(os.path.join(img_dir, fname))) for fname in img_files]
        
        return images, labels, img_files, masks if any(mask is not None for mask in masks) else None
