import numpy as np 
import cv2
from skimage.feature import local_binary_pattern
from .feature_extractors import ColorFeatureExtraction, TextureFeatureExtraction, ShapeFeatureExtraction
import time

def calculate_node_features(image, segments, config, segmentation_mask=None):
    features_config = config.get('features', {})
    unique_segments = np.unique(segments)
    is_grayscale = image.ndim == 2 or image.shape[2] == 1
    
    # If masks are used for segmentation, the background is set to label 0
    if unique_segments[0] == 0:
        if segmentation_mask is None:
            print(f"Warning: Segment ID 0 found with {config.get('segmentation', {}).get('method', 'Unknown segmentation')}, check segmentation.")
        unique_segments = unique_segments[1:]

    if segmentation_mask is not None:
        tmp = np.zeros_like(image)
        tmp[segmentation_mask == 255] = image[segmentation_mask == 255]
        image = tmp
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    image_rgb = image if image.ndim == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    lbp_image = local_binary_pattern(image_gray, 8, 1, method='uniform') if features_config.get('texture', {}).get('enabled', False) else None

    all_masks = [(segments == seg_id) for seg_id in unique_segments]
    
    start_time = time.time()
    node_pos = [np.column_stack(np.where(mask)).mean(axis=0)[::-1] for mask in all_masks]  # (x, y)
    end_time = time.time()

    feature_dict = {'pos': np.array(node_pos, dtype=np.float32)}
    feature_time = {'pos': end_time - start_time}
    extractors = {
        'color': lambda mask: ColorFeatureExtraction(image_rgb, mask).get_features(features_config.get('color', {}).get('color_spaces', ['rgb'])),
        'texture': lambda mask: TextureFeatureExtraction(image_gray, mask, lbp_image).get_features(),
        'shape': lambda mask: ShapeFeatureExtraction(mask.astype(np.uint8)).get_features()
    }
    # Patches have the same shape => Shape descriptors are not informative
    features_config['shape']['enabled'] = config['segmentation']['method'] != 'patch'
    
    for feature_type in ['color', 'texture', 'shape']:
        if features_config.get(feature_type, {}).get('enabled', False):
            
            start_time = time.time()
            feature = [extractors[feature_type](mask) for mask in all_masks]
            end_time = time.time()
            
            feature_dict[feature_type] = np.array(feature, dtype=np.float32)
            feature_time[feature_type] = end_time - start_time
    
    start_time = time.time()
    mean_colors = []
    for mask in all_masks:
        segment_pixels = image[mask]
        if segment_pixels.max() > 1:
            segment_pixels = segment_pixels / 255.0
        mean_color = np.mean(segment_pixels, axis=0)
        if is_grayscale:
            mean_colors.append([mean_color])
        else:
            mean_colors.append(mean_color)

    end_time = time.time()
    feature_dict['x'] = np.array(mean_colors, dtype=np.float32)
    feature_time['mean_rgb'] = end_time - start_time

    if features_config.get('virtual_nodes', {}).get('background', False) or features_config.get('virtual_nodes', {}).get('global', False):
        feature_dict, unique_segments = _add_virtual_node_features(image, segments, all_masks, feature_dict, unique_segments, config, segmentation_mask)

    return feature_dict, unique_segments, feature_time

def _add_virtual_node_features(image, segments, all_masks, feature_dict, unique_segments, config, segmentation_mask=None):
    features_config = config.get('features', {})
    h, w = image.shape[:2]
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    image_rgb = image if image.ndim == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    lbp_image = local_binary_pattern(image_gray, 8, 1, method='uniform') if features_config.get('texture', {}).get('enabled', False) else None

    new_segment_ids = []
    new_nodes_pos = []
    new_features = {f: [] for f in ['color', 'texture', 'shape'] if features_config.get(f, {}).get('enabled', False)}
    new_mean_colors = []

    virtual_nodes = [
        ('background', ~segmentation_mask.astype(bool) if segmentation_mask is not None else None),
        ('global', np.ones_like(segments, dtype=bool) if features_config.get('virtual_nodes', {}).get('global', False) else None)
    ]
    current_max_id = np.max(unique_segments)
    is_grayscale = image.ndim == 2 or image.shape[2] == 1
    for node_type, mask in virtual_nodes:
        if mask is not None:
            current_max_id += 1
            new_segment_ids.append(current_max_id)
            new_nodes_pos.append([w/2, h/2])

            segment_pixels = image[mask]
            if segment_pixels.size > 0:
                if segment_pixels.max() > 1:
                    segment_pixels = segment_pixels / 255.0
                mean_color = np.mean(segment_pixels, axis=0)
            elif is_grayscale:
                mean_color = np.array(0, dtype=np.float32)
            else:
                mean_color = np.zeros(3, dtype=np.float32)
            if is_grayscale:
                new_mean_colors.append([mean_color])
            else:
                new_mean_colors.append(mean_color)
            
            for feature_type in new_features:
                try:
                    if feature_type == 'color':
                        extractor = ColorFeatureExtraction(image_rgb, mask)
                        features = extractor.get_features(features_config.get('color', {}).get('color_spaces', ['rgb']))
                    elif feature_type == 'texture':
                        extractor = TextureFeatureExtraction(image_gray, mask, lbp_image)
                        features = extractor.get_features()
                    elif feature_type == 'shape':  
                        extractor = ShapeFeatureExtraction(mask.astype(np.uint8))
                        features = extractor.get_features()

                    expected_dim = feature_dict[feature_type].shape[1]
                    features = np.pad(features, (0, max(0, expected_dim - len(features))), 'constant')[:expected_dim]
                    new_features[feature_type].append(features)
                except Exception as e:
                    print(f"Error extracting {feature_type} features for {node_type} node: {e}")
                    new_features[feature_type].append(np.zeros(feature_dict[feature_type].shape[1], dtype=np.float32))

    if new_nodes_pos:
        feature_dict['pos'] = np.vstack([feature_dict['pos'], new_nodes_pos])
        for feature_type, features in new_features.items():
            feature_dict[feature_type] = np.vstack([feature_dict[feature_type], features])
        
        # new_x = np.hstack([new_features[f] for f in ['color', 'texture', 'shape'] if f in new_features])
        # feature_dict['x'] = np.vstack([feature_dict['x'], new_x])
        feature_dict['x'] = np.vstack([feature_dict['x'], new_mean_colors])
        
        unique_segments = np.concatenate([unique_segments, new_segment_ids])

    return feature_dict, unique_segments