import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float

def segment_image(image, config, mask=None):
    """Segments an image based on the configuration."""
    method = config.get('segmentation', {}).get('method', 'patch')
    if method == 'slic':
        return segment_slic(image, config, mask)
    elif method == 'slico':
        return segment_slic(image, config, mask, slic_zero=True)
    elif method == 'patch':
        return segment_patches(image, config)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def segment_slic(image, config, mask=None, slic_zero=False):
    """Performs SLIC superpixel segmentation."""
    # Ensure image is float [0, 1] for SLIC
    image_float = img_as_float(image)
    
    # SLIC works best on color images, convert if grayscale
    if image_float.ndim == 2:
        image_float = np.stack([image_float] * 3, axis=-1)
    elif image_float.shape[2] == 1:
         image_float = np.concatenate([image_float] * 3, axis=-1)
    
    n_segments = config['segmentation']['params'].get('n_segments', 75)
    enforce_connectivity = config['segmentation']['params'].get('enforce_connectivity', True)
    if slic_zero:
        segments = slic(
            image_float,
            n_segments=n_segments,
            start_label=1,
            enforce_connectivity=enforce_connectivity,
            mask=mask,
            slic_zero=True
        )
    else:
        compactness = config['segmentation']['params'].get('compactness', 10)
        sigma = config['segmentation']['params'].get('sigma', 1)

        segments = slic(
            image_float,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=1, # Start labels from 1 to avoid confusion with potential background 0
            enforce_connectivity=enforce_connectivity,
            mask=mask
        )
    return segments

def segment_patches(image, config):
    """Divides the image into non-overlapping patches."""
    patch_size = config['segmentation']['params'].get('patch_size', 4)
    h, w = image.shape[:2]
    rows = h // patch_size
    cols = w // patch_size
    segments = np.zeros((h, w), dtype=int)
    segment_id = 1
    for r in range(rows):
        for c in range(cols):
            r_start, r_end = r * patch_size, (r + 1) * patch_size
            c_start, c_end = c * patch_size, (c + 1) * patch_size
            segments[r_start:r_end, c_start:c_end] = segment_id
            segment_id += 1

    if h % patch_size != 0:
        segments[rows * patch_size:, :] = segments[rows * patch_size - 1:rows * patch_size, :].repeat(h % patch_size, axis=0)
    if w % patch_size != 0:
        segments[:, cols * patch_size:] = segments[:, cols * patch_size - 1:cols * patch_size].repeat(w % patch_size, axis=1)

    return segments