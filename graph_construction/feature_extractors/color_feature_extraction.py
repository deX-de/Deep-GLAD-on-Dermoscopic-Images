import numpy as np
from cached_property import cached_property
import cv2
from scipy.stats import skew

class ColorFeatureExtraction(object):
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def get_features(self, color_spaces=None):
        if color_spaces is None:
            color_spaces = ['rgb']

        features = []
        for space in color_spaces:
            features.extend(self._compute_color_features(space))
        return np.array(features)

    @cached_property
    def _masked_pixels(self):
        return self.image[self.mask]

    def _compute_statistics(self, channel_data):
        if len(channel_data) == 0:
            return np.zeros(5)
        if channel_data.max() > 1:
            channel_data = channel_data / 255.0
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std < 1e-8:
            skewness = 0.0
        else:
            skewness = skew(channel_data)
        return [mean, std, skewness]

    def _compute_color_features(self, color_space):
        if color_space.lower() == 'rgb':
            pixels = self._masked_pixels
        else:
            if color_space.lower() in ['lab', 'luv']:
                conversion_code = getattr(cv2, f'COLOR_RGB2{color_space.capitalize()}')    
            else:
                conversion_code = getattr(cv2, f'COLOR_RGB2{color_space.upper()}')
            converted_image = cv2.cvtColor(self.image, conversion_code)
            pixels = converted_image[self.mask]
        
        features = []
        for channel in range(pixels.shape[1]):
            stats = self._compute_statistics(pixels[:, channel])
            features.extend(stats)
        
        return features