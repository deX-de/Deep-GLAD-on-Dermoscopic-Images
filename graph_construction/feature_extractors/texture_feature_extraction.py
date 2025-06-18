import numpy as np
from cached_property import cached_property
from skimage.feature import graycomatrix, graycoprops

class TextureFeatureExtraction(object):
    def __init__(self, image, mask, lbp):
        self.image = image
        self.mask = mask
        self.lbp = lbp
        
    def get_features(self):
        return np.concatenate([self.lbp_features, self.glcm_features])

    @cached_property
    def _masked_region(self):
        # Get the bounding box of the mask
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the region
        mask_region = self.mask[rmin:rmax+1, cmin:cmax+1]
        image_region = self.image[rmin:rmax+1, cmin:cmax+1]
        return image_region * mask_region

    @cached_property
    def lbp_features(self):
        lbp_masked = self.lbp[self.mask]
        n_bins = 10
        hist, _ = np.histogram(lbp_masked, bins=np.arange(0, n_bins))
        return hist

    @cached_property
    def glcm_features(self):
        if self._masked_region.size < 2:
            return np.zeros(20)  # 5 properties * 4 angles
            
        # Rescale to [0, 31]
        levels = 32
        scaled_region = np.round((self._masked_region / 255.0) * (levels - 1)).astype(np.uint8)
        
        # Calculate GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(scaled_region, distances, angles, 
                           levels=levels, symmetric=True, normed=True)
        
        # Extract properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        features = []
        for prop in properties:
            features.extend(graycoprops(glcm, prop).ravel())
        return np.array(features)