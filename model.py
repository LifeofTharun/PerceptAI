import json
import numpy as np
from PIL import Image

def extract_image_features(image_path):
    """
    Extracts a normalized 8-dimensional feature vector from an image:
    1. Brightness: Mean grayscale intensity.
    2. Contrast: Standard deviation of grayscale intensity.
    3. Saturation: Mean saturation in HSV space.
    4. Warmth: Proportion of warm hues (reds, oranges, yellows) in the image.
    5. Edge Density: Average spatial gradient, representing texture level.
    6. Color Entropy: Normalised shannon entropy of grayscale intensities.
    7. Edge Contrast (New): Sharpness indicator (std dev of spatial gradients).
    8. Color Temp Balance (New): Ratio of warm tones to cool tones, representing warmth bias.
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to speed up calculation and normalize resolution impact
        img_small = img.resize((200, 200))
        img_arr = np.array(img_small, dtype=np.float32)
        
        # 1. Brightness
        gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
        brightness = float(np.mean(gray) / 255.0)
        
        # 2. Contrast
        contrast = float(np.std(gray) / 127.5)
        contrast = min(max(contrast, 0.0), 1.0)
        
        # 3. Saturation & 4. Warmth & 8. Color Temp Balance (HSV conversion)
        r, g, b_val = img_arr[:, :, 0] / 255.0, img_arr[:, :, 1] / 255.0, img_arr[:, :, 2] / 255.0
        mx = np.maximum(np.maximum(r, g), b_val)
        mn = np.minimum(np.minimum(r, g), b_val)
        df = mx - mn
        
        # Saturation
        s = np.zeros_like(mx)
        idx = mx > 0
        s[idx] = df[idx] / mx[idx]
        saturation = float(np.mean(s))
        
        # Hue
        h = np.zeros_like(mx)
        idx_r = (mx == r) & (df > 0)
        h[idx_r] = (60 * ((g[idx_r] - b_val[idx_r]) / df[idx_r]) + 360) % 360
        idx_g = (mx == g) & (df > 0)
        h[idx_g] = (60 * ((b_val[idx_g] - r[idx_g]) / df[idx_g]) + 120) % 360
        idx_b = (mx == b_val) & (df > 0)
        h[idx_b] = (60 * ((r[idx_b] - g[idx_b]) / df[idx_b]) + 240) % 360
        
        # Warmth: Hues in [0, 60] (Red to Yellow) or [300, 360] (Magenta to Red) with Saturation > 0.15
        warm_pixels = (((h >= 0) & (h <= 60)) | ((h >= 300) & (h <= 360))) & (s > 0.15)
        warmth = float(np.sum(warm_pixels) / h.size)
        
        # Color Temperature Balance: Ratio of Warm pixels to Cool pixels [120, 240] (Green to Blue)
        cool_pixels = (h >= 120) & (h <= 240) & (s > 0.15)
        warm_count = np.sum(warm_pixels)
        cool_count = np.sum(cool_pixels)
        total_color_pixels = warm_count + cool_count
        if total_color_pixels > 0:
            color_temp = float(warm_count / total_color_pixels)
        else:
            color_temp = 0.5  # Neutral default
            
        # 5. Edge Density (detail complexity) & 7. Edge Contrast (sharpness)
        dx = gray[:, 1:] - gray[:, :-1]
        dy = gray[1:, :] - gray[:-1, :]
        grad = np.abs(dx[1:, :]) + np.abs(dy[:, 1:])
        
        edge_density = float(np.mean(grad) / 30.0)
        edge_density = min(max(edge_density, 0.0), 1.0)
        
        edge_contrast = float(np.std(grad) / 15.0)
        edge_contrast = min(max(edge_contrast, 0.0), 1.0)
        
        # 6. Color Entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        color_entropy = float(entropy / 8.0)
        
        return {
            'brightness': round(brightness, 4),
            'contrast': round(contrast, 4),
            'saturation': round(saturation, 4),
            'warmth': round(warmth, 4),
            'edge_density': round(edge_density, 4),
            'color_entropy': round(color_entropy, 4),
            'edge_contrast': round(edge_contrast, 4),
            'color_temp': round(color_temp, 4)
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Default fallback neutral features
        return {
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'warmth': 0.5,
            'edge_density': 0.5,
            'color_entropy': 0.5,
            'edge_contrast': 0.5,
            'color_temp': 0.5
        }

class ContinuousLearner:
    """
    Manages predictions and weights updates using closed-form Weighted Ridge Regression
    on logit-transformed human scores.
    """
    
    # Feature ordering
    FEATURE_KEYS = [
        'brightness', 'contrast', 'saturation', 'warmth', 
        'edge_density', 'color_entropy', 'edge_contrast', 'color_temp'
    ]
    
    # Default Prior Weights (8-dimensional baseline)
    DEFAULT_WEIGHTS = [0.1, 0.3, 0.2, 0.1, 0.1, 0.05, 0.2, 0.1]
    DEFAULT_BIAS = -0.5
    
    def __init__(self, weights=None, bias=None):
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)
        else:
            self.weights = np.array(self.DEFAULT_WEIGHTS, dtype=np.float32)
            
        if bias is not None:
            self.bias = float(bias)
        else:
            self.bias = float(self.DEFAULT_BIAS)
            
    def predict(self, features):
        """
        Predicts appreciation percentage based on feature dictionary.
        Uses Sigmoid mapping: p = 1 / (1 + exp(-(w^T x + b)))
        Returns predicted percentage in range [0.0, 100.0]
        """
        x = np.array([features[k] for k in self.FEATURE_KEYS], dtype=np.float32)
        z = np.dot(self.weights, x) + self.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
        return float(p * 100.0)
        
    def retrain(self, training_data, regularization=0.5):
        """
        Fits a Weighted Ridge Regression model in closed-form using historical training data.
        
        training_data: list of tuples (feature_dict, rating_percentage, sample_weight)
        regularization (lambda): prevents overfitting and handles low-sample counts.
        """
        if not training_data:
            return self.weights.tolist(), self.bias, 0.0
            
        X_list = []
        y_logit_list = []
        w_sample_list = []
        
        for feats, rating, sample_weight in training_data:
            x = [feats[k] for k in self.FEATURE_KEYS]
            X_list.append(x)
            
            # Map rating [0, 100] to (0, 1) and clamp to avoid infinity in logit
            r_val = rating / 100.0
            r_clamp = max(0.01, min(0.99, r_val))
            logit_r = np.log(r_clamp / (1.0 - r_clamp))
            y_logit_list.append(logit_r)
            w_sample_list.append(sample_weight)
            
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_logit_list, dtype=np.float32)
        W = np.array(w_sample_list, dtype=np.float32)
        
        # Add bias column (column of 1s) to feature matrix X
        N = X.shape[0]
        X_bias = np.hstack([X, np.ones((N, 1), dtype=np.float32)])
        
        # Weighted Ridge Regression closed form:
        # w_bias = (X_bias^T W X_bias + lambda I)^-1 X_bias^T W y
        # We scale X_bias by W along the rows
        X_weighted = X_bias * W[:, np.newaxis]
        
        d_features = X_bias.shape[1]
        I = np.eye(d_features, dtype=np.float32)
        # We do not regularize the bias term as heavily as feature weights
        I[-1, -1] = 0.01 
        
        A = np.dot(X_bias.T, X_weighted) + regularization * I
        B = np.dot(X_weighted.T, y)
        
        try:
            w_bias = np.linalg.solve(A, B)
            self.weights = w_bias[:-1]
            self.bias = float(w_bias[-1])
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            pass
            
        # Calculate Mean Squared Error (MSE) on the percentage scale
        predictions = []
        for feats, _, _ in training_data:
            predictions.append(self.predict(feats))
            
        targets = np.array([item[1] for item in training_data])
        preds = np.array(predictions)
        mse = float(np.mean((targets - preds) ** 2))
        
        return self.weights.tolist(), self.bias, mse

    @classmethod
    def generate_baseline_seeds(cls):
        """
        Generates 20 synthetic baseline samples representing default model behavior.
        Used to seed the database and anchor the learning algorithm.
        Returns a list of tuples: (feature_dict, rating_percentage, sample_weight=1.0)
        """
        seeds = []
        learner = cls()
        
        np.random.seed(42)
        for _ in range(20):
            feats = {
                'brightness': float(np.random.uniform(0.3, 0.7)),
                'contrast': float(np.random.uniform(0.3, 0.8)),
                'saturation': float(np.random.uniform(0.2, 0.8)),
                'warmth': float(np.random.uniform(0.2, 0.7)),
                'edge_density': float(np.random.uniform(0.2, 0.6)),
                'color_entropy': float(np.random.uniform(0.4, 0.8)),
                'edge_contrast': float(np.random.uniform(0.2, 0.7)),
                'color_temp': float(np.random.uniform(0.3, 0.7))
            }
            # Predict default score
            default_score = learner.predict(feats)
            seeds.append((feats, default_score, 1.0))
            
        return seeds
