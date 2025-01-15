import pickle
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd

def calculate_entropy(array):
    # Get the histogram of pixel values
    hist, _ = np.histogram(array, bins=256, range=(0, 256), density=True)
    
    # Remove zero entries (log(0) is undefined)
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

img_path = "./test/dino_5.jpg"

img = Image.open(img_path)
img = np.array(img)

# convert to HLS
img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

features = {}

hue, lightness, saturation = cv2.split(img_hsl)
lightness_mean = np.mean(lightness)
lightness_std = np.std(lightness)
saturation_mean = np.mean(saturation)
saturation_std = np.std(saturation)
lightness_entropy = calculate_entropy(lightness)

X_dict = {
    "l_mean": lightness_mean,
    "l_std": lightness_std,
    "s_mean": saturation_mean,
    "s_std": saturation_std,
    "l_entropy": lightness_entropy
}
X = pd.DataFrame([X_dict])
print(X)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/model.pkl', 'rb') as f:
    rf = pickle.load(f)


X = scaler.transform(X)
print(X)
y = rf.predict(X)

print("Light classification:")
print(y)


