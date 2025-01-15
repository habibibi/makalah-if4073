import csv
import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np


def calculate_entropy(array):
    # Get the histogram of pixel values
    hist, _ = np.histogram(array, bins=256, range=(0, 256), density=True)
    
    # Remove zero entries (log(0) is undefined)
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


input_dir = './dataset' 
output_dir = './processed'

os.makedirs(output_dir, exist_ok=True)


light_level = [0.0,0.4,0.8,1.2,1.6,2.0]
label = ['very_dark','dark','normal','bright','very_bright']
csv_data = [["l_mean","l_std","s_mean","s_std", "l_entropy","s_entropy", "class"]]

for filename in os.listdir(input_dir):
    name = os.path.splitext(filename)[0]
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)

    # Apply transformations
    enhancer = ImageEnhance.Brightness(img)

    for i in range(len(label)):
        for j in range(3):
            factor = np.random.uniform(light_level[i],light_level[i+1])
            img_proc = enhancer.enhance(factor)

            output_path = os.path.join(output_dir, name + "_" + label[i] + ".jpg")
            img_proc.save(output_path)

            # convert to numpy format (RGB)
            img_rgb = np.array(img_proc)

            # convert to HLS
            img_hsl = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)

            features = {}

            hue, lightness, saturation = cv2.split(img_hsl)
            lightness_mean = np.mean(lightness)
            lightness_std = np.std(lightness)
            saturation_mean = np.mean(saturation)
            saturation_std = np.std(saturation)
            lightness_entropy = calculate_entropy(lightness)
            saturation_entropy = calculate_entropy(saturation)

            csv_data.append([lightness_mean, lightness_std, saturation_mean, saturation_std, lightness_entropy, saturation_entropy, i])

with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)



        