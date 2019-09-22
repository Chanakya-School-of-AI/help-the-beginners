import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np 
from collections import Counter

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

image_path = "./example-images/Swimming_dog_bgiu.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
# Now, flatten the image
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

# Fit KMeans clustering to the points
clf = KMeans(n_clusters = 10)
labels = clf.fit_predict(modified_image)

counts = Counter(labels)

center_colors = clf.cluster_centers_

ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

plt.figure(figsize = (8, 6))
plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
plt.savefig('color-pie-chart.png')
