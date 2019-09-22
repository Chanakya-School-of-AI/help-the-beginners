from skimage.color import rgb2lab
from skimage.color import deltaE_cie76
import color_extraction_image
import numpy as np

def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors,_,_ = color_extraction_image.extract_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image