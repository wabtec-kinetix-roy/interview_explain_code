import os
import numpy as np
import cv2
from scipy.io import loadmat
from skimage.morphology import remove_small_objects
from glob import glob
from PIL import Image


# Paths
MASK_DIR = 'masks'
IMAGE_DIR = 'images'
OUT_IMAGE_DIR = 'output'

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

# Label List
components = ['follower_block', 'plate_type_1', 'car_body_striker', 'coupler_head']

Y_DIM = 2048
X_DIM = 5712

im_files = glob(os.path.join(IMAGE_DIR, '*.jpg'))

for full_file_path in im_files:

    file_name = os.path.basename(full_file_path)

    im_name, ext = os.path.splitext(file_name)
	
    if ext.lower() not in ['.jpg', '.bmp', '.png']:
        continue
    
    mask_dir_path = os.path.join(MASK_DIR, im_name)
    
    mat_file = os.path.join(mask_dir_path, f"{im_name}.mat")
    
    if not os.path.exists(mat_file):
        continue 

    m = loadmat(mat_file)														
 
    fields = [f for f in m.keys() if not f.startswith('__')]
    
    final_mask = np.zeros((Y_DIM, X_DIM), dtype=np.uint8)
    
    for ic, component in enumerate(components, 1):

        field_names = [f for f in fields if component in f]
        
        mask = np.zeros((Y_DIM, X_DIM), dtype=bool)
        
        # What happening inside below for loop? Any issues ?
        for field in field_names:
            mask_data = m[field].astype(bool)
            if mask_data.shape != (Y_DIM, X_DIM):
                mask_data = cv2.resize(mask_data.astype(np.uint8), (X_DIM, Y_DIM), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask |= mask_data							 

        mask = remove_small_objects(mask, min_size=50)  

        final_mask = final_mask + (mask * ic) # What happening here? Any issues ?

    cmap = np.array([
                    [0 , 0, 0], 
                    [100, 0, 200], 
                    [100, 200, 0], 
                    [0, 100, 200],
                    [200, 0, 0]
                ]) / 255.0
    
    h, w = final_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)									
    
    rgb[final_mask == 0] = cmap[0]
    
    for i in range(len(components)):
        rgb[final_mask == (i+1)] = cmap[i+1]
    
    unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)
    print("Unique colors in the image:", unique_colors.shape[0])

    rgb_half = cv2.resize(rgb, (w//2, h//2), interpolation=cv2.INTER_LINEAR)  # What happening here? Any issues ?

    output_path = os.path.join(OUT_IMAGE_DIR, f"{im_name}.png")																   
    
    cv2.imwrite(output_path, cv2.cvtColor(rgb_half*255, cv2.COLOR_RGB2BGR))

    unique_colors_half = np.unique(rgb_half.reshape(-1, 3), axis=0)
    print("Unique colors in the resized image:", unique_colors_half.shape[0])