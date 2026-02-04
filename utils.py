# -*- coding: utf-8 -*-
"""
Quantification of facial hair loss from multitemporal images
CRIOMA - TIDOP Research Group, Escuela Politécnica Superior de Ávila

Author: Innes Barbero-García (ines.barbero@usal.es)
"""

import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import distance
import mediapipe as mp

def filter_main_component(image_path, area_thresh=200, dist_thresh=80, thresh_val=1):
    """
    Implements the custom automated filtering pipeline to isolate the primary ROI 
    and eliminate false-positive background artifacts.
    """
    # 1. Read image
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Binarize
    # Converts probability maps to binary using a high-sensitivity low-intensity threshold 
    # to ensure preservation of fine hair strands.
    _, binary = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Morphological cleaning
    # Applies morphological opening (9x9) to suppress high-frequency noise and dilation (21x21) 
    # to consolidate structural connectivity of the primary subject.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, kernel2)
    
    # 4. Connected components
    # Performs Connected Component Analysis (CCA) with 8-connectivity to label discrete objects.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)

    # 5. Get largest component (excluding background)
    # Identifies the primary hair component based on maximal area.
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    if len(component_areas) == 0:
        print("No components found.")
        return None

    largest_idx = 1 + np.argmax(component_areas)
    main_center = centroids[largest_idx]

    # 6. Mask with largest component and closest neighbours
    # Retains secondary components only if they satisfy spatial proximity (<80px) 
    # and minimum area (>200px) thresholds.
    mask_filtered = np.zeros_like(img_gray, dtype=np.uint8)

    for i in range(1, num_labels):  # omit background
        area = stats[i, cv2.CC_STAT_AREA]
        cX, cY = centroids[i]
        dist = np.linalg.norm(main_center - np.array([cX, cY]))
        if i == largest_idx or (area > area_thresh and dist < dist_thresh):
            mask_filtered[labels == i] = 255

    # 7. Apply mask on original image (without opening)
    # The final optimized mask is applied to the original grayscale image to excise noise.
    filtered_image = cv2.bitwise_and(img_gray, img_gray, mask=mask_filtered)
    cv2.imwrite(image_path, filtered_image)
    return filtered_image

def adapt_brightness(img, v_new_mean):
    """
    Performs photometric normalization in the HSV color space.
    Adjusts the Value (V) channel to match a reference mean intensity.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    value = int(v_new_mean - np.mean(v))
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def clean_and_get_hair_pixels(folder_out, img_name):
    img_path = os.path.join(folder_out, 'outputs_deteccion/DAM_Net_best_model.pth', img_name)
    img = cv2.imread(img_path, 0)
    kernel = np.ones((25, 25), np.uint8)
    _, img_bin = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    labeled, num_objects = ndimage.label(opening)

    mask = np.ones(img.shape + (1,), dtype=np.uint8)
    if num_objects > 0:
        biggest = (labeled == (np.bincount(labeled.flat)[1:].argmax() + 1))
        biggest_ind = np.argwhere(biggest == True)

    for i in range(1, num_objects + 1):
        obj = labeled == i
        ind = np.argwhere(obj == 1)
        if i != (np.bincount(labeled.flat)[1:].argmax() + 1):
            if len(ind) < 100 or \
               ('eye1' in img_name and np.max(ind[:, 1]) < np.min(biggest_ind[:, 1])) or \
               ('eye2' in img_name and np.min(ind[:, 1]) > np.max(biggest_ind[:, 1])) or \
               (np.min(distance.cdist(ind[::5], biggest_ind[::5]).min(axis=1)) > 50):
                mask[obj] = 0

    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    masked_img = img * mask.squeeze()

    labeled, num_objects = ndimage.label(masked_img)
    for i in range(1, num_objects):
        obj = labeled == i
        ind = np.argwhere(obj == 1)
        if len(ind) < 100:
            mask[obj] = 0

    masked_img = masked_img * mask.squeeze()
    _, masked_bin = cv2.threshold(masked_img, 40, 255, cv2.THRESH_BINARY)
    output_path = os.path.join(folder_out, 'outputs_deteccion', img_name)
    cv2.imwrite(output_path, masked_bin)
    return np.count_nonzero(masked_bin)

def extract_eyes(folder, name):
    """
    Detects periocular landmarks and extracts the Region of Interest (ROI).
    """
    image = cv2.imread(folder+name)
    
    # Resize before detection
    # A resized representation is created to stabilize inference and reduce computational costs.
    target_width = 600
    resize_ratio = image.shape[1] / target_width 
    scale_factor = target_width / image.shape[1]
    resized_image = cv2.resize(image, (target_width, int(image.shape[0] * scale_factor)))
    resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Initialize Face Mesh
    # Applies MediaPipe Face Mesh in static image mode to estimate the dense facial mesh.
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    # Detect landmarks on resized image
    results = face_mesh.process(resized_rgb)
    
    
    # MediaPipe Eye Landmark Indices 
    eye1_indices = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55, 33, 7, 163, 144, 145, 153, 154, 155, 133]
    eye2_indices = [336, 296, 334, 293, 300, 285, 295, 282,283,276, 362, 382, 381, 380, 374, 373, 390, 249, 263]

    eye1 = []
    eye2 = []
    
    # Map coordinates back to original resolution.
    if results.multi_face_landmarks:
        
        for face_landmarks in results.multi_face_landmarks:
            for idx in eye1_indices:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * resized_image.shape[1])
                y = int(lm.y * resized_image.shape[0])
                eye1.append((x, y))

    
            for idx in eye2_indices:
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * resized_image.shape[1])
                y = int(lm.y * resized_image.shape[0])
                eye2.append((x, y))
    
    both_eyes = np.concatenate((eye1,eye2))
                
    eye1 = np.array(eye1, dtype=np.int32)
    eye2 = np.array(eye2, dtype=np.int32)
    both_eyes = np.array(both_eyes, dtype=np.int32)


    margin=50
    # Extract subimages from high-res image, using the detected points
    # Computes the ROI by determining the axis-aligned rectangle containing all points plus a margin.
    image_eye1 = image[int((np.min(eye1[:,1])-margin)*resize_ratio):int((np.max(eye1[:,1])+margin)*resize_ratio), int((np.min(eye1[:,0])-margin)*resize_ratio):int((np.max(eye1[:,0])+margin)*resize_ratio)]
    image_eye2 = image[int((np.min(eye2[:,1])-margin)*resize_ratio):int((np.max(eye2[:,1])+margin)*resize_ratio), int((np.min(eye2[:,0])-margin)*resize_ratio):int((np.max(eye2[:,0])+margin)*resize_ratio)]
    image_both_eyes = image[int((np.min(both_eyes[:,1])-margin)*resize_ratio):int((np.max(both_eyes[:,1])+margin)*resize_ratio), int((np.min(both_eyes[:,0])-margin)*resize_ratio):int((np.max(both_eyes[:,0])+margin)*resize_ratio)]
    
    
    image_eye1_rgb = cv2.cvtColor(image_eye1, cv2.COLOR_BGR2RGB)
    image_eye2_rgb = cv2.cvtColor(image_eye2, cv2.COLOR_BGR2RGB)
    image_both_eyes_rgb = cv2.cvtColor(image_both_eyes, cv2.COLOR_BGR2RGB)
        
    plt.imsave(folder + name[:-4] + '_eyes.jpg', image_both_eyes_rgb)
    plt.imsave(folder + name[:-4] + '_eye1.jpg', image_eye1_rgb)
    plt.imsave(folder + name[:-4] + '_eye2.jpg', image_eye2_rgb)
    
    # pixel where eyes area start
    # Stores the ROI top-left origin to allow mapping back to local ROI coordinates.
    ini_both_eyes = (int((np.min(both_eyes[:,1])-margin)*resize_ratio), int((np.min(both_eyes[:,0])-margin)*resize_ratio))
    ini_eye1= (int((np.min(eye1[:,0])-margin)*resize_ratio), int((np.min(eye1[:,1])-margin)*resize_ratio))
    ini_eye2= (int((np.min(eye2[:,0])-margin)*resize_ratio), int((np.min(eye2[:,1])-margin)*resize_ratio))
    return (np.transpose(np.transpose(both_eyes)[0:2])*resize_ratio, np.transpose(np.transpose(eye1)[0:2])*resize_ratio, np.transpose(np.transpose(eye2)[0:2])*resize_ratio, ini_both_eyes, ini_eye1, ini_eye2)


def draw_convex_hull(img, points, color=(1, 1, 1), thickness=20):
    if len(points) >= 3:
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(img, hull, color)
        cv2.polylines(img, [hull], isClosed=True, color=color, thickness=thickness)
    return img

def generate_trimaps_first_image(folder, image_name, eyes, ini_eyes, ini_eye1, ini_eye2):
    """
    Generates eyebrow trimaps by geometric aggregation of the superior periocular landmarks.
    """
    # Two eyes at a time
    image = cv2.imread(os.path.join(folder, image_name))
    image_eye1 = cv2.imread(os.path.join(folder, image_name[:-4] + '_eye1.JPG'))
    image_eye2 = cv2.imread(os.path.join(folder, image_name[:-4] + '_eye2.JPG'))

    tri_b = np.zeros_like(image)
    tri_l = np.zeros_like(image)

    distance_y = abs(eyes[0][1] - eyes[1][1])

    # Eyebrows
    # Computes a convex hull from the landmark set, shifted to the eyebrow location.
    line = np.concatenate((eyes[:10], eyes[19:28]))
    draw_convex_hull(tri_b, line + [0, int(distance_y * 0.6)])
    kernel = np.ones((int(distance_y * 1.5), int(distance_y / 5)), np.uint8)
    tri_b = cv2.dilate(tri_b, kernel, iterations=1) * 128
    
    # Eyelashes
    draw_convex_hull(tri_l, eyes[10:19])
    draw_convex_hull(tri_l, eyes[28:])
    kernel = np.ones((int(distance_y * 0.8), int(distance_y * 0.8)), np.uint8)
    tri_l = cv2.dilate(tri_l, kernel, iterations=1) * 128
    
    if not os.path.exists(os.path.join(folder, 'image1')):
        os.makedirs(os.path.join(folder, 'image1'))
        
    if not os.path.exists(os.path.join(folder, 'image1', 'trimap')):
        os.makedirs(os.path.join(folder, 'image1', 'trimap'))
        

    # Crop trimaps to the specific eye ROIs.
    for label, tri, ini, eye_img in [('b_eye1', tri_b, ini_eye1, image_eye1),
                                     ('b_eye2', tri_b, ini_eye2, image_eye2),
                                     ('l_eye1', tri_l, ini_eye1, image_eye1),
                                     ('l_eye2', tri_l, ini_eye2, image_eye2)]:
        trimap_cropped = tri[ini[1]:ini[1] + eye_img.shape[0], ini[0]:ini[0] + eye_img.shape[1]]
        # Save images and trimaps in a specific folder to process separately
        cv2.imwrite(os.path.join(folder, 'image1', 'trimap', f"{image_name[:-4]}_{label}.jpg"), trimap_cropped)
        
def copy_t1_images(folder):
    """
    Searches for .jpg files starting with 'T1' in the source folder 
    and copies them to the final folder.
    """
    # 1. Create the destination folder if it doesn't exist
    source_folder = os.path.join(folder,'images')
    destination_folder = os.path.join(folder, 'image1','images')
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created new folder: {destination_folder}")

    files_copied = 0

    # 2. Iterate through files in the source folder
    try:
        for filename in os.listdir(source_folder):
            # 3. Check conditions: Starts with 'T1' AND ends with '.jpg' (case-insensitive)
            if filename.startswith("T1") and filename.lower().endswith(".jpg"):
                
                # Construct full file paths
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, filename)

                # 4. Copy the file (only if it is a file, not a directory)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {filename}")
                    files_copied += 1
        
        print(f"\nOperation complete. Total files copied: {files_copied}")

    except FileNotFoundError:
        print(f"Error: The source folder '{source_folder}' was not found.")
    except PermissionError:
        print("Error: Permission denied. Check your folder permissions.")
        
                    
def generate_trimaps_other_images(folder, image_time, DAM_NET_CHECKPOINT ):

    """
    Generates a trimap mask based on the convex hull 
    of the white pixels from the first image detection (binary).
    """  
    
    
    if not os.path.exists(os.path.join(folder, 'trimaps')):
        os.makedirs(os.path.join(folder, 'trimaps'))
        print('Generated trimaps folder')

    # For each subimage of T1
    for label in ['_b_eye1', '_b_eye2', '_l_eye1', '_l_eye2']:
        filter_main_component(os.path.join(folder, 'outputs_detection', 'DAM_Net_best_model.pth', 'T1' + label +'.jpg'))    
        binary_image = cv2.imread(os.path.join(folder, 'outputs_detection', 'DAM_Net_best_model.pth', 'T1' + label +'.jpg'), 0)
        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if binary_image is None:
            print("Warning: Mask not found")
            continue 
            
        if not contours:
            print('Image not loaded correctly')
            return np.zeros_like(binary_image)
    
        # 2. Unify all points
        # If there are multiple white "spots", create a single Convex Hull covering them all:
        all_points = np.vstack(contours)
        
        # 3. Compute Convex Hull
        # Uses the geometry of the detected hair to define the support region.
        hull = cv2.convexHull(all_points)
        
        # 4. Draw filled Hull on a black image
        hull_mask = np.zeros_like(binary_image)
        cv2.drawContours(hull_mask, [hull], -1, (255), thickness=cv2.FILLED)
        
        
    
        # 5. Dilate
        # Expand the region to include all plausible eyebrow hair pixels.
        kernel = np.ones((10, 10), np.uint8)
        hull_mask = cv2.dilate(hull_mask, kernel, iterations=1)
            
            
    
        # 6. Convert to intensity 128 (trimap format)
        # Convert any pixel > 0 to 128
        trimap_output = (hull_mask > 0).astype(np.uint8) * 128
        
  
        cv2.imwrite(os.path.join(folder, 'trimaps', f"{image_time}{label}_tr2.jpg"), trimap_output )