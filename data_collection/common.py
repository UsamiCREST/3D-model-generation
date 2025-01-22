### Common functions ###

def color_sides(img):
    # color sides
    h, w = img.shape[0], img.shape[1]
    for i in range(h):
        img[i, 0] = 255
        img[i, -1] = 255
        
    for j in range(1, w):
        img[0, j] = 255
        img[-1, j] = 255
    return img

def segmentation(img):
    h, w = img.shape[0], img.shape[1] # Obtain size of image
    end = h - 1
    Grain_number = 1
    # for grayscale and edge image
    if len(img.shape) == 2:
        img = color_sides(img)
    # for color images
    elif len(img.shape) == 3:
        # find edges
        img = cv2.Canny(img, 5, 220)
        img = color_sides(img)
        # dilate edge
        kernel = np.ones((3,3),np.uint8)
        img = cv2.dilate(img, kernel, iterations = 1)
        # erode edge
        kernel = np.ones((3, 3),np.uint8)
        img = cv2.erode(img, kernel, iterations = 1)
        
    # Change image type to prepare for segmentation and change grain boundary 255 to -1 for later convinience
    img = img.astype(np.int32)
    img[img == 255] = -1 
    
    # Label the first grain to start with
    for i in range(w):
        if img[0][i]== 0:
            img[0][i] = 1
            Grain_number = 1
            break
        else:
            continue
            
    unique_values = np.unique(img[0][:])
    
    # Label first row
    for j in range(w):
        if j == 0:   # For the most left pixel
            print(img[0][j])
        else:   # For non most left pixel
            if img[0][j] != -1:   # Determine whether an arbitrary pixel was located insider or outside a crystal grain
                if img[0][j-1] == -1:   # When left pixel is a grain boundary
                    Grain_number += 1
                    img[0][j] = Grain_number
                else:   # When left pixel is not a grain boundary
                    img[0][j] = img[0][j-1]
    
    # For 2~ rows
    for i in trange(1, h):
        for j in range(w):
            if j == 0:   # For the most left pixel
                if img[i][j] == 0:
                    if img[i-1][j] != -1:
                        img[i][j] = img[i-1][j]
                    else:
                        Grain_number += 1
                        img[i][j] = Grain_number
                        
            else:
                if img[i][j] == 0:
                    candidate = []
                    if (img[i-1][j] == -1) and (img[i][j-1] == -1):
                        Grain_number += 1
                        img[i][j] = Grain_number
                    else:
                        if img[i-1][j] != -1:
                            candidate.append(img[i-1][j])
                        if img[i][j-1] != -1:
                            candidate.append(img[i][j-1])
                        img[i][j] = min(candidate)
    
    # Correction of mislabeling from bottom right pixel
    num_correction = 0
    for i in trange(end,-1,-1):
        for j in range(end,-1,-1):
            if img[i][j] != -1:
                candidate = [99999]
                
                if i==end and j==end:
                    if img[i-1][j] != -1:
                        candidate.append(img[i-1][j])
                    if img[i][j-1] != -1:
                        candidate.append(img[i][j-1])
                        
                    if img[i][j] > min(candidate):
                        num_correction += 1
                        img[i][j] = min(candidate)
                    
                elif i==end:
                    if img[i-1][j] != -1:
                        candidate.append(img[i-1][j])
                    if img[i][j-1] != -1:
                        candidate.append(img[i][j-1])
                    if img[i][j+1] != -1:
                        candidate.append(img[i][j+1])
                        
                    if img[i][j] > min(candidate):
                        num_correction += 1
                        img[i][j] = min(candidate)
                
                elif j==end:
                    if img[i-1][j] != -1:
                        candidate.append(img[i-1][j])
                    if img[i][j-1] != -1:
                        candidate.append(img[i][j-1])
                    if img[i+1][j] != -1:
                        candidate.append(img[i+1][j])
                        
                    if img[i][j] > min(candidate):
                        num_correction += 1
                        img[i][j] = min(candidate)
                
                else:
                    if img[i-1][j] != -1:
                        candidate.append(img[i-1][j])
                    if img[i+1][j] != -1:
                        candidate.append(img[i+1][j])
                    if img[i][j-1] != -1:
                        candidate.append(img[i][j-1])
                    if img[i][j+1] != -1:
                        candidate.append(img[i][j+1])
                        
                    if img[i][j] > min(candidate):
                        num_correction += 1
                        img[i][j] = min(candidate)
    
    for i in trange(h*w):
        if num_correction == 0:
            break
            
        else:
            num_correction = 0
    
            for i in range(end,-1,-1):
                for j in range(end,-1,-1):
                    if img[i][j] != -1:
                        candidate = [99999]
    
    
                        if i==end and j==end:
                            if img[i-1][j] != -1:
                                candidate.append(img[i-1][j])
                            if img[i][j-1] != -1:
                                candidate.append(img[i][j-1])
    
                            if img[i][j] > min(candidate):
                                num_correction += 1
                                img[i][j] = min(candidate)
    
    
                        elif i==end:
                            if img[i-1][j] != -1:
                                candidate.append(img[i-1][j])
                            if img[i][j-1] != -1:
                                candidate.append(img[i][j-1])
                            if img[i][j+1] != -1:
                                candidate.append(img[i][j+1])
    
                            if img[i][j] > min(candidate):
                                num_correction += 1
                                img[i][j] = min(candidate)
    
    
                        elif j==end:
                            if img[i-1][j] != -1:
                                candidate.append(img[i-1][j])
                            if img[i][j-1] != -1:
                                candidate.append(img[i][j-1])
                            if img[i+1][j] != -1:
                                candidate.append(img[i+1][j])
    
                            if img[i][j] > min(candidate):
                                num_correction += 1
                                img[i][j] = min(candidate)
    
                        else:
                            if img[i-1][j] != -1:
                                candidate.append(img[i-1][j])
                            if img[i+1][j] != -1:
                                candidate.append(img[i+1][j])
                            if img[i][j-1] != -1:
                                candidate.append(img[i][j-1])
                            if img[i][j+1] != -1:
                                candidate.append(img[i][j+1])
    
                            if img[i][j] > min(candidate):
                                num_correction += 1
                                img[i][j] = min(candidate)
                                
    return img

def get_RGB(p, blank='w'):
    """
    Return IPF colors for stereographic position.
    Parameter
    ---------
    p : (n, 2) array
        (x, y) of n points in stereographic plane.
    Return
    ------
    RGB : (n, 3) array
        Normalized RGB values
    """
    if p.ndim == 1:
        p = np.expand_dims(p, axis=0)
    # Vertices
    p0 = np.array([0, 0])
    p1 = np.array([np.sqrt(2)-1, 0])
    p2 = np.array([1, 1])*(np.sqrt(3)-1)/2
    pG = (p0 + p1 + p2)/3
    # Segments
    m0, m1, m2 = (p0, p1, p2) - pG
    s0 = p1 - p0
    s2 = p0 - p2
    distance = lambda p, s, ps: np.abs(np.cross(s, p-ps))/np.linalg.norm(s)
    L_P0_M1 = distance(p0, m1, pG)
    L_P0_M2 = distance(p0, m2, pG)
    L_P1_M2 = distance(p1, m2, pG)
    L_P1_M0 = distance(p1, m0, pG)
    L_P2_M0 = distance(p2, m0, pG)
    L_P2_M1 = distance(p2, m1, pG)
    L_PG_S0 = distance(pG, s0, p0)
    L_PG_S2 = distance(pG, s2, p2)
    is_inside = np.all([p[:,1]<=p[:,0],
                        (p[:,0]+1)**2+p[:,1]**2<=np.sqrt(2)**2,
                        p[:,1]>=0], axis=0)
    is_above = lambda p, v: 1/np.divide(*(pG-v))*(p[:,0]-pG[0])+pG[1] <= p[:,1]
    is_where = np.array([is_inside, is_above(p, p0), is_above(p, p1), is_above(p, p2)]).T
    bln_region = [[True, False, False, True], [True, True, False, True], # Red
                  [True, False, False, False], [True, False, True, False], # Green
                  [True, True, True, False],[True, True, True, True]] # Blue
    if blank == 'w':
        RGB = np.ones((len(p), 3))
    else:
        RGB = np.zeros((len(p), 3))
    # Red-1
    indx = np.arange(len(p))[np.all(is_where == bln_region[0], axis=-1)]
    #print(indx.size)
    if not(indx.size == 0):    
        p_ = p[indx]
        r = np.ones(len(p_))
        g = 1 - distance(p_, m2, pG)/L_P0_M2
        b = distance(p_, s0, p0)/L_PG_S0
        RGB[indx] = np.array([r, g, b]).T
    # Red-2   
    indx = np.arange(len(p))[np.all(is_where == bln_region[1], axis=-1)]
    #print(indx.size)
    if not(indx.size == 0):
        p_ = p[indx]
        r = np.ones(len(p_))
        b = 1 - distance(p_, m1, pG)/L_P0_M1
        g = distance(p_, s2, p2)/L_PG_S2
        RGB[indx] = np.array([r, g, b]).T
    # Green-1
    indx = np.arange(len(p))[np.all(is_where == bln_region[2], axis=-1)]
    #print(indx.size)
    if not(indx.size == 0):
        p_ = p[indx]
        g = np.ones(len(p_))
        r = 1 - distance(p_, m2, pG)/L_P1_M2
        b = distance(p_, s0, p0)/L_PG_S0
        RGB[indx] = np.array([r, g, b]).T
    # Green-2
    indx = np.arange(len(p))[np.all(is_where == bln_region[3], axis=-1)]
    #print(indx.size)
    if not(indx.size == 0):
        p_ = p[indx]
        g = np.ones(len(p_))
        b = 1 - distance(p_, m0, pG)/L_P1_M0
        slope = 1/np.divide(*(pG-p_).T)
        x_sol = 1/(slope**2+1)*(slope**2*pG[0] - slope*pG[1]
                                + np.sqrt(-slope**2*pG[0]**2 - 2*slope**2*pG[0] + slope**2
                                          + 2*slope*pG[0]*pG[1] + 2*slope*pG[1] - pG[1]**2 + 2) - 1)
        px = np.array([x_sol, slope*(x_sol-pG[0])+pG[1]]).T
        r = np.linalg.norm(p_-px, axis=1)/np.linalg.norm(pG-px, axis=1)
        RGB[indx] = np.array([r, g, b]).T
    # Blue-1
    indx = np.arange(len(p))[np.all(is_where == bln_region[4], axis=-1)]
    if not(indx.size == 0):    
        p_ = p[indx]
        b = np.ones(len(p_))
        g = 1 - distance(p_, m0, pG)/L_P2_M0
        slope = 1/np.divide(*(pG-p_).T)
        x_sol = 1/(slope**2+1)*(slope**2*pG[0] - slope*pG[1]
                                + np.sqrt(-slope**2*pG[0]**2 - 2*slope**2*pG[0] + slope**2
                                          + 2*slope*pG[0]*pG[1] + 2*slope*pG[1] - pG[1]**2 + 2) - 1)
        px = np.array([x_sol, slope*(x_sol-pG[0])+pG[1]]).T
        r = np.linalg.norm(p_-px, axis=1)/np.linalg.norm(pG-px, axis=1)
        RGB[indx] = np.array([r, g, b]).T
    # Blue-2
    indx = np.arange(len(p))[np.all(is_where == bln_region[5], axis=-1)]
    if not(indx.size == 0):
        p_ = p[indx]
        b = np.ones(len(p_))
        r = 1 - distance(p_, m1, pG)/L_P2_M1
        g = distance(p_, s2, p2)/L_PG_S2
        RGB[indx] = np.array([r, g, b]).T

    return np.clip(RGB, 0, 1)

def stereographic2sst(v):
    """
    Return stereographic positions of orientations
    in standard stereographic triangle.

    Parameter
    ---------
    v : (n, 3) array
        n orientation vectors.

    Return
    ------
    p : (n, 2) array
        (x, y) of n points in stereographic plane.
    """
    if v.ndim == 1:
        v = np.expand_dims(v, axis=0)
    v = v/np.linalg.norm(v, axis=1).reshape(-1,1)
    yxz = np.sort(np.abs(v),axis=1)
    vs = yxz[:,:2]/(1+yxz[:,2]).reshape(-1,1)
    vs = np.roll(vs, 1, axis=1)

    return vs

def get_representative(im):
    h, w = im.shape[:2]
    ixx, iyy = np.meshgrid(np.arange(w), np.arange(h))
    
    im_ = binary_fill_holes(im)
    im_[:,[0,-1]] = 0
    im_[[0,-1],:] = 0
    im_dt = distance_transform_edt(im_)
    indx = im_dt.argmax()
    ix = ixx.flatten()[indx]
    iy = iyy.flatten()[indx]
    return ix, iy

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Generalized function to fill any target RGB color in an image
def fill_target_color_in_image(image, target_rgb):
   # Create a mask to detect the target color (black or any specified RGB value) in the first three channels (RGB)
    target_mask = np.all(image[..., :3] == target_rgb, axis=-1)

    # Create a copy of the original image for modification
    filled_image = image.copy()

    # Loop through each pixel in the image and replace the target color pixels
    for i in range(1, target_mask.shape[0] - 1):
        for j in range(1, target_mask.shape[1] - 1):
            if target_mask[i, j]:  # If the pixel matches the target color
                # Get the neighboring pixels that are not the target color
                neighbors = [
                    filled_image[i-1, j, :3],  # RGB channels
                    filled_image[i+1, j, :3],
                    filled_image[i, j-1, :3],
                    filled_image[i, j+1, :3]
                ]
                # Remove neighbors that match the target color
                neighbors = [n for n in neighbors if not np.all(n == target_rgb)]
                if neighbors:
                    # Replace the target pixel's RGB channels with a random neighbor color
                    # filled_image[i, j, :3] = random.choice(neighbors)
                    # Replace the target pixel's RGB channels with the most frequent neighboring color (dominant)
                    unique, counts = np.unique(neighbors, axis=0, return_counts=True)
                    # unique, counts = np.unique(neighbors, axis=1, return_counts=True)
                    filled_image[i, j, :3] = unique[np.argmax(counts)]
                    # filled_image[i, j, :3] = neighbors[0]

    # color sides
    h, w = filled_image.shape[0], filled_image.shape[1]
    for i in range(h):
        filled_image[i, 0] = filled_image[i, 1]
        filled_image[i, -1] = filled_image[i, -2]
        
    for j in range(1, w):
        filled_image[0, j] = filled_image[1, j]
        filled_image[-1, j] = filled_image[-2, j]
    
    return filled_image

# Generalized function to fill any target RGB color in an image
def randomly_fill_target_color_in_image(image, target_rgb):
   # Create a mask to detect the target color (black or any specified RGB value) in the first three channels (RGB)
    target_mask = np.all(image[..., :3] == target_rgb, axis=-1)

    # Create a copy of the original image for modification
    filled_image = image.copy()

    # Loop through each pixel in the image and replace the target color pixels
    for i in range(1, target_mask.shape[0] - 1):
        for j in range(1, target_mask.shape[1] - 1):
            if target_mask[i, j]:  # If the pixel matches the target color
                # Get the neighboring pixels that are not the target color
                neighbors = [
                    filled_image[i-1, j, :3],  # RGB channels
                    filled_image[i+1, j, :3],
                    filled_image[i, j-1, :3],
                    filled_image[i, j+1, :3]
                ]
                # Remove neighbors that match the target color
                neighbors = [n for n in neighbors if not np.all(n == target_rgb)]
                if neighbors:
                    # Replace the target pixel's RGB channels with a random neighbor color
                    filled_image[i, j, :3] = random.choice(neighbors)
                    # Replace the target pixel's RGB channels with the most frequent neighboring color (dominant)
                    unique, counts = np.unique(neighbors, axis=0, return_counts=True)
                    # unique, counts = np.unique(neighbors, axis=1, return_counts=True)
                    # filled_image[i, j, :3] = unique[np.argmax(counts)]
                    # filled_image[i, j, :3] = neighbors[0]

    # color sides
    h, w = filled_image.shape[0], filled_image.shape[1]
    for i in range(h):
        filled_image[i, 0] = filled_image[i, 1]
        filled_image[i, -1] = filled_image[i, -2]
        
    for j in range(1, w):
        filled_image[0, j] = filled_image[1, j]
        filled_image[-1, j] = filled_image[-2, j]
    
    return filled_image

def labeling(img):
    print("return labeled image and dataframe")
    R, G, B, x, y = [], [], [], [], []
    height, width = img.shape[0], img.shape[1]
    # row = 0
    for h in range(height):
        for w in range(width):
            R.append(img[h, w][0])
            G.append(img[h, w][1])
            B.append(img[h, w][2])
            x.append(h)
            y.append(w)
            # df_color.loc[row] = img[h, w][0], img[h, w][1], img[h, w][2], h, w
            # row += 1
    
    img_label = segmentation(img)
    # fig, ax = plt.subplots()
    # ax.imshow(img_label);
    grain_id = img_label.reshape(img.shape[0] * img.shape[1])
    
    data = {"R":R, "G":G, "B":B, "x":x, "y":y, "ID":grain_id}
    df = pd.DataFrame.from_dict(data)
    return img_label, df

def color_smallgrains(img, df):
    print("return image colored black for small grains")
    # paint black for small grains that means area under 50
    df_count = df.groupby("ID").size().reset_index(name = "count").copy()
    small_grains_ID = df_count[df_count["count"] <= 100]["ID"].values
    # print(small_grains_ID)
    for id in small_grains_ID:
        df.loc[df["ID"] == id, "R"] = 0
        df.loc[df["ID"] == id, "G"] = 0
        df.loc[df["ID"] == id, "B"] = 0
        
    df.loc[df["ID"] == -1, "R"] = 0
    df.loc[df["ID"] == -1, "G"] = 0
    df.loc[df["ID"] == -1, "B"] = 0
    selected_columns = ["R", "G", "B"]
    img_blacked = df[selected_columns].values
    img_blacked = img_blacked.reshape(img.shape[0],img.shape[1],3)
    # fig, ax = plt.subplots()
    # ax.imshow(img_blacked);
    img_blacked_filled = randomly_fill_target_color_in_image(img_blacked, (0,0,0))
    # fig, ax = plt.subplots()
    # ax.imshow(img_blacked_filled);
    return img_blacked_filled, df

def get_centroids(df):
    print("return df with centroids and lists")
    centroid_x_dict = {}
    centroid_y_dict = {}
    centroid_x_list = []
    centroid_y_list = []

    id_list = np.unique(df["ID"].values)
    for ID in id_list[1:]:
        x_coords = df[df["ID"] == ID]["x"].values
        centroid_x = sum(x_coords)/len(x_coords)
        centroid_x_dict[ID] = centroid_x
        centroid_x_list.append(centroid_x)
        
        y_coords = df[df["ID"] == ID]["y"].values
        centroid_y = sum(y_coords)/len(y_coords)
        centroid_y_dict[ID] = centroid_y
        centroid_y_list.append(centroid_y)
    df["r_x"] = df["ID"].map(centroid_x_dict)
    df["r_y"] = df["ID"].map(centroid_y_dict)
    return df, centroid_x_list, centroid_y_list

def average_grain(df):
    df_average = df.copy()
    id_list = np.unique(df["ID"].values)
    
    # color grain as black
    condition = df_average["ID"] == -1
    df_average.loc[condition, "R"] = 0
    df_average.loc[condition, "G"] = 0
    df_average.loc[condition, "B"] = 0
    
    for ID in id_list[1:]:
        condition = df_average["ID"] == ID
        df_ID = df[condition]
        mean_R = int(df_ID["R"].mean())
        mean_G = int(df_ID["G"].mean())
        mean_B = int(df_ID["B"].mean())
    
        df_average.loc[condition, "R"] = mean_R
        df_average.loc[condition, "G"] = mean_G
        df_average.loc[condition, "B"] = mean_B
    
    selected_columns = ["R", "G", "B"]
    img_average = df_average[selected_columns].values
    img_average = img_average.reshape(256,256,3)
    img_average = color_sides(img_average)
    
    # fig, ax = plt.subplots()
    # ax.imshow(img_average);
    return df_average, img_average

def get_gb_image(img):
    edge = cv2.Canny(img, 100, 200)
    # dilate edge
    kernel = np.ones((5,5),np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    edge = cv2.dilate(edge, kernel, iterations = 1)

    # erode edge
    kernel = np.ones((3, 3),np.uint8)
    edge = cv2.erode(edge, kernel, iterations = 1)    

    img[edge == 255] = (0,0,0)
    return img

import numpy as np
import random
import cv2
from skimage.morphology import skeletonize

def fill_target_color_with_non_black(image, target_rgb):
    # Create a mask for the target color
    target_mask = np.all(image[..., :3] == target_rgb, axis=-1)
    filled_image = image.copy()
    h, w = target_mask.shape

    # Use padding to avoid boundary issues
    padded_image = np.pad(filled_image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    padded_mask = np.pad(target_mask, ((1, 1), (1, 1)), mode='constant', constant_values=False)

    # Iterate over all target pixels
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if padded_mask[i, j]:  # If pixel matches target color
                # Get the 4-connected neighbors
                neighbors = [
                    padded_image[i-1, j, :3],
                    padded_image[i+1, j, :3],
                    padded_image[i, j-1, :3],
                    padded_image[i, j+1, :3]
                ]
                # Remove neighbors that match the target color or are black
                neighbors = [n for n in neighbors if not np.all(n == target_rgb) and not np.all(n == [0, 0, 0])]
                if neighbors:
                    # Replace with the dominant neighbor color
                    unique, counts = np.unique(neighbors, axis=0, return_counts=True)
                    filled_image[i-1, j-1, :3] = unique[np.argmax(counts)]

    return filled_image

def skeletonize_original(img):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    skeleton = skeletonize(binary_mask // 255).astype(np.uint8) * 255
    
    result = image.copy()
    for i in range(3):
        result[:, :, i][binary_mask == 255] = 255
        result[:, :, i][skeleton == 255] = 0
    return result

def skeletonize_fill_white(image, repeat):
    for i in range(repeat):
        image = skeletonize_original(image)
        image = randomly_fill_target_color_in_image(image, (255, 255, 255))
    return image

def skeletonize_fill_all(image, repeat):
    for i in range(repeat):
        image = skeletonize_original(image)
        image = randomly_fill_target_color_in_image(image, (255, 255, 255))
    image = randomly_fill_target_color_in_image(image, (0, 0, 0))
    return image