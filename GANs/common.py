# define function

def segmentation(img):
    h, w = img.shape[0], img.shape[1]
    end = h-1
    
    # find edges
    edge = cv2.Canny(img, 5, 220)

    # make mask to color the right side with white
    mask = np.zeros((h, w))
    for i in range(1, mask.shape[0] - 1):
        # mask[i, 0] = 255
        mask[i, end] = 255
    # for j in range(mask.shape[1]):
    #     mask[0, j] = end
    #     mask[end, j] = end

    edge[mask == end] = 255

    # dilate edge
    kernel = np.ones((4,4),np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    edge = cv2.dilate(edge, kernel, iterations = 1)

    # erode edge
    kernel = np.ones((3, 3),np.uint8)
    edge = cv2.erode(edge, kernel, iterations = 1)

    # Change image type to prepare for segmentation and change grain boundary 255 to -1 for later convinience
    edge = edge.astype(np.int32)
    
    edge[edge == 255] = -1

    for i in range(w):
        if edge[0][i]==0:
            edge[0][i] = 1
            Grain_number = 1
            break
        else:
            continue

    # Label first row
    for j in range(w):
        if j == 0:   # For the most left pixel
            print(edge[0][j])
        else:   # For non most left pixel
            if edge[0][j] != -1:   # Determine whether an arbitrary pixel was located insider or outside a crystal grain
                if edge[0][j-1] == -1:
                    Grain_number += 1
                    edge[0][j] = Grain_number
                else:
                    edge[0][j] = edge[0][j-1]
    
    # For 2~ rows
    for i in trange(1,h):
        for j in range(w):
            if j == 0:
                if edge[i][j] == 0:
                    if edge[i-1][j] != -1:
                        edge[i][j] = edge[i-1][j]
                    else:
                        Grain_number += 1
                        edge[i][j] = Grain_number
                        
            else:
                if edge[i][j] == 0:
                    candidate = []
                    if (edge[i-1][j] == -1) and (edge[i][j-1] == -1):
                        Grain_number += 1
                        edge[i][j] = Grain_number
                    else:
                        if edge[i-1][j] != -1:
                            candidate.append(edge[i-1][j])
                        if edge[i][j-1] != -1:
                            candidate.append(edge[i][j-1])
                        edge[i][j] = min(candidate)
    
    # Correction of mislabeling from bottom right pixel
    num_correction = 0
    for i in trange(end,-1,-1):
        for j in range(end,-1,-1):
            if edge[i][j] != -1:
                candidate = [99999]
                
                
                if i==end and j==end:
                    if edge[i-1][j] != -1:
                        candidate.append(edge[i-1][j])
                    if edge[i][j-1] != -1:
                        candidate.append(edge[i][j-1])
                        
                    if edge[i][j] > min(candidate):
                        num_correction += 1
                        edge[i][j] = min(candidate)
                    
                    
                elif i==end:
                    if edge[i-1][j] != -1:
                        candidate.append(edge[i-1][j])
                    if edge[i][j-1] != -1:
                        candidate.append(edge[i][j-1])
                    if edge[i][j+1] != -1:
                        candidate.append(edge[i][j+1])
                        
                    if edge[i][j] > min(candidate):
                        num_correction += 1
                        edge[i][j] = min(candidate)
                
                
                elif j==end:
                    if edge[i-1][j] != -1:
                        candidate.append(edge[i-1][j])
                    if edge[i][j-1] != -1:
                        candidate.append(edge[i][j-1])
                    if edge[i+1][j] != -1:
                        candidate.append(edge[i+1][j])
                        
                    if edge[i][j] > min(candidate):
                        num_correction += 1
                        edge[i][j] = min(candidate)
                
                else:
                    if edge[i-1][j] != -1:
                        candidate.append(edge[i-1][j])
                    if edge[i+1][j] != -1:
                        candidate.append(edge[i+1][j])
                    if edge[i][j-1] != -1:
                        candidate.append(edge[i][j-1])
                    if edge[i][j+1] != -1:
                        candidate.append(edge[i][j+1])
                        
                    if edge[i][j] > min(candidate):
                        num_correction += 1
                        edge[i][j] = min(candidate)
    
    for i in trange(h*w):
        if num_correction == 0:
            break
            
        else:
            num_correction = 0
    
            for i in range(h-1,-1,-1):
                for j in range(w-1,-1,-1):
                    if edge[i][j] != -1:
                        candidate = [99999]
    
    
                        if i==end and j==end:
                            if edge[i-1][j] != -1:
                                candidate.append(edge[i-1][j])
                            if edge[i][j-1] != -1:
                                candidate.append(edge[i][j-1])
    
                            if edge[i][j] > min(candidate):
                                num_correction += 1
                                edge[i][j] = min(candidate)
    
    
                        elif i==end:
                            if edge[i-1][j] != -1:
                                candidate.append(edge[i-1][j])
                            if edge[i][j-1] != -1:
                                candidate.append(edge[i][j-1])
                            if edge[i][j+1] != -1:
                                candidate.append(edge[i][j+1])
    
                            if edge[i][j] > min(candidate):
                                num_correction += 1
                                edge[i][j] = min(candidate)
    
    
                        elif j==end:
                            if edge[i-1][j] != -1:
                                candidate.append(edge[i-1][j])
                            if edge[i][j-1] != -1:
                                candidate.append(edge[i][j-1])
                            if edge[i+1][j] != -1:
                                candidate.append(edge[i+1][j])
    
                            if edge[i][j] > min(candidate):
                                num_correction += 1
                                edge[i][j] = min(candidate)
    
                        else:
                            if edge[i-1][j] != -1:
                                candidate.append(edge[i-1][j])
                            if edge[i+1][j] != -1:
                                candidate.append(edge[i+1][j])
                            if edge[i][j-1] != -1:
                                candidate.append(edge[i][j-1])
                            if edge[i][j+1] != -1:
                                candidate.append(edge[i][j+1])
    
                            if edge[i][j] > min(candidate):
                                num_correction += 1
                                edge[i][j] = min(candidate)
    
    im_label = edge
    return im_label


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

def color_sides(image):
    # color sides
    h, w = image.shape[0], image.shape[1]
    for i in range(h):
        image[i, 0] = image[i, 1]
        image[i, -1] = image[i, -2]
        
    for j in range(1, w):
        image[0, j] = image[1, j]
        image[-1, j] = image[-2, j]

    return image

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
    # img_average = img_average.reshape(img.shape[0],img.shape[1],3)
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

def crystal_orientation_averaging(image):
    img_label, df = labeling(image)
    img_blacked_filled = color_smallgrains(image, df)
    
    df, rx_list, ry_list = get_centroids(df)   
    df_average, img_gen_average = average_grain(df)
    
    img_gen_average_randomfilled = randomly_fill_target_color_in_image(img_gen_average, (0,0,0))
    img_gen_average_randomfilled = skeletonize_fill_all(img_gen_average_randomfilled, 10)
    
    return img_gen_average_randomfilled