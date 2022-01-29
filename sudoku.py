import os
import queue
from queue import Queue
import cv2 as cv
import numpy as np
import math


def area_length(coord_x, coord_y):
    return math.sqrt((coord_x[0] - coord_y[0])**2 + (coord_x[1] - coord_y[1])**2)
def cut_sudoku(img, corners):
    edge_points = np.array(corners, dtype='float32')
    edge_points = np.float32(reorder(edge_points))
    area = max([
        area_length(corners[2], corners[1]),
        area_length(corners[0], corners[3]),
        area_length(corners[2], corners[3]),
        area_length(corners[0], corners[1])
    ])
    final_edge_points = np.array([[0, 0], [area - 1, 0], [0, area - 1], [area - 1, area - 1]], dtype="float32")
    matrix = cv.getPerspectiveTransform(edge_points, final_edge_points)
    return cv.warpPerspective(img, matrix, (int(area), int(area)))
def clear_patches(img, lines_horizontal, lines_vertical):
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    len_vertical = len(lines_vertical)
    digit_height, digit_width = img.shape[0] // 9, img.shape[1] // 9
    len_horizontal = len(lines_horizontal)
    for i in range(len_horizontal - 1):
        for j in range(len_vertical - 1):
            y_min = lines_vertical[j][0][0] + 5
            y_max = lines_vertical[j + 1][1][0] - 5
            x_min = lines_horizontal[i][0][1] + 5
            x_max = lines_horizontal[i + 1][1][1] - 5
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            mean_image_black = np.sum(patch == 0)
            if mean_pixels != 0:
                img[x_min:x_max, y_min:y_max] = 0
        if len(lines_vertical) == 9:
            len_vertical = 9
            y_min = lines_vertical[8][0][0] + 5
            y_max = lines_vertical[8][1][0] + digit_width - 5
            x_min = lines_horizontal[i][0][1] + 5
            x_max = lines_horizontal[i + 1][1][1] - 5
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            if mean_pixels != 0:
                img[x_min:x_max, y_min:y_max] = 0
        if len(lines_horizontal) == 9:
            for j in range(len(lines_vertical) - 1):
                y_min = lines_vertical[j][0][0] + 5
                y_max = lines_vertical[j + 1][1][0] - 5
                x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 5
                x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 5
                patch = img[x_min:x_max, y_min:y_max].copy()
                mean_pixels = np.mean(patch)
                if mean_pixels != 0:
                    img[x_min:x_max, y_min:y_max] = 0
            if len(lines_vertical) == 9:
                y_min = lines_vertical[len_vertical - 1][0][0] + 5
                y_max = lines_vertical[len_vertical - 1][1][0] + digit_width - 5
                x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 5
                x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 5
                patch = img[x_min:x_max, y_min:y_max].copy()
                mean_pixels = np.mean(patch)
                if mean_pixels != 0:
                    img[x_min:x_max, y_min:y_max] = 0
    thresh = cv.dilate(img, kernel, iterations = 5)
    return thresh
def color_filter(img_raw):
    img = cv.cvtColor(img_raw, cv.COLOR_BGR2HSV)
    red_index = np.array([30, 20, 1])
    red_last_index = np.array([255, 255, 255])
    grid = cv.inRange(img, red_index, red_last_index)
    if np.mean(grid == 0) - np.mean(grid == 255) >= 0.7:
        return 0
    return 1
def crop_images(task_number, path, predictions_filename_path, index):
    if index > 9 :
        img = cv.imread(path + str(index) +".jpg")
    else:
        img = cv.imread(path + "0" + str(index) + ".jpg")
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    result = preprocess_image(img, task_number)
    new_image = cut_sudoku(img, result)
    process_cropped_sudoku(new_image, path, predictions_filename_path, index, task_number)
def extract_vertical_lines(filtered_vertical):
    mask_vertical = filtered_vertical // 255
    mask_vertical=np.sum(mask_vertical , axis=0)
    lines=[]
    for i in range(0,len(mask_vertical)):
        if(mask_vertical[i]!=0):
            lines.append([(i, 0), (i, filtered_vertical.shape[0])])
    distinct_lines = []
    distinct_lines.append(lines[0])
    for line in lines:
        if line[0][0] - distinct_lines[-1][0][0] > 20:
            distinct_lines.append(line)
    correct_lines = distinct_lines[-10:]
    return correct_lines
def extract_horizontal_lines(filtered_horizontal):
    mask_horizontal = filtered_horizontal // 255
    mask_horizontal=np.sum(mask_horizontal , axis=1)
    lines=[]
    for i in range(0,len(mask_horizontal)):
        if(mask_horizontal[i]!=0):
            lines.append([(0, i), (filtered_horizontal.shape[1], i)])
    distinct_lines = []
    correct_lines = []
    if len(lines) != 0:
        distinct_lines.append(lines[0])
    for line in lines:
        if line[0][1] - distinct_lines[-1][0][1] > 20:
            distinct_lines.append(line)
    correct_lines = distinct_lines[-10:]
    return correct_lines
def filter(image):
    kernel_vertical = np.array([100*[0],100*[1],100*[0]])
    kernel_vertical=np.transpose(kernel_vertical)
    kernel_vertical= kernel_vertical/ kernel_vertical.sum()
    filtered_vertical = 255 - cv.filter2D(255 - image, -1,  kernel_vertical)
    filtered_vertical[filtered_vertical < 255] = 0
    kernel_horizontal = np.array([100*[0],100*[1],100*[0]])
    kernel_horizontal=  kernel_horizontal/  kernel_horizontal.sum()
    filtered_horizontal = 255 - cv.filter2D(255 - image, -1,kernel_horizontal)
    filtered_horizontal[filtered_horizontal < 255] = 0
    return filtered_vertical,filtered_horizontal
def get_depth(img,mask, zone_label,lines_horizontal, lines_vertical, start_width,start_height):
    height, width = img.shape[0], img.shape[1]
    visited = np.zeros((height, width))
    digit_box_width = width // 9
    digit_box_height = height // 9
    next_visited = Queue()
    next_visited .put([start_height, start_width])
    counter = 0
    while next_visited .empty() == False:
        (coord_x, coord_y) = next_visited .get()
        i, j = coord_x, coord_y
        # If there is a white patch from the current zone then search its neighbors
        # Otherwise, we are outside the are
        if visited[i, j] == 0 and mask[i, j] == 0 and \
                np.mean(img[i:i+digit_box_height,j:j+digit_box_width] == 255) > np.mean(img[i:i+digit_box_height,j:j+digit_box_width] == 0):
            mask[i:i+digit_box_width, j:j+digit_box_height] = zone_label
            visited[i:i+digit_box_width, j:j+digit_box_height] = 1
            counter +=1
            neighbors = []
            if coord_x + 2*digit_box_height <= height:
                patch = img[coord_x + digit_box_height:coord_x + 2*digit_box_height, coord_y:coord_y+digit_box_width].copy()
                if np.mean(patch == 255) > np.mean(patch == 0):
                    neighbors.append([coord_x + digit_box_height, coord_y])
            if coord_x - digit_box_height >= 0:
                patch = img[coord_x - digit_box_height:coord_x , coord_y:coord_y + digit_box_width].copy()
                if np.mean(patch == 255) > np.mean(patch == 0):
                    neighbors.append([coord_x - digit_box_height, coord_y])
            if coord_y + 2*digit_box_width <= width :
                patch = img[coord_x:coord_x+digit_box_height, coord_y + digit_box_width:coord_y + 2*digit_box_width].copy()
                if np.mean(patch == 255) > np.mean(patch == 0):
                    neighbors.append([coord_x, coord_y + digit_box_width])
            if coord_y - digit_box_width >= 0:
                patch = img[coord_x:coord_x+digit_box_height, coord_y - digit_box_width:coord_y].copy()
                if np.mean(patch == 255) > np.mean(patch == 0):
                    neighbors.append([coord_x, coord_y - digit_box_width])
            for neighbor in neighbors:
                if visited[neighbor[0],neighbor[1]] == 0 and mask[neighbor[0],neighbor[1]] == 0:
                    next_visited .put(neighbor)
    if counter != 0:
        zone_label += 1
    return mask, zone_label
def get_patch(f, img, lines_horizontal, lines_vertical, ls, digit_width, digit_height):
    len_vertical = len(lines_vertical)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 15
            y_max = lines_vertical[j + 1][1][0] - 10
            x_min = lines_horizontal[i][0][1] + 15
            x_max = lines_horizontal[i + 1][1][1] - 10
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            if mean_pixels != 0:
                f.write("x")
            else:
                f.write("o")
        if len(lines_vertical) == 9:
            len_vertical = 9
            y_min = lines_vertical[8][0][0] + 15
            y_max = lines_vertical[8][1][0] + digit_width - 10
            x_min = lines_horizontal[i][0][1] + 15
            x_max = lines_horizontal[i + 1][1][1] - 10
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            if mean_pixels != 0:
                f.write("x")
            else:
                f.write("o")
        if i != len(lines_horizontal) - 2:
            f.write("\n")
    if len(lines_horizontal) == 9:
        f.write("\n")
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 15
            y_max = lines_vertical[j + 1][1][0] - 10
            x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 15
            x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 10
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            if mean_pixels != 0:
                f.write("x")
            else:
                f.write("o")
        if len(lines_vertical) == 9:
            y_min = lines_vertical[len_vertical - 1][0][0] + 15
            y_max = lines_vertical[len_vertical - 1][1][0]+ digit_width - 10
            x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 15
            x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 10
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            if mean_pixels != 0:
                f.write("x")
            else:
                f.write("o")
def get_gray_patch(f, img, lines_horizontal, lines_vertical):
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    img_copy = img.copy()
    img = clear_patches(img, lines_horizontal, lines_vertical)
    edges = cv.Canny(img, 255, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    areas = []
    for i in range(len(contours)):
        if len(contours[i]) > 3:
            areas.append(cv.contourArea(contours[i]))
    areas = np.array(areas)
    mean_area = np.mean(areas)
    contours = [component for component in contours if cv.contourArea(component) >= mean_area]
    single_contour = []
    all_contours = np.zeros(img.shape, np.uint8)
    for i in range(len(contours)):
       if cv.contourArea(contours[i]) >= mean_area:
           mask = np.zeros(img.shape, np.uint8)
           cv.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)
           cv.drawContours(all_contours, [contours[i]], 0, (255, 255, 255), -1)
           single_contour.append(mask)
    all_contours = cv.dilate(all_contours, kernel, iterations=20)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    new_image = cv.bitwise_not(all_contours)
    last_zones = cv.bitwise_not(all_contours)
    if len(single_contour) < 8:
        img = cv.erode(img, kernel, iterations = 3)
        new_image = cv.bitwise_not(img)
        new_image = cv.bitwise_and(last_zones, last_zones, mask = new_image)
        new_image = cv.dilate(new_image, kernel, iterations = 2)
        edges = cv.Canny(new_image, 255, 400)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if len(contours[i]) > 3:
                mask = np.zeros(img.shape, np.uint8)
                cv.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)
                single_contour.append(mask)
    single_contour.append(new_image)
    set_zones(f, img_copy, single_contour, lines_horizontal, lines_vertical)
def normalize_image(img):
    noise = cv.dilate(img, np.ones((7,7),np.uint8))
    blur = cv.medianBlur(img, 21)
    res = 255 - cv.absdiff(img, blur)
    no_shdw = cv.normalize(res,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    return no_shdw
def preprocess_image(img, task_number):
    if task_number == "task1":
        thresh = turn_binary(img)
    else:
        if color_filter(img) == 0:
            thresh = turn_binary(img)
        else:
            thresh = turn_binary_colored(img)
    edges = cv.Canny(thresh, 255, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left
    return [top_left, top_right, bottom_left, bottom_right]
def reorder(corners):
    corners = corners.reshape((4, 2))
    corners_mask = np.zeros((4, 1,2 ), dtype= np.int32)
    corners_mask[0] = corners[np.argmin(corners.sum(1))]
    corners_mask[3] = corners[np.argmax(corners.sum(1))]
    corners_mask[1] = corners[np.argmin(np.diff(corners, axis = 1))]
    corners_mask[2] = corners[np.argmax(np.diff(corners, axis = 1))]
    return corners_mask
def show_image(title,image):
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def turn_binary(img):
    image_raw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    image = normalize_image(image_raw)
    image_g_blur = cv.GaussianBlur(image, (5, 5), 0)
    image_sharpened = cv.addWeighted(image, 1.2, image_g_blur, 1 , 0)
    thresh = cv.adaptiveThreshold(image_sharpened, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                           cv.THRESH_BINARY_INV, 3, 2)

    thresh = cv.dilate(thresh, kernel, iterations = 1)
    return thresh
def turn_binary_colored(img):
    image_raw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    average = 50
    image_raw = np.where((255 - image_raw) < average, 255, image_raw + average)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    image = normalize_image(image_raw)
    image_g_blur = cv.GaussianBlur(image, (5, 5), 0)
    image_sharpened = cv.addWeighted(image, 1, image_g_blur, 0.5, 0)
    _, thresh = cv.threshold(image_sharpened, 200, 255, cv.THRESH_BINARY_INV)
    thresh = cv.dilate(thresh, kernel, iterations=1)
    return thresh
def get_colored_thresh(img_raw, color):
    img = cv.cvtColor(img_raw, cv.COLOR_BGR2HSV)
    if color == "gray":
        first_index = np.array([15, 15, 15])
        last_index = np.array([255, 255, 255])
    elif color == "green&red":
        first_index = np.array([0, 10, 50])
        last_index = np.array([70, 255, 255])
    elif color == "red":
        first_index = np.array([0, 10, 20])
        last_index = np.array([20, 255, 255])
    elif color == "green":
        first_index = np.array([20, 10, 0])
        last_index = np.array([70, 255, 255])
    elif color == "blue":
        first_index = np.array([60, 0, 20])
        last_index = np.array([200, 255, 255])
    grid = cv.inRange(img, first_index, last_index)
    return grid
def get_colored_patch(f,img,thresh_red,thresh_green,thresh_blue,lines_horizontal, lines_vertical):
    zone = 1
    height, width = img.shape[0], img.shape[1]
    mask = np.zeros((height, width))
    digit_width = width // 9
    digit_height = height // 9
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            patch_blue = thresh_blue[x_min:x_max, y_min:y_max].copy()
            patch_green = thresh_green[x_min:x_max, y_min:y_max].copy()
            patch_red = thresh_red[x_min:x_max, y_min:y_max].copy()
            mean_pixels_black_red = np.mean(patch_red == 0)
            mean_pixels_white_red = np.mean(patch_red == 255)
            mean_pixels_black_green = np.mean(patch_green == 0)
            mean_pixels_white_green = np.mean(patch_green == 255)
            mean_pixels_black_blue = np.mean(patch_blue == 0)
            mean_pixels_white_blue = np.mean(patch_blue == 255)
            #If we reached a new zone, then label each patch of it
            if mask[x_min:x_max, y_min:y_max].all() == 0:
                if mean_pixels_white_red > mean_pixels_black_red:
                    mask, zone = get_depth(thresh_red, mask, zone, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
                elif mean_pixels_white_blue > mean_pixels_black_blue:
                    mask, zone = get_depth(thresh_blue, mask, zone, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
                elif mean_pixels_white_green > mean_pixels_black_green:
                    mask, zone = get_depth(thresh_green, mask, zone, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min,x_max):
                for l in range(y_min,y_max):
                    if mask[k,l] != 0:
                        number_zone = int(mask[k,l])
                        break
            f.write(str(number_zone) + empty)
        if len(lines_vertical) == 9:
            len_vertical = 9
            y_min = lines_vertical[8][0][0] + 20
            y_max = lines_vertical[8][1][0] + digit_width - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            if mask[x_min:x_max, y_min:y_max].all() == 0:
                if mean_pixels_white_red > mean_pixels_black_red:
                    mask, zone = get_depth(thresh_red, mask, zone, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
                elif mean_pixels_white_blue > mean_pixels_black_blue:
                    mask, zone = get_depth(thresh_blue, mask, zone, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
                elif mean_pixels_white_green > mean_pixels_black_green:
                    mask, zone = get_depth(thresh_green, mask, zone, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min, x_max):
                for l in range(y_min, y_max):
                    if mask[k, l] != 0:
                        number_zone = int(mask[k, l])
                        break
            f.write(str(number_zone) + empty)
        if i != len(lines_horizontal) - 2:
            f.write("\n")
    if len(lines_horizontal) == 9:
        f.write("\n")
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 20
            x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min, x_max):
                for l in range(y_min, y_max):
                    if mask[k, l] != 0:
                        number_zone = int(mask[k, l])
                        break
            f.write(str(number_zone) + empty)
        if len(lines_vertical) == 9:
            y_min = lines_vertical[len_vertical - 1][0][0] + 20
            y_max = lines_vertical[len_vertical - 1][1][0] + digit_width - 20
            x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 20
            x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min, x_max):
                for l in range(y_min, y_max):
                    if mask[k, l] != 0:
                        number_zone = int(mask[k, l])
                        break
            f.write(str(number_zone) + empty)
def process_cropped_sudoku(img, path, predictions_filename_path, index, task_number):
    f = open(predictions_filename_path + str(index) + "_predicted.txt", "w+")
    if task_number == "task1" :
        thresh = turn_binary(img)
        lines_vertical = []
        width = img.shape[1]
        height = img.shape[0]
        digit_box_width = width // 9
        digit_box_height = height // 9
        k = index
        for i in range(0, img.shape[1], digit_box_width):
            l = []
            l.append((i, 0))
            l.append((i, img.shape[1] - 1))
            lines_vertical.append(l)
        lines_horizontal = []
        for i in range(0, img.shape[0], digit_box_height):
            l = []
            l.append((0, i))
            l.append((img.shape[0] - 1, i))
            lines_horizontal.append(l)
        get_patch(f, thresh, lines_horizontal, lines_vertical, k, digit_box_width, digit_box_height)
    else :
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        if color_filter(img) == 0:
            thresh = get_colored_thresh(img, "gray")
            thresh = cv.erode(thresh, kernel, iterations = 1)
            thresh = cv.dilate(thresh, kernel, iterations = 3)
        else:
            thresh = turn_binary_colored(img)
            thresh_blue = get_colored_thresh(img, "blue")
            thresh_green = get_colored_thresh(img, "green")
            thresh_red = get_colored_thresh(img, "red")
        filtered_vertical, filtered_horizontal = filter(thresh)
        correct_vertical_lines = extract_vertical_lines(filtered_vertical)
        correct_horizontal_lines = extract_horizontal_lines(filtered_horizontal)
        lines_vertical = []
        width = img.shape[1]
        height = img.shape[0]
        digit_box_width = width // 9
        digit_box_height = height // 9
        for i in range(0, img.shape[1], digit_box_width):
            l = []
            l.append((i, 0))
            l.append((i, img.shape[1] - 1))
            lines_vertical.append(l)
        lines_horizontal = []
        for i in range(0, img.shape[0], digit_box_height):
            l = []
            l.append((0, i))
            l.append((img.shape[0] - 1, i))
            lines_horizontal.append(l)
        if color_filter(img) == 0:
            get_gray_patch(f, thresh, lines_horizontal, lines_vertical)
        else:
            get_colored_patch(f, thresh, thresh_red, thresh_green, thresh_blue, lines_horizontal, lines_vertical)
    f.close()
def set_zones(f,img,zones_list,lines_horizontal, lines_vertical):
    height, width = img.shape[0], img.shape[1]
    digit_width = width // 9
    digit_height = height // 9
    mask = np.zeros((height, width))
    zone_label = 1
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            if mask[x_min:x_max, y_min:y_max].all() == 0:
                for zone in zones_list:
                    mean_pixels_black = np.mean(zone[x_min:x_max, y_min:y_max].copy() == 0)
                    mean_pixels_white = np.mean(zone[x_min:x_max, y_min:y_max].copy() == 255)
                    if mean_pixels_white > mean_pixels_black:
                        mask, zone_label = get_depth(zone, mask, zone_label, lines_horizontal, lines_vertical,y_min - 20, x_min - 20)
                        break
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min,x_max):
                for l in range(y_min,y_max):
                    if mask[k,l] != 0:
                        number_zone = int(mask[k,l])
                        break
            f.write(str(number_zone) + empty)
            #show_image("patch", patch)
        if len(lines_vertical) == 9:
            len_vertical = 9
            y_min = lines_vertical[8][0][0] + 20
            y_max = lines_vertical[8][1][0] + digit_width - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            if mask[x_min:x_max, y_min:y_max].all() == 0:
                for zone in zones_list:
                    mean_pixels_black = np.mean(zone[x_min:x_max, y_min:y_max].copy() == 0)
                    mean_pixels_white = np.mean(zone[x_min:x_max, y_min:y_max].copy() == 255)
                    if mean_pixels_white > mean_pixels_black:
                        mask, zone = get_depth(zone, mask, zone_label, lines_horizontal, lines_vertical, y_min - 20, x_min - 20)
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min, x_max):
                for l in range(y_min, y_max):
                    if mask[k, l] != 0:
                        number_zone = int(mask[k, l])
                        break
            f.write(str(number_zone) + empty)
        if i != len(lines_horizontal) - 2:
            f.write("\n")
    if len(lines_horizontal) == 9:
        f.write("\n")
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 20
            x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min, x_max):
                for l in range(y_min, y_max):
                    if mask[k, l] != 0:
                        number_zone = int(mask[k, l])
                        break
            f.write(str(number_zone) + empty)
        if len(lines_vertical) == 9:
            y_min = lines_vertical[len_vertical - 1][0][0] + 20
            y_max = lines_vertical[len_vertical - 1][1][0] + digit_width - 20
            x_min = lines_horizontal[len(lines_horizontal) - 1][0][1] + 20
            x_max = lines_horizontal[len(lines_horizontal) - 1][1][1] + digit_height - 20
            patch = img[x_min:x_max, y_min:y_max].copy()
            mean_pixels = np.mean(patch)
            empty = ""
            number_zone = 0
            if mean_pixels != 0:
                empty = "x"
            else:
                empty = "o"
            for k in range(x_min, x_max):
                for l in range(y_min, y_max):
                    if mask[k, l] != 0:
                        number_zone = int(mask[k, l])
                        break
            f.write(str(number_zone) + empty)
def sudoku(task_number, images_path, predictions_filename_path):
    if task_number == "task1":
        for i in range(1, 21):
            crop_images(task_number, images_path, predictions_filename_path, i)
    elif task_number == "task2":
        for i in range(1, 41):
            crop_images(task_number, images_path, predictions_filename_path, i)
    else:
        print("You entered an invalid task number.")

# Change this on your machine
"""
images_path = "D:\An III\ComputerVision\Laborator\Proiect1\\antrenare\jigsaw\\"
predictions_filename_path = "D:\An III\ComputerVision\Laborator\Proiect1\evaluare\\fisiere_solutie\Costrun_Larisa_343\jigsaw\\"
task_number = "task2"
"""
images_path = "D:\An III\ComputerVision\Laborator\Proiect1\\antrenare\clasic\\"
predictions_filename_path = "D:\An III\ComputerVision\Laborator\Proiect1\evaluare\\fisiere_solutie\Costrun_Larisa_343\clasic\\"
task_number = "task1"

sudoku(task_number, images_path, predictions_filename_path)