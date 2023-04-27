# Find the best match between 'piece' image and patches of the main image.

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Match():
    def __init__(self, num_matches=0, x1=0, y1=0, x2=0, y2=0):
        self.num_matches = num_matches
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

# Try to find an orb detector that yields a good number of key points but not too many.
# return Orb detector object else None
def get_orb_detector(img, nfeatures=100):
    img_x, img_y = (img.shape[1], img.shape[0])

    patchSize = min([img_x,img_y,31])

    for edgeThreshold in range(2,100,1):
        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=nfeatures,
                            edgeThreshold=edgeThreshold,
                            patchSize=patchSize)

        # find the list of keypoints with this Orb configuration
        kp = orb.detect(img,None)
        # print(f"edgeThresh {edgeThreshold} yielded {len(kp)} key points")
        if (len(kp) < nfeatures):
            return orb

    return None

# Add colors and wrap at wrap point
def add_color(col1, col2, wrap=256):
    ret_col = []
    for i,c in enumerate(col1):
        ret_col.append((c + col2[i]) % wrap)
    return ret_col

# Draw rect in image. Color can be absolute or additive.
# NOTE: img is matrix [row][column][color] i.e. [y][x][color]
# top left is (x1,y1), bottom right is (x2,y2)
# color is [r,g,b]
# mode 0 = absolute, 1 = additive (wraps beyond 255)
def draw_rect(img, x1,y1,x2,y2,color, mode=0):
    assert ((x2>=x1) and (y2>=y1)), "draw_rect: requires (x2>=x1) and (y2>=y1)"
    img_x, img_y = (img.shape[1], img.shape[0])
    # clip to img
    first_x = max([x1,0])
    last_x  = min([x2, img_x-1])
    first_y = max([y1,0])
    last_y  = min([y2, img_y-1])

    # draw lines
    for x in range(first_x, last_x+1):
        if (mode==0):
            img[first_y][x] = color
            img[last_y][x]  = color
        else:
            img[first_y][x] = add_color(img[first_y][x], color)
            img[last_y][x]  = add_color(img[last_y][x], color)

    for y in range(first_y, last_y+1):
        if (mode==0):
            img[y][first_x] = color
            img[y][last_x]  = color
        else:
            img[y][first_x] = add_color(img[y][first_x], color)
            img[y][last_x]  = add_color(img[y][last_x], color)


# openCV stores color us a numpy array in reverse order (i.e. b,g,r) so this simply reverses it.
def reverse_img_color(img):
    width, height = (img.shape[1], img.shape[0])
    t = 0
    for y in range(height):
        for x in range(width):
            tmp = img[y][x][2]
            img[y][x][2] = img[y][x][0]
            img[y][x][0] = tmp

# Create a copy of a sub image
def get_sub_image(img, x1, y1, x2, y2):
    sub_img = img[y1:y2+1, x1:x2+1].copy()
    return sub_img

# Merge two lists of matches. Final list has no more than max_results elements.
def merge_matches(max_results, match_list, new_match_list):
    L = match_list
    max_num_matches_seen = 0
    for m in match_list:
        if m.num_matches > max_num_matches_seen:
            max_num_matches_seen = m.num_matches

    for m in new_match_list:
        if  m.num_matches >= max_num_matches_seen:         
            max_num_matches_seen = m.num_matches   
            if len(L) >= max_results:
                L.pop(0)
            L.append(m)

    return ((max_num_matches_seen, L))

# Match given piece against a horizontal stripe of an image.
def find_piece_in_h_stripe(max_num_matches_seen, max_results, img_main, piece_des, y, patch_incr, patch_width, patch_height, main_x, orb, bf):
    ''' max_num_matches_seen is max number of descriptor matches seen so far for this process.
        max_results is max number of results to return from this function.
        Returns list of matches and updates max_num_matches_seen.
    '''
    match_list = [] # returned list of best matches
    patch_counter = 0 # count of number patches processed
    no_kp_counter = 0 # num patches with no detectable keypoints.

    patch_x1 = 0
    patch_x2 = patch_width - 1
    patch_y1 = y
    patch_y2 = y + patch_height - 1

    while patch_x2 <= main_x:
        patch_counter = patch_counter + 1
        patch = get_sub_image(img_main, patch_x1, patch_y1, patch_x2, patch_y2)

        patch_kp, patch_des = orb.detectAndCompute(patch,None)
        if len(patch_kp) > 0 :
            # Match descriptors.
            matches = bf.match(piece_des, patch_des)
            num_matches = len(matches)
            if (num_matches>= max_num_matches_seen): 
                if (num_matches > max_num_matches_seen):
                    print(f"Saw {num_matches} descriptor matches")
                max_num_matches_seen = num_matches   
                if len(match_list) >= max_results:
                    match_list.pop(0)
                match_list.append(Match(num_matches, patch_x1, patch_y1, patch_x2, patch_y2))
        else:
            no_kp_counter = no_kp_counter + 1
        patch_x1 = patch_x1 + patch_incr
        patch_x2 = patch_x1 + patch_width

    return((match_list, max_num_matches_seen, no_kp_counter))

# Match given piece against a list of horizontal stripes of an image.
# And then merge the matches, keeping only the best.
def find_piece_in_stripes(max_results, img_main, piece_des, y_list, patch_incr, patch_width, patch_height, main_x, orb, bf):
    ''' max_results is max number of results to return from this function.
        Returns list of matches and updates max_num_matches_seen.
        Also returns number of patches that had no keypoints
    '''
    match_list = [] # returned list of best matches
    patch_counter = 0 # count of number patches processed
    no_kp_counter = 0 # num patches with no detectable keypoints.
    max_num_matches_seen = 0 # max matches so far from a single patch

    print(f"process find_piece_in_stripes() will process {len(y_list)} horizontal stripes.")
    for y in y_list:
        (new_match_list, new_max_num_matches_seen, new_no_kp_counter) = find_piece_in_h_stripe(max_num_matches_seen, max_results, img_main, piece_des, y, patch_incr, patch_width, patch_height, main_x, orb, bf)
        if new_max_num_matches_seen > max_num_matches_seen:
            max_num_matches_seen = new_max_num_matches_seen
        no_kp_counter = no_kp_counter + new_no_kp_counter
        (max_num_matches_seen, match_list) = merge_matches(max_results, match_list, new_match_list)

    return((match_list, max_num_matches_seen, no_kp_counter))

