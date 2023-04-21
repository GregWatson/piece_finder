# Find the best match between 'piece' image and patches of the main image.

import sys
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

orb = None
bf = None

cpu_count = 1
MP = True
if MP:
    import multiprocessing as mp
    cpu_count = os.cpu_count()
    cpu_count = 1
    # print(f"found {cpu_count} CPUs.")

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
    for y in range(height):
        for x in range(width):
            img[y][x] = [img[y][x][2], img[y][x][1], img[y][x][0] ]

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
def find_piece_in_h_stripe(max_num_matches_seen, max_results, img_main, piece_des, y, patch_incr, patch_width, patch_height, main_x):
    ''' max_num_matches_seen is max number of descriptor matches seen so far for this process.
        max_results is max number of results to return from this function.
        Returns list of matches and updates max_num_matches_seen.
    '''
    global orb, bf
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
def find_piece_in_stripes(max_results, img_main, piece_des, y_list, patch_incr, patch_width, patch_height, main_x):
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
        (new_match_list, new_max_num_matches_seen, new_no_kp_counter) = find_piece_in_h_stripe(max_num_matches_seen, max_results, img_main, piece_des, y, patch_incr, patch_width, patch_height, main_x)
        if new_max_num_matches_seen > max_num_matches_seen:
            max_num_matches_seen = new_max_num_matches_seen
        no_kp_counter = no_kp_counter + new_no_kp_counter
        (max_num_matches_seen, match_list) = merge_matches(max_results, match_list, new_match_list)

    return((match_list, max_num_matches_seen, no_kp_counter))


def mp_test (x, output):
    output.put("This is proc " + str(x))

def mp_find_piece_in_stripes(x, output, max_results, img_main, piece_des, y_list, patch_incr, patch_width, patch_height, main_x):
    (match_list, max_num_matches_seen, no_kp_counter) = find_piece_in_stripes(max_results, img_main, piece_des, y_list, patch_incr, patch_width, patch_height, main_x)

    output.put((x, match_list, max_num_matches_seen, no_kp_counter))



#------------------------------------------------------------------
if __name__ == "__main__":

    max_results = 4  # maximum number of matching patches to display
    red = (200,0,0)
    grey100 = (100,100,100)

    #main_name = 'hedgehog_cover.jpg'
    #piece_name = 'hedgehog_pc1.jpg'
    main_name = 'hello.bmp'
    piece_name = 'hello_pc1.bmp'

    print(f"Loading main image {main_name} and piece image {piece_name}")

    path_main = os.path.normpath(os.path.join(os.getcwd(), '..', 'resources', main_name))
    path_piece = os.path.normpath(os.path.join(os.getcwd(), '..', 'resources', piece_name))

    img_main = cv.imread(path_main)   
    assert img_main is not None, f"{path_main} file could not be read, check with os.path.exists()"
    img_piece = cv.imread(path_piece)
    assert img_piece is not None, f"{path_piece} file could not be read, check with os.path.exists()"

    print(f"Reversing BGR color to be RGB...")
    reverse_img_color(img_main)
    reverse_img_color(img_piece)

    for rot in range(0):
        print("rotating piece 90 degrees")
        img_piece = cv.rotate(img_piece, cv.ROTATE_90_CLOCKWISE)

    # make a copy of orig image - we will 'draw' on this new one.
    img = img_main.copy()

    # get image sizes
    main_x, main_y = (img_main.shape[1], img_main.shape[0])
    piece_x, piece_y = (img_piece.shape[1], img_piece.shape[0])

    assert ((main_x >= piece_x) and (main_y >= piece_y)), f"sub image (piece) must not be smaller than main image in either dimension"

    # Find an orb matcher that works well on the piece image
    orb = get_orb_detector(img_piece)
    assert orb is not None, f"Unable to locate a suitable key point matcher for piece '{path_piece}'"

    # Get key points and descriptors for piece
    piece_kp, piece_des = orb.detectAndCompute(img_piece, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    patch_incr = max(int(piece_x/10), 5) # num pixels between start points of each patch taken from main
    print(f"Using x,y increment value of {patch_incr} to scan across main image.")

    patch_width = min([piece_x + 2*patch_incr, piece_x])
    patch_height = min([piece_y + 2*patch_incr, piece_y])

    patch_counter = 0
    no_kp_counter = 0
    patch_y1 = 0
    patch_y2 = patch_height
    
    match_list = []
    max_num_matches_seen = 0

    if MP:
        output = mp.Queue()

    # create a list of stripes for each CPU (process)
    y_list = [[] for i in range(cpu_count)]
    proc_id = 0
    while patch_y2 <= main_y:
        y_list[proc_id].append(patch_y1)
        proc_id = (proc_id + 1) % cpu_count
        patch_y1 = patch_y1 + patch_incr
        patch_y2 = patch_y1 + patch_height - 1

    #-------------------------------------------
    # MP compute

    if MP:
        processes = [mp.Process(target=mp_find_piece_in_stripes, 
                                args=(x, output, max_results, img_main, piece_des, y_list[x], patch_incr, patch_width, patch_height, main_x)) for x in range(cpu_count)]
        #processes = [mp.Process(target=mp_test, 
        #                        args=(x, output)) for x in range(cpu_count)]

        for p in processes:
            p.start()

        result = []
        for ii in range(cpu_count):
            (x, match_list, max_num_matches_seen, no_kp_counter) = output.get(True)
            print(f"{x} saw {len(match_list)} matches")

        for p in processes:
            p.join()

        # print(f"Output is:{result}")

    #-------------------------------------------
    # Single process compute

    else:
        (match_list, max_num_matches_seen, no_kp_counter) = find_piece_in_stripes(max_results, img_main, piece_des, y_list, patch_incr, patch_width, patch_height, main_x)



    if len(match_list) > 0:
        print(f"Maximum number of kp matches in one patch was {max_num_matches_seen}.")
        for match in match_list:
            draw_rect(img, match.x1, match.y1, match.x2, match.y2, grey100, mode=1)

    print(f"Processed {patch_counter} patches.")
    if (no_kp_counter > 0) : print(f"Saw {no_kp_counter} patches with no key points.")

    plt.imshow(img)
    plt.show()