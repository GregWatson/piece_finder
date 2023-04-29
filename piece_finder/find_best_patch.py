# Find the best match between 'piece' image and patches of the main image.

import os
import sys
sys.path.append('D:\\Users\\gwatson\\Development\\Python\\piece_finder\\piece_libs')
from utils import *

max_results = 4  # maximum number of matching patches to display
red = (200,0,0)
grey100 = (100,100,100)

main_name = 'hedgehog_cover.jpg'
piece_name = 'hedgehog_pc1.jpg'
#main_name = 'hello.bmp'
#piece_name = 'hello_pc1.bmp'

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
patch_y2 = patch_height-1

match_list = []
max_num_matches_seen = 0

# create a list of stripes.
y_list = []
while patch_y2 <= main_y:
    y_list.append(patch_y1)
    patch_y1 = patch_y1 + patch_incr
    patch_y2 = patch_y1 + patch_height - 1

(match_list, max_num_matches_seen, no_kp_counter) = find_piece_in_stripes(max_results, img_main, piece_des, y_list, patch_incr, patch_width, patch_height, main_x, orb, bf)

if len(match_list) > 0:
    print(f"Maximum number of kp matches in one patch was {max_num_matches_seen}.")
    for match in match_list:
        draw_rect(img, match.x1, match.y1, match.x2, match.y2, grey100, mode=1)

print(f"Processed {patch_counter} patches.")
if (no_kp_counter > 0) : print(f"Saw {no_kp_counter} patches with no key points.")

plt.imshow(img)
plt.show()