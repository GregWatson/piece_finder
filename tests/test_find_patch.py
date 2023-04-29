import os
import unittest
import sys
sys.path.append('D:\\Users\\gwatson\\Development\\Python\\piece_finder\\piece_libs')
from utils import *
import time
import random

class TestFindPatch(unittest.TestCase):

    def test_hedgehog_cover(self):

        main_name = 'hedgehog_cover.jpg'
        num_pieces = 1000
        max_results = 4

        js = Jigsaw(name='hedgehog_cover', num_pieces=1000)
        js.load_jigsaw_from_file(main_name)

        js.print()


        # patch_incr is how many pixels between each patch that we look at.
        patch_incr = max(round(js.piece_x/10), 5) # num pixels between start points of each patch taken from main
        print(f"Using x,y pixel increment value of {patch_incr} to scan across main image.")

        patch_width = js.piece_x + 2*patch_incr
        patch_height = js.piece_y + 2*patch_incr

        # choose random coordinates for a piece (piece is smaller than a patch).
        # This is the piece we will try to find.
        x0 = random.randint(0, js.x - js.piece_x)
        y0 = random.randint(0, js.y - js.piece_y)

        # !!!! Need to work on this piece.
        x0 = 1296
        y0 = 740


        x1 = x0 + js.piece_x - 1
        y1 = y0 + js.piece_y - 1

        print(f"Find patch randomly selected with top left at ({x0},{y0}), bottom right at ({x1},{y1}).")

        piece_img = get_sub_image(js.img, x0,y0, x1,y1)

        # Find an orb matches that works OK on this piece
        orb = get_orb_detector(piece_img)
        assert orb is not None, f"Unable to locate a suitable key point matcher (orb detector) for image '{js.name}'"

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        assert bf is not None, f"Unable to create a BF Matcher object"

        # for rot in range(0):
        #     print("rotating piece 90 degrees")
        #     img_piece = cv.rotate(img_piece, cv.ROTATE_90_CLOCKWISE)


        # Get key points and descriptors for piece
        piece_kp, piece_des = orb.detectAndCompute(piece_img, None)


        patch_counter = 0
        no_kp_counter = 0
        patch_y1 = 0
        patch_y2 = patch_height-1

        match_list = []
        max_num_matches_seen = 0

        # create a list of stripes.
        y_list = []
        while patch_y2 <= js.y:
            y_list.append(patch_y1)
            patch_y1 = patch_y1 + patch_incr
            patch_y2 = patch_y1 + patch_height - 1

        (match_list, max_num_matches_seen, no_kp_counter) = find_piece_in_stripes(max_results, js.img, piece_des, y_list, patch_incr, patch_width, patch_height, js.x, orb, bf)

        if len(match_list) > 0:
            print(f"Maximum number of kp matches in one patch was {max_num_matches_seen}.")
            for match in match_list:
                print(f"Saw {match.num_matches} matches at ({match.x1}, {match.y1}), ({match.x2}, {match.y2})")

        print(f"Processed {patch_counter} patches.")
        if (no_kp_counter > 0) : print(f"Saw {no_kp_counter} patches with no key points.")

        self.assertEqual(True,False) # Just until we get this all cleaned up and matching.


if __name__ == '__main__':
    unittest.main()