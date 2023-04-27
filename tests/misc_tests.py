import os
import unittest
import sys
sys.path.append('D:\\Users\\gwatson\\Development\\Python\\piece_finder\\piece_libs')
from utils import *
import time

class TestRGBtoBGRswap(unittest.TestCase):

    # Swap image from BGR to RGB
    def test_timeit(self):

        main_name = 'hedgehog_cover.jpg'
        #main_name = 'hello.bmp'

        print(f"Loading main image {main_name}")
        path_main = os.path.normpath(os.path.join(os.getcwd(), '..', 'resources', main_name))

        img_main = cv.imread(path_main)   
        assert img_main is not None, f"{path_main} file could not be read, check with os.path.exists()"
        print(f"Reversing BGR color to be RGB...")

        t0 = time.time()
        reverse_img_color(img_main)
        t1 = time.time()
        print(f"Time taken was {t1-t0}.")

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()