# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import unittest
import yaml
import os
import numpy as np
from pathlib import Path
from correct_images.corrector import *
import pandas as pd
import tempfile


class testCaseCorrector(unittest.TestCase):
    def test_debayer(self):

        path_root = Path(__file__).resolve().parents[1]
        test_yaml_path = path_root / "tests" / "test.yaml"
        with test_yaml_path.open("r") as stream:
            params = yaml.safe_load(stream)

        image_bayer = np.array(params["Test_images"]["bayer"]["image_1"])
        bayer_pattern_choices = np.array(params["Test_images"]["bayer"]["pattern"])

        # instantiate corrector class
        corrector = Corrector(True)
        # test debayer for each bayer pattern choices:
        for i in range(len(bayer_pattern_choices)):
            bayer_pattern = bayer_pattern_choices[i].get("choice")
            image_rgb = corrector.debayer(image_bayer, bayer_pattern)
            print('-------------------')
            print(image_rgb)

            # TODO check the values
            if bayer_pattern == "grbg":
                self.assertEqual(
                    image_rgb[1, 1, 0], 240, "Blue channel value is incorrect"
                )
                self.assertEqual(
                    image_rgb[1, 1, 1], 120, "Green channel value is incorrect"
                )
                self.assertEqual(
                    image_rgb[1, 1, 2], 120, "Red channel value is incorrect"
                )

            elif bayer_pattern == "rggb":
                self.assertEqual(
                    image_rgb[1, 1, 0], 120, "Blue channel value is incorrect"
                )
                self.assertEqual(
                    image_rgb[1, 1, 1], 180, "Green channel value is incorrect"
                )
                self.assertEqual(
                    image_rgb[1, 1, 2], 240, "Red channel value is incorrect"
                )


if __name__ == "__main__":
    unittest.main()
