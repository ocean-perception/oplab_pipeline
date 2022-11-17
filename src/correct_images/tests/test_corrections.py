# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from correct_images import corrections
from correct_images.tools.numerical import mean_std, RunningMeanStd


class testCorrections(unittest.TestCase):
    def setUp(self):
        path_root = Path(__file__).resolve().parents[1]
        test_yaml_path = path_root / "tests" / "test.yaml"
        with test_yaml_path.open("r") as stream:
            params = yaml.safe_load(stream)

        self.image_bayer = (
            np.array(params["Test_images"]["bayer"]["image_1"]).astype(np.float32)
            / 255.0
        )
        self.bayer_pattern_choices = np.array(params["Test_images"]["bayer"]["pattern"])

        self.bw_images = []
        self.rgb_images = []
        self.distance_matrices = []

        self.bw_images.append(
            np.array(params["Test_images"]["Black_White"]["image_1"]).astype(np.float32)
            / 255.0
        )
        self.bw_images.append(
            np.array(params["Test_images"]["Black_White"]["image_2"]).astype(np.float32)
            / 255.0
        )
        self.bw_images.append(
            np.array(params["Test_images"]["Black_White"]["image_3"]).astype(np.float32)
            / 255.0
        )

        self.rgb_images.append(
            np.array(params["Test_images"]["RGB"]["image_1"]).astype(np.float32) / 255.0
        )
        self.rgb_images.append(
            np.array(params["Test_images"]["RGB"]["image_2"]).astype(np.float32) / 255.0
        )
        self.rgb_images.append(
            np.array(params["Test_images"]["RGB"]["image_3"]).astype(np.float32) / 255.0
        )

        self.distance_matrices.append(np.array(params["Distance"]["distance_1"]))
        self.distance_matrices.append(np.array(params["Distance"]["distance_2"]))
        self.distance_matrices.append(np.array(params["Distance"]["distance_3"]))

    def test_attenuation(self):
        image_height, image_width, image_channels = self.rgb_images[0].shape
        attenuation_parameters = np.empty(
            (image_channels, image_height, image_width, 3)
        )
        correction_gains = np.empty((image_channels, image_height, image_width))

        images = np.array(self.rgb_images).reshape(
            [len(self.rgb_images), image_height * image_width, image_channels]
        )

        distances = np.array(self.distance_matrices).reshape(
            [len(self.distance_matrices), image_height * image_width]
        )

        test_directory = Path(tempfile.gettempdir())
        attenuation_parameters = corrections.calculate_attenuation_parameters(
            images, distances, image_height, image_width, image_channels, test_directory
        )

        print(attenuation_parameters.shape)

        target_altitude = 2.0
        correction_gains = corrections.calculate_correction_gains(
            target_altitude,
            attenuation_parameters,
            image_height,
            image_width,
            image_channels,
        )

        corrected_rgb = np.empty(
            (image_height, image_width, image_channels, len(self.rgb_images))
        )
        for k in range(len(self.rgb_images)):
            img = self.rgb_images[k]
            dist = self.distance_matrices[k]
            corrected = img.copy()
            corrected = corrections.attenuation_correct(
                img,
                dist,
                attenuation_parameters,
                correction_gains,
            )
            corrected_rgb[:, :, :, k] = corrected
        # TODO what do we test here?

    def test_debayer(self):
        # test debayer for each bayer pattern choices:
        for i in range(len(self.bayer_pattern_choices)):
            bayer_pattern = self.bayer_pattern_choices[i].get("choice")
            image_rgb = corrections.debayer(self.image_bayer, bayer_pattern)

            if bayer_pattern == "grbg":
                self.assertAlmostEqual(
                    image_rgb[1, 1, 0],
                    75 / 255.0,
                    2,
                    "Red channel value is incorrect",
                )
                self.assertAlmostEqual(
                    image_rgb[1, 1, 1],
                    60 / 255.0,
                    2,
                    "Green channel value is incorrect",
                )
                self.assertAlmostEqual(
                    image_rgb[1, 1, 2],
                    120 / 255.0,
                    2,
                    "Blue channel value is incorrect",
                )

            elif bayer_pattern == "rggb":
                self.assertAlmostEqual(
                    image_rgb[1, 1, 0],
                    150 / 255.0,
                    2,
                    "Red channel value is incorrect",
                )
                self.assertAlmostEqual(
                    image_rgb[1, 1, 1],
                    120 / 255.0,
                    2,
                    "Green channel value is incorrect",
                )
                self.assertAlmostEqual(
                    image_rgb[1, 1, 2],
                    60 / 255.0,
                    2,
                    "Blue channel value is incorrect",
                )

    def test_gamma(self):
        pass

    def test_manual_balance(self):
        pass

    def test_pixel_stat(self):
        image_height, image_width, image_channels = self.rgb_images[0].shape

        image_means = np.empty(
            (image_height, image_width, image_channels), dtype=np.float32
        )
        image_stds = np.empty(
            (image_height, image_width, image_channels), dtype=np.float32
        )
        for i in range(image_channels):
            img = np.array(self.rgb_images)[:, :, :, i] / 255.0
            image_means[:, :, i], image_stds[:, :, i] = mean_std(img)

        target_mean = 30
        target_std = 5

        corrected_imgs = np.empty(
            (len(self.rgb_images), image_height, image_width, image_channels),
            dtype=np.float32,
        )
        for k in range(len(self.rgb_images)):
            for i in range(image_channels):
                img = self.rgb_images[k][:, :, i] / 255.0
                corrected_imgs[k, :, :, i] = (
                    corrections.pixel_stat(
                        img,
                        image_means[:, :, i],
                        image_stds[:, :, i],
                        target_mean,
                        target_std,
                    )
                    * 255.0
                )

        final_image_means = np.empty(
            (image_height, image_width, image_channels), dtype=np.float32
        )
        final_image_stds = np.empty(
            (image_height, image_width, image_channels), dtype=np.float32
        )
        for i in range(image_channels):
            img = corrected_imgs[:, :, :, i]
            final_image_means[:, :, i], final_image_stds[:, :, i] = mean_std(img)

        final_image_means = np.around(final_image_means)
        final_image_stds = np.around(final_image_stds)

        targeted_mean = np.ones(
            (image_height, image_width, image_channels), dtype=np.uint8
        ) * int(target_mean / 100.0 * 255.0)
        targeted_std = np.ones(
            (image_height, image_width, image_channels), dtype=np.uint8
        ) * int(target_std / 100.0 * 255.0)

        print(final_image_means)
        print(final_image_stds)

        np.testing.assert_allclose(final_image_means, targeted_mean, atol=1)
        np.testing.assert_allclose(final_image_stds, targeted_std, atol=1)

    def test_rescale(self):
        image_size = 1000
        fake_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        target_pixel_size_m = 0.03
        altitude = 1.0
        f_x = 1000.0
        f_y = f_x
        current_pixel_size = altitude / f_x
        scale = current_pixel_size / target_pixel_size_m
        expected_image_size = int(image_size * scale)
        maintain_pixels = False
        interpolate_methods = [
            "bicubic",
            "bilinear",
            "nearest_neighbour",
            "lanczos",
        ]
        for interpolate_method in interpolate_methods:
            img_scaled = corrections.rescale(
                fake_img,
                interpolate_method,
                target_pixel_size_m,
                altitude,
                f_x,
                f_y,
                maintain_pixels,
            )
            m, n, p = img_scaled.shape
            self.assertEqual(m, expected_image_size)
            self.assertEqual(n, expected_image_size)
            self.assertEqual(p, 3)

        maintain_pixels = True
        expected_image_size = image_size
        for interpolate_method in interpolate_methods:
            img_scaled = corrections.rescale(
                fake_img,
                interpolate_method,
                target_pixel_size_m,
                altitude,
                f_x,
                f_y,
                maintain_pixels,
            )
            m, n, p = img_scaled.shape
            print(m, n, p)
            self.assertEqual(m, expected_image_size)
            self.assertEqual(n, expected_image_size)
            self.assertEqual(p, 3)

    def test_running_mean_std(self):
        dimensions = (10, 10, 3)
        runner = RunningMeanStd(dimensions)
        all_images = []
        for i in range(100):
            img = (np.random.rand(*dimensions) * 25.0 + 127.0) / 255.0
            all_images.append(img)
            runner.compute(img)
        all_images = np.array(all_images)
        true_mean = np.mean(all_images, axis=0)
        true_std = np.std(all_images, axis=0)
        np.testing.assert_allclose(runner.mean, true_mean, atol=2)
        np.testing.assert_allclose(runner.std, true_std, atol=2)

    def test_running_mean_std_with_outliers(self):
        dimensions = (10, 10, 3)
        runner = RunningMeanStd(dimensions)
        all_images = []
        for i in range(100):
            img = np.random.rand(*dimensions) * 0.1 + 0.3
            if i % 10 == 0:
                img = np.random.rand(*dimensions) * 0.7 + 0.3
            else:
                all_images.append(img)
            runner.compute(img)
        all_images = np.array(all_images)
        true_mean = np.mean(all_images, axis=0)
        true_std = np.std(all_images, axis=0)
        # Note, allow a 10% error in the mean and std
        np.testing.assert_allclose(runner.mean, true_mean, rtol=0.1, atol=0.1)
        np.testing.assert_allclose(runner.std, true_std, rtol=0.1, atol=0.1)

    def test_undistort(self):
        # TODO
        pass
