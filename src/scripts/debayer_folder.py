import argparse
import contextlib
from pathlib import Path

import cv2
import joblib
import numpy as np
from tqdm import tqdm


# Joblib and tqdm solution to progressbars
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# convert bayer image to RGB based
# on the bayer pattern for the camera
def debayer(image_fn, pattern: str, output_folder: Path) -> np.ndarray:
    """Perform debayering of input image

    Parameters
    -----------
    image : numpy.ndarray
        image data to be debayered
    pattern : string
        bayer pattern

    Returns
    -------
    numpy.ndarray
        Debayered image
    """

    image = cv2.imread(str(image_fn), -1)
    fname_stem = image_fn.stem

    corrected_rgb_img = None
    if pattern == "rggb" or pattern == "RGGB":
        corrected_rgb_img = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR_EA)
    elif pattern == "grbg" or pattern == "GRBG":
        corrected_rgb_img = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2BGR_EA)
    elif pattern == "bggr" or pattern == "BGGR":
        corrected_rgb_img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR_EA)
    elif pattern == "gbrg" or pattern == "GBRG":
        corrected_rgb_img = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR_EA)
    elif pattern == "mono" or pattern == "MONO":
        return image
    else:
        print("Bayer pattern not supported (", pattern, ")")
    fname = output_folder / (fname_stem + ".png")
    cv2.imwrite(str(fname), corrected_rgb_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to images.")
    parser.add_argument("extension", help="extension of images (e.g. jpg, png)")
    parser.add_argument(
        "pattern", help="Bayer pattern (e.g. RGGB, GRBG, BGGR, GBRG, MONO)"
    )
    parser.add_argument("output_folder", help="Output folder to write processed images")
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    image_folder = Path(args.path)
    image_list = [p for p in image_folder.glob("*." + args.extension)]
    print("Found", len(image_list), "images")

    with tqdm_joblib(tqdm(desc="Debayer", total=len(image_list))) as progress_bar:
        joblib.Parallel(n_jobs=-2, verbose=0)(
            joblib.delayed(debayer)(image_list[idx], args.pattern, output_folder)
            for idx in range(len(image_list))
        )
