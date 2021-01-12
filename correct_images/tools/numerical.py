from typing import Tuple
import numpy as np
from tqdm import trange
from oplab import Console
import joblib
import datetime
from scipy import optimize
import math
from datetime import datetime
from numba import njit, prange
from ..loaders import default


class RunningMeanStd:
    __slots__ = ["_mean", "mean2", "_std", "count", "clipping_max"]
    def __init__(self, dimensions, clipping_max=250):
        self._mean = np.zeros(dimensions, dtype=np.float32)
        self.mean2 = np.zeros(dimensions, dtype=np.float32)
        self._std = np.zeros(dimensions, dtype=np.float32)
        self.count = 0
        self.clipping_max = clipping_max
    
    def compute(self, image):
        self.count += 1
        # clipping to mean if above threshold
        image[image > self.clipping_max] = self._mean[image > self.clipping_max]
        image = image.astype(np.float32)
        delta = image - self._mean
        self._mean += delta / self.count
        self.mean2 += delta * (image - self._mean)

    @property
    def mean(self):
        if self.count > 1:
            return self._mean
        else: 
            return None
    
    @property
    def std(self):
        if self.count > 1:
            self._std = np.sqrt(self.mean2 / self.count)
            self._std[self._std < 1] = 1.0
            return self._std
        else:
            return None


def running_mean_std(file_list, loader=default.loader):
    """Compute running mean and std of a list of image filenames

    Parameters
    ----------
    file_list : list
        List of image filenames. Can be list of str or list of Path
    loader : function
        Function to read one filename into a numpy array

    Returns
    -------
    (np.ndarray, np.ndarray)
        Mean and std arrays
    """
    count = 0
    tmp = loader(file_list[0])
    dimensions = tmp.shape

    mean = np.zeros(dimensions, dtype=np.float32)
    mean2 = np.zeros(dimensions, dtype=np.float32)
    std = np.zeros(dimensions, dtype=np.float32)

    for item in file_list:
        value = loader(item).astype(np.float32)
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        mean2 += delta * delta2
    if count > 1:
        std = np.sqrt(mean2 / count)
    return mean, std


def median_array(data: np.ndarray) -> np.ndarray:
    if len(data.shape) == 3:
        # Monochrome
        return median_array_impl(data)
    elif len(data.shape) == 4:
        # Colour
        [n, a, b, c] = data.shape
        median_array = np.zeros((a, b, c), dtype=np.float32)
        for c in range(data.shape[3]):
            median_array[:, :, c] = median_array_impl(data[:, :, :, c])
        return median_array
        
@njit
def median_array_impl(data: np.ndarray) -> np.ndarray:
    [n, a, b] = data.shape
    median_array = np.zeros((a, b), dtype=np.float32)
    for i in range(a):
        for j in range(b):
            lst = [0]*n
            for k in range(n):
                lst[k] = data[k, i, j]
            s = sorted(lst)
            median = 0
            if n % 2 == 0:
                for k in range(n//2-1, n//2+1):
                    median += s[k]
                median /= 2.0
            else:
                median = s[n//2]
            median_array[i, j] = median
    return median_array

@njit
def mean_array(data: np.ndarray) -> np.ndarray:
    [n, a, b] = data.shape
    mean_array = np.zeros((a, b), dtype=np.float32)
    for i in range(a):
        for j in range(b):
            mean = 0.
            for k in range(n):
                count = k + 1
                new_value = data[k, i, j]
                delta = new_value - mean
                mean += delta / count
            mean_array[i, j] = mean
    return mean_array


@njit
def mean_std_array(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    [n, a, b] = data.shape

    mean_array = np.zeros((a, b), dtype=np.float32)
    std_array = np.zeros((a, b), dtype=np.float32)

    # Welford's online algorithm
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for i in range(a):
        for j in range(b):
            mean = 0.
            mean_sq = 0.
            for k in range(n):
                count = k + 1
                new_value = data[k, i, j]
                delta = new_value - mean
                mean += delta / count
                delta2 = new_value - mean
                mean_sq += delta*delta2
            mean_array[i, j] = mean
            std_array[i, j] = math.sqrt(mean_sq/n)
    return mean_array, std_array


# calculate the mean and std of an image
def mean_std(data, calculate_std=True):
    """Compute mean and std for image intensities and distance values

    Parameters
    -----------
    data : numpy.ndarray
        data series
    calculate_std : bool
        denotes to compute std along with mean

    Returns
    --------
    numpy.ndarray
        mean and std of input image
    """

    mean_array, std_array = mean_std_array(data)
    if calculate_std:
        return mean_array, std_array
    else:
        return mean_array


def memory_efficient_std(data):
    ret_std = np.zeros(data.shape[1])
    BLOCKSIZE = 256
    for block_start in range(0, data.shape[1], BLOCKSIZE):
        block_data = data[:, block_start:block_start + BLOCKSIZE]
        ret_std[block_start:block_start + BLOCKSIZE] = np.std(block_data, dtype=np.float64, axis=0)
    return ret_std


def image_mean_std_trimmed(data, ratio_trimming=0.2, calculate_std=True):
    """Compute trimmed mean and std for image intensities using parallel computing

    Parameters
    -----------
    data : numpy.ndarray
        image intensities
    ratio_trimming : float
        trim ratio
    calculate_std : bool
        denotes to compute std along with mean

    Returns
    --------
    numpy.ndarray
        Trimmed mean and std
    """

    [n, a, b] = data.shape
    ret_mean = np.zeros((a, b), np.float32)
    ret_std = np.zeros((a, b), np.float32)

    effective_index = [list(range(0, n))]

    message = "calculating mean and std of images " + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if ratio_trimming <= 0:
        ret_mean, ret_std = mean_std_array(data)
        return ret_mean, ret_std

    else:

        for idx_a in trange(a, ascii=True, desc=message):
            results = [None]*b
            for idx_b in range(b):
                results[idx_b] = calc_mean_and_std_trimmed(
                    data[effective_index, idx_a, idx_b][0],
                    ratio_trimming,
                    calculate_std,
                )
            ret_mean[idx_a, :] = np.array(results)[:, 0]
            ret_std[idx_a, :] = np.array(results)[:, 1]
        if not calculate_std:
            return ret_mean
        else:
            return ret_mean, ret_std


def calc_mean_and_std_trimmed(data, rate_trimming, calc_std=True):
    """Compute trimmed mean and std for image intensities

    Parameters
    -----------
    data : numpy.ndarray
        image intensities
    rate_trimming : float
        trim ratio
    calc_std : bool
        denotes to compute std along with mean

    Returns
    --------
    numpy.ndarray
    """

    mean = None
    std = None
    if rate_trimming <= 0:
        mean, std = mean_std_array(data)
    else:
        sorted_values = np.sort(data)
        idx_left_limit = int(len(data) * rate_trimming / 2.0)
        idx_right_limit = int(len(data) * (1.0 - rate_trimming / 2.0))
        mean, std = mean_std_array(sorted_values[idx_left_limit:idx_right_limit])
    return np.array([mean, std])





