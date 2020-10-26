import numpy as np
from tqdm import trange
from oplab import Console
import joblib
import datetime
from scipy import optimize
import math
from datetime import datetime
from numba import jit, njit


@njit
def mean_std_array(data: np.ndarray) -> (np.ndarray, np.ndarray):
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

    """
    [n, a, b] = data.shape

    #Console.info(f"{type(data)} of shape {data.shape}")
    data = data.reshape((n, a * b))
    #Console.info(f"{type(data)} of shape {data.shape}")
    print("memory allocated...")

    ret_mean = data.mean(dtype=np.float64, axis=0)
    print("mean calculated...")
    if calculate_std:
        ret_std = memory_efficient_std(data)  #data.std(axis=0)
        print("std calculated...")
        return ret_mean.reshape((a, b)), ret_std.reshape((a, b))
    else:
        return ret_mean.reshape((a, b))
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
        ret_mean = np.mean(data, axis=0)
        ret_std = np.std(data, axis=0)
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
            """
            results = joblib.Parallel(n_jobs=-2, verbose=3, max_nbytes=1e6)(
                [
                    joblib.delayed(calc_mean_and_std_trimmed)(
                        data[effective_index, idx_a, idx_b][0],
                        ratio_trimming,
                        calculate_std,
                    )
                    for idx_b in trange(b)
                ]
            )
            """
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
        """
        mean = np.mean(data)
        if calc_std:
            std = np.std(data)
        """
    else:
        sorted_values = np.sort(data)
        idx_left_limit = int(len(data) * rate_trimming / 2.0)
        idx_right_limit = int(len(data) * (1.0 - rate_trimming / 2.0))

        mean, std = mean_std_array(sorted_values[idx_left_limit:idx_right_limit])
        """
        mean = np.mean(sorted_values[idx_left_limit:idx_right_limit])
        std = 0

        if calc_std:
            std = np.std(sorted_values[idx_left_limit:idx_right_limit])
        """

    return np.array([mean, std])





