# Encoding: utf-8
import random
import numpy as np


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        inliers = []
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
                inliers.append(data[j])

        #print(s)
        #print('estimate:', m,)
        #print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    #print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, inliers


def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[-5:5, 2:6]
        return xx, yy, (-d - a * xx - b * yy) / c


def plane_fitting_ransac(cloud_xyz,
                         min_distance_threshold,
                         sample_size,
                         goal_inliers,
                         max_iterations,
                         plot=False):
    model, inliers = run_ransac(
        cloud_xyz,
        estimate,
        lambda x, y: is_inlier(x, y, min_distance_threshold),
        sample_size,
        goal_inliers,
        max_iterations)
    a, b, c, d = model
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(cloud_xyz.T[0], cloud_xyz.T[1], cloud_xyz.T[2])
        xx, yy, zz = plot_plane(a, b, c, d)
        ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))
        plt.show()
    return (a, b, c, d), inliers


if __name__ == '__main__':
    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3

    # test data
    xyzs = np.random.random((n, 3)) * 10
    xyzs[:50, 2:] = xyzs[:50, :1]

    # RANSAC
    a, b, c, d = plane_fitting_ransac(xyzs, 0.01, 3, goal_inliers, max_iterations, plot=True)

    #m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    #a, b, c, d = m
