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


def bounding_box(iterable):
    print(iterable.shape)
    min_x, min_y = np.min(iterable, axis=0)
    max_x, max_y = np.max(iterable, axis=0)
    return min_x, max_x, min_y, max_y


def plot_plane(a, b, c, d):
        yy, zz = np.mgrid[-4:4, 0:8]
        return (-d - c * zz - b * yy) / a, yy, zz


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
        #min_x, max_x, min_y, max_y = bounding_box(cloud_xyz[:, 0:2])
        xx, yy, zz = plot_plane(a, b, c, d)  # , min_x, max_x, min_y, max_y)
        ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))
        ax.scatter3D(cloud_xyz.T[0], cloud_xyz.T[1], cloud_xyz.T[2])
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        plt.show()
    return (a, b, c, d), inliers


if __name__ == '__main__':
    n = 100
    max_iterations = 100
    goal_inliers = n * 0.8

    N_POINTS = 100
    TARGET_A = -0.6
    TARGET_B = -0.3
    TARGET_C = -0.4
    TARGET_D = -1.5
    EXTENTS = 10.0
    NOISE = 0.1

    # create random data
    xyzs = np.zeros((N_POINTS, 3))
    xyzs[:, 0] = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
    xyzs[:, 1] = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
    for i in range(N_POINTS):
        xyzs[i, 2] = (-TARGET_D - xyzs[i, 0]*TARGET_A - xyzs[i, 1]*TARGET_B)/TARGET_C + np.random.normal(scale=NOISE)

    # RANSAC
    m, inliers = plane_fitting_ransac(xyzs, 0.01, 3, goal_inliers, max_iterations, plot=True)
    scale = TARGET_D/m[3]
    print(np.array(m)*scale)
    print('Inliers: ', len(inliers))

    #m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    #a, b, c, d = m
