import numpy as np
import math
from sklearn.neighbors import KDTree


def to_homogeneous(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def angle_between(u, v, n=None):
    """Get angle between vectors u,v with sign based on plane with unit normal n
    """
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))


def minimum_curvature_direction(xyzs):
    axyz = to_homogeneous(xyzs[:3])
    _, _, vh = np.linalg.svd(axyz)
    result = vh[0, :]
    result = result[:3] / np.linalg.norm(result)
    return result

def generate_circle_by_angles(t, C, r, theta, phi):
        # Orthonormal vectors n, u, <n,u>=0
        n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        u = np.array([-np.sin(phi), np.cos(phi), 0])
        
        # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
        P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
        return P_circle

def rodrigues_rot(P, n0, n1):
    """RODRIGUES ROTATION
     - Rotate given points based on a starting and ending vector
     - Axis k and angle of rotation theta given by vectors n0,n1
       P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
    """
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    P_rot = np.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))

    return P_rot

def rotation_matrix(axis, theta):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def create_rotation_matrix(v1, v2, v3):
    """Returns the rotation matrix associated with the vector basis provided

    Parameters
    ----------
    v1 : np.array
        Three-dimensional vector
    v2 : np.array
        Three-dimensional vector
    v3 : np.array
        Three-dimensional vector

    Returns
    -------
    np.array
        3x3 Rotation matrix
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    return np.array([[v1[0], v2[0], v3[0]],
                     [v1[1], v2[1], v3[1]],
                     [v1[2], v2[2], v3[2]]])


class Paraboloid:
    # coef must be defined by default, length must be the same as the return value of self._order()
    coef = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    def __init__(self, * coef):
        if len(coef) == 1:
            coef = list(coef)
        if  len(coef) == len(self.coef):
            self.coef = np.array(coef, dtype=np.float64)
        elif len(coef) != 0:
            raise ValueError("Curve is defined by {0} parameters".format(len(self.coef)))

    def _order(self, x, y):
        return np.array([x**2, x*y, y**2, x, y, 1])

    def fit(self, * points_list):
        A = np.zeros((6, 6))
        Y = np.zeros((6, 1))
        for x, y, z in points_list:
            Q = np.asmatrix(self._order(x, y)).T
            A += Q * Q.T
            Y += z * Q
        Ainv = np.linalg.inv(A)
        self.coef = np.array(np.ravel(Ainv @ Y))
        return self.coef

    def image(self, * v) :
        return (self._order(* v) * self.coef).sum()

class Circle:
    centre = np.array([0, 0, 0])
    radius = 1.0

    def __init__(self):
        pass

    def fit_circle_2d(self, x, y, w=[]):
        """FIT CIRCLE 2D
         - Find center [xc, yc] and radius r of circle fitting to set of 2D points
         - Optionally specify weights for points
        
         - Implicit circle function:
           (x-xc)^2 + (y-yc)^2 = r^2
           (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
           c[0]*x + c[1]*y + c[2] = x^2+y^2
        
         - Solution by method of least squares:
           A*c = b, c' = argmin(||A*c - b||^2)
           A = [x y 1], b = [x^2+y^2]
        """
        A = np.array([x, y, np.ones(len(x))]).T
        b = x**2 + y**2
        
        # Modify A,b for weighted least squares
        if len(w) == len(x):
            W = np.diag(w)
            A = np.dot(W,A)
            b = np.dot(W,b)
        
        # Solve by method of least squares
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        
        # Get circle parameters from solution c
        xc = c[0]/2
        yc = c[1]/2
        r = np.sqrt(c[2] + xc**2 + yc**2)
        return xc, yc, r

    def fit(self, P):
        # (1) Fitting plane by SVD for the mean-centered data
        # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
        P_mean = P.mean(axis=0)
        P_centered = P - P_mean
        U,s,V = np.linalg.svd(P_centered)

        # Normal vector of fitting plane is given by 3rd column in V
        # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
        normal = V[2,:]
        d = -np.dot(P_mean, normal)  # d = -<p,n>

        # (2) Project points to coords X-Y in 2D plane
        P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

        # (3) Fit circle in new 2D coords
        xc, yc, r = self.fit_circle_2d(P_xy[:,0], P_xy[:,1])

        # (4) Transform circle center back to 3D coords
        C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + P_mean
        C = C.flatten()

        return C, r

class CircularCone:
    """Class to model a circular cone and fit points to it using geometric principles.
    """
    def __init__(self):
        self.apex = None
        self.axis = None
        self.half_angle = None
        self.seed_size = 15
        self.seed_radius = 0.4

    def distanceTo(self, point):
        """Compute distance from point to modelled cone
        The distance from a point :math:`p_i` to a cone with apex :math:`Ap` and 
        basis :math:`[\\vec{u}, \\vec{n}_1, \\vec{n}_2]`, where :math:`\\vec{u}` is
        the cone asis is defined by:

        .. math:: 
            (p_i - Ap) = \\begin{bmatrix} \\vec{u}, \\vec{n}_1, \\vec{n}_2 \\end{bmatrix} 
            \\begin{bmatrix} \\alpha \\\\ \\beta \\\\ \\gamma \\end{bmatrix}

        being :math:`\\gamma` the signed distance between the point and the cone.
        """
        w = point - self.apex
        if np.linalg.norm(w) == 0:
            # The point is in the apex
            return 0
        w = w / np.linalg.norm(w)

        v = self.axis / np.linalg.norm(self.axis)
        wxv = np.cross(w, v)
        if np.linalg.norm(wxv) == 0:
            # The point lies in the axis
            d = np.linalg.norm(point-self.apex)
            return d*math.cos(self.half_angle)
        n1 = wxv / np.linalg.norm(wxv)
        R = rotation_matrix(n1, - self.half_angle)
        u = np.dot(R, v)
        uxn1 = np.cross(u, n1)
        n2 = uxn1 / np.linalg.norm(uxn1)
        basis = create_rotation_matrix(u, n1, n2)
        abc = basis.T @ w
        signed_distance = abc[2]
        return np.abs(signed_distance)

    """
    def fit(self, points):
        #Fit a cone to a set of points

        # Get a plane fitted
        plane = fit_plane(points)

        # Get the centroid
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        mean_z = np.mean(points[:, 2])
        mean_xyz = np.array([mean_x, mean_y, mean_z])

        # Get the stats of the cloud
        min_z
        max_z
        std_z = 

        # Slice in Z and fit circles
        circles = []
        for z in z_slices:
            points_slice = slice_points(points, z)
            circle = fit_circle(points_slice)
            circles.append(circle)

        # Find apex, angle and axis from circles
        apex, angle, axis = fit_cone(circles)
    """