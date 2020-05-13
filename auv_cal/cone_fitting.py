import numpy as np
import numpy.matlib as mat


def to_homogeneous(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def minimum_curvature_direction(xyzs):
    axyz = to_homogeneous(xyzs[:3])
    _, _, vh = np.linalg.svd(axyz)
    result = vh[0, :]
    result = result[:3] / np.linalg.norm(result)
    return result


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
        A = mat.zeros((6, 6))
        Y = mat.zeros((6, 1))
        for x, y, z in points_list:
            Q = np.asmatrix(self._order(x, y)).T
            A += Q * Q.T
            Y += z * Q
        self.coef = np.array(np.ravel(A.I * Y))
        return self.coef

    def image(self, * v) :
        return (self._order(* v) * self.coef).sum()


class CircularCone:
    """Class to model a circular cone and fit points to it using geometric principles.
    """
    def __init__(self):
        self.apex = None
        self.axis = None
        self.angle = None

    def pointConeDistance(self, point):
        """Compute distance from point to modelled cone
        The distance from a point :math:`p_i` to a cone with apex :math:`Ap` and 
        basis :math:`[\\vec{u}, \\vec{n}_1, \\vec{n}_2]`, where :math:`\\vec{u}` is
        the cone asis is defined by:

        .. math:: 
            (p_i - Ap) = \\begin{bmatrix} \\vec{u}, \\vec{n}_1, \\vec{n}_2 \\end{bmatrix} 
            \\begin{bmatrix} \\alpha \\\\ \\beta \\\\ \\gamma \\end{bmatrix}

        being :math:`\\beta` the signed distance between the point and the cone.
        """
        w = point - self.apex
        v = self.axis

        wxv = np.cross(w, v)
        n1 = wxv / np.linalg.norm(wxv)
        u = rotate(v, n1, self.angle)
        uxn1 = np.cross(u, n1)
        n2 = uxn1 / np.linalg.norm(uxn1)
        basis = create_rot_mat(u, n1, n2)
        abc = np.linalg.inv(basis) @ (point - self.apex)
        signed_distance = abc[1]
        return np.abs(signed_distance)


    def fit(self, points):
        """Fit a cone to a set or points [Ruiz2013]_

        .. [Ruiz2013] Ruiz O.; Arroyave, S.; Acosta, D.: Fitting of Analytic Surfaces to Noisy Point Clouds. In: American Journal of Computational Mathematics, Vol. 3 No. 1A, 2013, pp. 18-26. doi: 10.4236/ajcm.2013.31A004.

        1) Find minimum curvature direction K_min of a set of seed points Ln
        2) Fit paraboloid:  :math:`p(x, y) =  ax^2 + by^2 + cxy + dx + ey + f`.
        3) Calculate eigenvector of its Hessian matrix 
        
        .. math:: 
            H(p) = \\begin{bmatrix} 2e & d \\\\ d & 2f \\end{bmatrix}
        
        4) Find approximated apex by averaging cross points of all
           lines defined by its seed point and its K_{min}
        5) Find centre of gravity of circunference passing through the
           points at the same lambda from the vectors found.
        6) Estimate the opening angle with angle from these vectors
           to the newly found axis.
        """

