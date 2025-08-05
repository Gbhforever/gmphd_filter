import numpy as np
import numpy.linalg as lin
from typing import List, Dict, Any
import sympy
from sympy.codegen.ast import float64

global UKF_count
global SVSF_count
import scipy.special
import time

def multivariate_gaussian(x: np.ndarray, m: np.ndarray, P: np.ndarray) -> float:
    """
    Multivatiate Gaussian Distribution

    :param x: vector
    :param m: distribution mean vector
    :param P: Covariance matrix
    :return: probability density function at x
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (lin.det(P) ** 0.5))
    second_part = -0.5 * (x - m) @ lin.inv(P) @ (x - m)
    return first_part * np.exp(second_part)


def multivariate_gaussian_predefined_det_and_inv(x: np.ndarray, m: np.ndarray, detP: np.float64,
                                                 invP: np.ndarray) -> float:
    """
    Multivariate Gaussian Distribution with provided determinant and inverse of the Gaussian mixture.
    Useful in case when we already have precalculted determinant and inverse of the covariance matrix.
    :param x: vector
    :param m: distribution mean
    :param detP: determinant of the covariance matrix
    :param invP: inverse of the covariance matrix
    :return: probability density function at x
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (detP ** 0.5))
    second_part = -0.5 * (x - m) @ invP @ (x - m)
    return first_part * np.exp(second_part)


def clutter_intensity_function(z: np.ndarray, lc: int, surveillance_region: np.ndarray):
    """
    Clutter intensity function, with the uniform distribution through the surveillance region, pg. 8
    in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark.
    :param z:
    :param lc: average number of false detections per time step
    :param surveillance_region: np.ndarray of shape (number_dimensions, 2) giving the range(min and max) for each
                                dimension
    """
    if surveillance_region[0][0] <= z[0] <= surveillance_region[0][1] and surveillance_region[1][0] <= z[1] <= \
            surveillance_region[1][1]:
        # example in two dimensions: lc/((xmax - xmin)*(ymax-ymin))
        return lc / ((surveillance_region[0][1] - surveillance_region[0][0]) * (
                surveillance_region[1][1] - surveillance_region[1][0]))
    else:
        return 0.0


class GaussianMixture:
    def __init__(self, w: List[np.float64], m: List[np.ndarray], P: List[np.ndarray], e: List[np.ndarray]):
        """
        The Gaussian mixture class

        :param w: list of scalar weights
        :param m: list of np.ndarray means
        :param P: list of np.ndarray covariance matrices

        Note that constructor creates detP and invP variables which can be used instead of P list, for covariance matrix
        determinant and inverse. These lists cen be initialized with assign_determinant_and_inverse function, and
        it is useful in case we already have precalculated determinant and inverse earlier.

        :param e: list of np.ndarray errors
        """
        self.w = w
        self.m = m
        self.P = P
        self.detP = None
        self.invP = None
        self.e = e

    def set_covariance_determinant_and_inverse_list(self, detP: List[np.float64], invP: List[np.ndarray]):
        """
        For each Gaussian component, provide the determinant and the covariance inverse
        :param detP: list of determinants for each Gaussian component in the mixture
        :param invP: list of covariance inverses for each Gaussian component in the mixture
        """
        self.detP = detP
        self.invP = invP

    def mixture_value(self, x: np.ndarray):
        """
        Gaussian Mixture function for the given vector x
        """
        sum = 0
        if self.detP is None:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian_predefined_det_and_inv(x, self.m[i], self.detP[i],
                                                                                self.invP[i])
        return sum

    def mixture_single_component_value(self, x: np.ndarray, i: int) -> float:
        """
        Single Gaussian Mixture component value for the given vector
        :param x: vector
        :param i: index of the component
        :returns: probability density function at x, multiplied with the component weght at the index i
        """
        if self.detP is None:
            return self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            return self.w[i] * multivariate_gaussian_predefined_det_and_inv(x, self.m[i], self.detP[i], self.invP[i])

    def mixture_component_values_list(self, x: np.ndarray) -> List[float]:
        """
        Sometimes it is useful to have value of each component multiplied with its weight
        :param x: vector
        :return: List[np.float64]:
        List of components values at x, multiplied with their weight.
        """
        val = []
        if self.detP is None:
            for i in range(len(self.w)):
                val.append(self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i]))
        else:
            for i in range(len(self.w)):
                val.append(
                    self.w[i] * multivariate_gaussian_predefined_det_and_inv(x, self.m[i], self.detP[i], self.invP[i]))
        return val

    def saturate(self,e: np.ndarray,G: np.ndarray,index) -> np.ndarray:
        sat =[]
        if(index ==-1):
            index = [0, np.shape(G)[0]-1]

        s = e/G[index[0]:index[1]]
        for i in range(np.shape(s)[0]):
            if s[i] >= 1:
                s[i] = 1
            elif s[i] <= -1:
                s[i] = -1
        sat.append(s)

        return sat

    def copy(self):
        w = self.w.copy()
        m = []
        P = []
        e = []
        for m1 in self.m:
            m.append(m1.copy())
        for P1 in self.P:
            P.append(P1.copy())
        for e1 in self.e:
            e.append(e1.copy())
        return GaussianMixture(w, m, P, e)


def get_matrices_inverses(P_list: List[np.ndarray]) -> List[np.ndarray]:
    inverse_P_list = []
    for P in P_list:
        inverse_P_list.append(lin.inv(P))
    return inverse_P_list


def get_matrices_determinants(P_list: List[np.ndarray]) -> List[float]:
    """
    :param P_list: list of covariance matrices
    :return:
    """
    detP = []
    for P in P_list:
        detP.append(lin.det(P))
    return detP


def thinning_and_displacement(v: GaussianMixture, p, F: np.ndarray, Q: np.ndarray):
    """
    For the given Gaussian mixture v, perform thinning with probability P and displacement with N(x; F @ x_prev, Q)
    See https://ieeexplore.ieee.org/document/7202905 for details
    """
    w = []
    m = []
    P = []
    for weight in v.w:
        w.append(weight * p)
    for mean in v.m:
        m.append(F @ mean)
    for cov_matrix in v.P:
        P.append(Q + F @ cov_matrix @ F.T)
    return GaussianMixture(w, m, P,v.e)

def non_linear_predict(v: GaussianMixture, p, F: sympy.Matrix, Q: np.ndarray,F_jacobian: sympy.Matrix):
    """
    For the given Gaussian mixture v, perform thinning with probability P and displacement with N(x; F @ x_prev, Q)
    See https://ieeexplore.ieee.org/document/7202905 for details
    """
    w_s = []
    m = []
    P = []
    x, y, xd, yd, w = sympy.symbols("x,y,xd,yd,w")
    index=0
    for weight in v.w:
        w_s.append(weight * p)
    for mean in v.m:
        F_c = F.subs([(x, mean[0]), (y, mean[1]), (xd, mean[2]), (yd, mean[3]), (w, mean[4])])
        F_c = sympy.matrix2numpy(F_c, dtype=float)
        F_c = np.hstack(F_c)
        m.append(F_c)
    for cov_matrix in v.P:
        F_j = F_jacobian.subs([(x, v.m[index][0]), (y, v.m[index][1]), (xd, v.m[index][2]), (yd, v.m[index][3]), (w, v.m[index][4])])
        F_j = sympy.matrix2numpy(F_j, dtype=float)
        F_j = np.vstack(F_j)
        P.append(Q + F_j @ cov_matrix @ F_j.T)
        index = index+1
    return GaussianMixture(w_s, m, P,v.e)
def linear_update(v: GaussianMixture, p, F: np.ndarray, Q: np.ndarray):
    """
    For the given Gaussian mixture v, perform thinning with probability P and displacement with N(x; F @ x_prev, Q)
    See https://ieeexplore.ieee.org/document/7202905 for details
    """
    w = []
    m = []
    P = []
    for weight in v.w:
        w.append(weight * p)
    for mean in v.m:
        m.append(F @ mean)
    for cov_matrix in v.P:
        P.append(Q + F @ cov_matrix @ F.T)
    return GaussianMixture(w, m, P)

# def unscented_transform(u:GaussianMixture,Q:np.ndarray,R:np.ndarray):
#     sigma_points=[]
#     W_m = []
#     W_c =[]
#     a = 1*10**-3
#     B = 2
#     k = 0.1
#     if not (len(u.w)==0):
#         #L_s = np.shape(u.m[0])[0]
#         #L_s=12
#         #lamb = a**2 *(L_s+k)-L_s
#
#         D = np.shape(u.m)[1]
#         P_size = np.shape(u.P[0])[0]
#         Q_size = np.shape(Q)[0]
#         R_size = np.shape(R)[0]
#         m_size = np.shape(u.m[0])[0]
#         D_total = P_size+Q_size+R_size
#         P_z = np.zeros((D_total-P_size,P_size))
#         Q_z_1 = np.zeros((P_size,Q_size))
#         Q_z_2 = np.zeros((R_size,Q_size))
#         R_z = np.zeros((D_total-R_size,R_size))
#
#         L_s = D_total
#         lamb = a ** 2 * (L_s + k) - L_s
#
#         for i in range(len(u.w)):
#             P_l = np.vstack((u.P[i],P_z))
#             Q_l = np.vstack((Q_z_1,Q,Q_z_2))
#             R_l = np.vstack((R_z,R))
#             C = np.hstack((P_l,Q_l,R_l))
#             #L = np.linalg.cholesky(D_total*C)
#             L = np.linalg.cholesky((lamb+D_total)* C)
#             #L = np.linalg.cholesky((L_s+lamb)*u.P[i])
#             m = np.hstack((u.m[i],np.zeros((1,D_total-m_size))[0])).transpose()
#             u_s = []
#             u_s.append(m)
#             w_m =[]
#             w_c= []
#             w_m_0 = lamb/(L_s+lamb)
#             #w_m_0 = 1/3
#             w_c_0 = (lamb/(L_s+lamb)) + (1-a**2+B)
#             w_m.append(w_m_0)
#             w_c.append(w_c_0)
#             for j in range(P_size+Q_size+R_size):
#                 x = m + L[:,j]
#                 w = 1/(2*(L_s+lamb))
#                 #w = (1-w_m_0)/(2*D_total)
#                 w_m.append(w)
#                 w_c.append(w)
#                 u_s.append(x)
#             for j in range(P_size+Q_size+R_size):
#                 x = m - L[:,j]
#                 w = 1 / (2 * (L_s + lamb))
#                 w_m.append(w)
#                 w_c.append(w)
#                 u_s.append(x)
#             sigma_points.append(u_s)
#             W_m.append(w_m)
#             W_c.append(w_c)
#         # for component in sigma_points:
#         #     for point in component:
#         #
#     return sigma_points,W_m,W_c

class GmphdFilter_UKF:
    def __init__(self, model: Dict[str, Any]):
        """
        The Gaussian Mixture Probability Hypothesis Density filter implementation.
        "The Gaussian mixture probability hypothesis density filter" by Vo and Ma.

        https://ieeexplore.ieee.org/document/1710358

        We assume linear transition and measurement model in the
        following form
            x[k] = Fx[k-1] + w[k-1]
            z[k] = Hx[k] + v[k]
        Inputs:

        - model: dictionary which contains the following elements(keys are strings):

               F: state transition matrix

               H: measurement matrix

               Q: process noise covariance matrix(of variable w[k]).

               R: measurement noise covariance matrix(of variable v[k]).

             p_d: probability of target detection

             p_s: probability of target survival

            Spawning model, see pg. 5. of the paper. It's a Gaussian Mixture conditioned on state

             F_spawn:  d_spawn: Q_spawn: w_spawn: lists of ndarray objects with the same length, see pg. 5

             G: boundary layer width

             g: svsf convergence rate

             u: number of measureable states from 0->n-1

             l number of unmeasureable states from n-> sizeof state vector

            clutt_int_fun: reference to clutter intensity function, gets only one argument, which is the current measure

               T: U: Jmax: Pruning parameters, see pg. 7.

            birth_GM: The Gaussian Mixture of the birth intensity


        """
        # to do: dtype, copy, improve performance
        self.p_s = model['p_s']
        self.F = model['F']
        self.F_jacobian = model['F_jacobian']
        self.Q = model['Q']
        self.w_spawn = model['w_spawn']
        self.F_spawn = model['F_spawn']
        self.d_spawn = model['d_spawn']
        self.Q_spawn = model['Q_spawn']
        self.birth_GM = model['birth_GM']
        self.p_d = model['p_d']
        self.H = model['H']
        self.R = model['R']
        self.clutter_density_func = model['clutt_int_fun']
        self.T = model['T']
        self.U = model['U']
        self.Jmax = model['Jmax']

        self.G = model['G']
        self.g = model['g']

        self.u = model['u']
        self.l = model['l']
        self.t = np.eye(np.shape(self.F)[0])
        assert(self.u+self.l == np.shape(self.F)[0])

        #SVSF precomputed Matrices
        self.A = model['A']
        self.H_1 = lin.inv(self.H[:, 0:self.u])
        phi = self.t @ self.A @ lin.inv(self.t)
        [phi_x, phi_y] = np.shape(phi)
        self.phi_22 = phi[int(phi_x / 2):phi_x, int(phi_y / 2):phi_y]
        self.phi_12 = phi[0:int(phi_x / 2), int(phi_y / 2):phi_y]
        #self.phi_12_inv = lin.inv(self.phi_12)


    def spawn_mixture(self, v: GaussianMixture) -> GaussianMixture:
        """
        Spawning targets in prediction step
        """
        w = []
        m = []
        P = []
        e =[]
        for i, w_v in enumerate(v.w):
            for j, w_spawn in enumerate(self.w_spawn):
                w.append(w_v * w_spawn)
                m.append(self.F_spawn[j] @ v.m[i] + self.d_spawn[j])
                P.append(self.Q_spawn[j] + self.F_spawn[j] @ v.P[i] @ self.F_spawn[j].T)
        return GaussianMixture(w, m, P, e)
    def birth(self, v: GaussianMixture) -> GaussianMixture:
        birth_copy = self.birth_GM.copy()
        return GaussianMixture(v.w+birth_copy.w,v.m+birth_copy.m,v.P+birth_copy.P,v.e+birth_copy.e)

    def prediction(self, v: GaussianMixture) -> GaussianMixture:
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # v_pred = v_s + v_spawn +  v_new_born
        birth_copy = self.birth_GM.copy()
        # targets that survived v_s:
        v_s = non_linear_predict(v, self.p_s, self.F, self.Q, self.F_jacobian)
        # spawning targets
        v_spawn = self.spawn_mixture(v)
        # final phd of prediction
        return GaussianMixture(v_s.w + v_spawn.w + birth_copy.w, v_s.m + v_spawn.m + birth_copy.m,
                               v_s.P + v_spawn.P + birth_copy.P, v_s.e+v_spawn.e+birth_copy.e)

    def correction(self, v: GaussianMixture, Z: List[np.ndarray]) -> GaussianMixture:
        """
        Correction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture obtained from the prediction step
        - Z: Measurement set, containing set of observations
        """
        x, y, xd, yd, w = sympy.symbols("x,y,xd,yd,w")
        v_residual = thinning_and_displacement(v, self.p_d, self.H, self.R)
        detP = get_matrices_determinants(v_residual.P)
        invP = get_matrices_inverses(v_residual.P)
        v_residual.set_covariance_determinant_and_inverse_list(detP, invP)

        K_EKF = []
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K_EKF.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])
        v_copy = v.copy()
        weight = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P
        P_orig = v.copy().P
        e = v_copy.e
        #a = time.time()
        for z in Z:
            values = v_residual.mixture_component_values_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                error = z - self.H @ v.m[i]
                sat = v.saturate(error,self.G,[0,self.u])
                weigh =  values[i]/normalization_factor
                if(weigh >= self.T):
                    if (np.any(np.abs(sat)>=1.)):
                        phi_12 = self.phi_12
                        phi_12 = phi_12.subs(([(xd, v.m[i][2]), (yd, v.m[i][3]), (w, v.m[i][4])]))
                        phi_12 = sympy.matrix2numpy(phi_12, dtype=float)
                        inv_phi_12 = lin.pinv(phi_12)
                        phi_22 = self.phi_22
                        phi_22 = phi_22.subs(([(xd, v.m[i][2]), (yd, v.m[i][3]), (w, v.m[i][4])]))
                        phi_22 = sympy.matrix2numpy(phi_22, dtype=float)

                        t_1 = phi_22 @ inv_phi_12 @ error
                        E_z = abs(error) + np.diagflat(np.full((self.u,1),self.g)) @ abs(v.e[i])
                        k_u = self.H_1 @ np.diagflat(E_z * v.saturate(error,self.G,[0,self.u-1])) @ lin.pinv(np.diagflat(error))

                        E_y = abs(t_1) + np.diagflat(np.full((self.l,1),self.g)) @ abs(inv_phi_12@error)
                        k_l = np.diagflat(E_y) * v.saturate(t_1,self.G,[self.u-1,np.shape(self.G)[0]]) @ lin.pinv(np.diagflat(t_1))@phi_22@inv_phi_12

                        K = np.vstack((k_u,k_l))
                        x_po = v.m[i] + K @ error

                        S_k = self.H_1 @ v.P[i][0:int(np.shape(v.P[i])[0]/2),0:int(np.shape(v.P[i])[0]/2)] @ self.H_1.T + self.R
                        P_kpo = v.P[i] - K@self.H@v.P[i] - v.P[i]@self.H.T@K.T + K@S_k@K.T
                        e_kpo = z - self.H @ x_po
                        weight.append(weigh)
                        m.append(x_po)
                        P.append(P_kpo)
                        e.append(e_kpo)
                        #print("using svsf")
                        global SVSF_count
                        SVSF_count = SVSF_count + 1
                    else:
                        weight.append(weigh)
                        m.append(v.m[i] + K_EKF[i] @ (z - v_residual.m[i]))
                        P.append(P_kk[i].copy())
                        e.append(z- self.H @(v.m[i] + K_EKF[i] @ (z - v_residual.m[i])))
                        #print("using EKF")
                        global EKF_count
                        EKF_count = EKF_count+1
        #print('svsf time: ' + str(time.time() - a) + ' sec')

        return GaussianMixture(weight, m, P,e)
    # def unscented_predict_update(self, v: GaussianMixture, Z: List[np.ndarray]) -> GaussianMixture:
    #     x, y, xd, yd, w = sympy.symbols("x,y,xd,yd,w")
    #     #a = time.time()
    #     sigma_points,W_m,W_c = unscented_transform(v,self.Q,self.R)
    #     #sigma_points, W_m, W_c = unscented_transform(v)
    #     #print('sigma time: ' + str(time.time() - a) + ' sec')
    #     UKF_K=[]
    #     P_kk=[]
    #     M_kpr=[]
    #     Z_kpr = []
    #     S_kpr=[]
    #     ptime = 0
    #     pretime =0
    #     for i,target in enumerate(sigma_points):
    #         x_l=[]
    #         z_l=[]
    #         #a = time.time()
    #         for points in target:
    #             X = points[0:5]
    #             V = points[5:10]
    #             E = points[10:12]
    #             x_pred = self.F.xreplace({x: X[0], y: X[1], xd: X[2], yd: X[3], w: X[4]})
    #             x_pred = sympy.matrix2numpy(x_pred).flatten() + V
    #             z_pred = self.H @ x_pred +E
    #             x_l.append(x_pred)
    #             z_l.append(z_pred)
    #         #ptime = ptime + time.time() - a
    #         m_kpr = np.zeros((np.shape(x_l[0])[0]))
    #         z_kpr =np.zeros((np.shape(z_l[0])[0]))
    #         for j in range(len(x_l)):
    #             #m_kpr = m_kpr + W_m[i][j] * sympy.matrix2numpy(x_l[j])
    #             #z_kpr = z_kpr + W_m[i][j] * sympy.matrix2numpy(z_l[j])
    #             m_kpr = m_kpr + W_m[i][j] * x_l[j]
    #             z_kpr = z_kpr + W_m[i][j] * z_l[j]
    #         M_kpr.append(m_kpr.astype(float))
    #         Z_kpr.append(z_kpr.astype(float))
    #         P_kpr = np.zeros((np.shape(x_l[0])[0],np.shape(x_l[0])[0]))
    #         s_kpr = np.zeros((np.shape(z_l[0])[0],np.shape(z_l[0])[0]),dtype=float)
    #         G_k = np.zeros((np.shape(x_l[0])[0], np.shape(z_l[0])[0]))
    #         # for j in range(len(x_l)):
    #         #     P_kpr = P_kpr + W_c[i][j] * (sympy.matrix2numpy(x_l[j])-m_kpr) @ (
    #         #             sympy.matrix2numpy(x_l[j])-m_kpr).transpose()
    #         #     s_kpr = s_kpr + W_c[i][j] * (sympy.matrix2numpy(z_l[j]) - z_kpr) @ (
    #         #                 sympy.matrix2numpy(z_l[j]) - z_kpr).transpose()
    #         #     G_k = G_k + W_c[i][j] * (sympy.matrix2numpy(x_l[j]) - m_kpr) @ (
    #         #             sympy.matrix2numpy(z_l[j]) - z_kpr).transpose()
    #         #a = time.time()
    #         for j in range(len(x_l)):
    #             # xd = (x_l[j]-m_kpr)
    #             # zd = (z_l[j] - z_kpr)
    #             # P_kpr = P_kpr + W_c[i][j] * np.outer(xd,xd.transpose())
    #             # s_kpr = s_kpr + W_c[i][j] * np.outer(zd,zd.transpose())
    #             # G_k = G_k + W_c[i][j] * np.outer(xd,zd.transpose())
    #             P_kpr = P_kpr + W_c[i][j] * np.outer((x_l[j]-m_kpr),(
    #                     x_l[j]-m_kpr).transpose())
    #             s_kpr = s_kpr + W_c[i][j] * np.outer((z_l[j] - z_kpr) ,(
    #                         z_l[j] - z_kpr).transpose())
    #             G_k = G_k + W_c[i][j] * np.outer((x_l[j] - m_kpr) , (
    #                     z_l[j] - z_kpr).transpose())
    #         #pretime = pretime + time.time() - a
    #         UKF_K.append((G_k @ np.linalg.inv(s_kpr.astype(float))).astype(float))
    #         P_kpr = 0.5*P_kpr + 0.5*P_kpr.transpose()
    #         P_kpr = P_kpr + 1*10**-8 * np.eye(np.shape(x_l[0])[0])
    #         p_tmp = P_kpr - G_k @ np.linalg.inv(s_kpr.astype(float)) @ G_k.transpose()
    #         #p_tmp = np.around(p_tmp.astype(np.double),10)
    #         P_kk.append(p_tmp.astype(float))
    #         S_kpr.append(s_kpr)
    #     #print("pretime is " + str(pretime) + "sec")
    #     #print("pointtime is " + str(ptime) + "sec")
    #     v_copy = v.copy()
    #     weight = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
    #     m = v_copy.m
    #     P = v_copy.P
    #     e = v_copy.e
    #     for j, z in enumerate(Z):
    #         Values = []
    #         for i in range(len(v.w)):
    #             values = self.p_d * v_copy.w[i] * multivariate_gaussian(z,Z_kpr[i].astype(float),S_kpr[i].astype(float))
    #             Values.append(values)
    #         normalization_factor = np.sum(Values) + self.clutter_density_func(z)
    #         for i in range(len(v.w)):
    #             error = z - self.H @ v.m[i]
    #             sat = v.saturate(error, self.G, [0, self.u])
    #             if (Values[i] == 0):
    #                 weigh = 0
    #             else:
    #                 weigh = Values[i] / normalization_factor
    #             if (weigh >= self.T):
    #                 if (np.any(np.abs(sat)>=1.)):
    #                     phi_12 = self.phi_12.copy()
    #                     phi_12 = phi_12.xreplace({xd:v.m[i][2],yd:v.m[i][3],w:v.m[i][4]})
    #                     phi_12 = sympy.matrix2numpy(phi_12, dtype=float)
    #                     inv_phi_12 = lin.pinv(phi_12)
    #                     phi_22 = self.phi_22.copy()
    #                     phi_22 = phi_22.xreplace({xd:v.m[i][2],yd:v.m[i][3],w:v.m[i][4]})
    #                     phi_22 = sympy.matrix2numpy(phi_22, dtype=float)
    #
    #                     t_1 = phi_22 @ inv_phi_12 @ error
    #                     E_z = abs(error) + np.diagflat(np.full((self.u, 1), self.g)) @ abs(v.e[i])
    #                     k_u = self.H_1 @ np.diagflat((E_z) * v.saturate(error, self.G, [0, self.u - 1])) @ lin.pinv(
    #                         np.diagflat(error))
    #
    #                     E_y = abs(t_1) + np.diagflat(np.full((self.l, 1), self.g)) @ abs(inv_phi_12 @ error)
    #                     k_l = np.diagflat(E_y) * v.saturate(t_1, self.G, [self.u - 1, np.shape(self.G)[0]]) @ lin.pinv(
    #                         np.diagflat(t_1)) @ phi_22 @ inv_phi_12
    #
    #                     K = np.vstack((k_u, k_l))
    #                     x_po = v.m[i] + K @ error
    #
    #                     # S_k = self.H_1 @ v.P[i][0:int(np.shape(v.P[i])[0]/2),0:int(np.shape(v.P[i])[0]/2)] @ self.H_1.T + self.R
    #                     S_k = self.H @ v.P[i] @ self.H.T + self.R
    #                     P_kpo = v.P[i] - K @ self.H @ v.P[i] - v.P[i] @ self.H.T @ K.T + K @ S_k @ K.T
    #                     # try:
    #                     #     P_kpo = v.P[i] - K@self.H@v.P[i] - v.P[i]@self.H.T@K.T + K@S_k@K.T
    #                     # except:
    #                     #     bk=0
    #                     e_kpo = z - self.H @ x_po
    #                     weight.append(weigh)
    #                     m.append(x_po)
    #                     P.append(P_kpo)
    #                     e.append(e_kpo)
    #                     # print("using svsf")
    #                     global SVSF_count
    #                     SVSF_count = SVSF_count + 1
    #                 else:
    #                     weight.append(Values[i] / normalization_factor)
    #                     m.append(M_kpr[i].flatten() + UKF_K[i] @ (z - Z_kpr[i].flatten()))
    #                     e.append(z - self.H @ (M_kpr[i].flatten() + UKF_K[i] @ (z - Z_kpr[i].flatten())))
    #                     P.append(P_kk[i].copy())
    #
    #     return GaussianMixture(weight, m, P,e)
    def unscented_transform(self,u: GaussianMixture):
        sigma_point_arry = []
        W_m = []
        W_c = []
        a = 1 * 10 ** -3
        B = 2
        kappa = 0
        n = np.shape(u.m[0])[0] + np.shape(self.Q)[0]+np.shape(self.R)[0]

        if not (len(u.w) == 0):
            for i in range(len(u.w)):
                lambda_ = (a ** 2 * (n + kappa)) - n
                P = scipy.linalg.block_diag(u.P[i],self.Q,self.R)
                x = np.zeros(n)
                x[0:u.m[i].shape[0]] = u.m[i]
                U = np.linalg.cholesky((lambda_ + n) * P)
                c = 1 / (2 * (n + lambda_))
                Wm = np.full(2 * n + 1, c)
                Wc = np.full(2 * n + 1, c)
                Wm[0] = lambda_ / (n + lambda_)
                Wc[0] = (lambda_ / (n + lambda_)) + (1 - a ** 2 + B)
                sigmas = np.zeros((2 * n + 1, n))
                sigmas[0] = x
                for k in range(n):
                    sigmas[k + 1] = np.subtract(x, -U[:, k])
                    sigmas[n + k + 1] = np.subtract(x, U[:, k])
                sigma_point_arry.append(sigmas)
                W_m.append(Wm)
                W_c.append(Wc)

        return sigma_point_arry, W_m, W_c
    def unscented_predict_update(self, v: GaussianMixture, Z: List[np.ndarray]) -> GaussianMixture:
        x, y, xd, yd, w = sympy.symbols("x,y,xd,yd,w")
        #sigma_points,W_m,W_c = unscented_transform(v,self.Q,self.R)
        sigma_points, W_m, W_c = self.unscented_transform(v)
        UKF_K=[]
        P_kk=[]
        P_kpr=[]
        M_kpr=[]
        Z_kpr = []
        S_kpr=[]
        ptime = 0
        pretime =0
        #a = time.time()
        size_Q = self.Q.shape
        size_R = self.R.shape
        size_X = v.m[0].shape
        size_Z = (self.H@v.m[0]).shape
        for i,target in enumerate(sigma_points):
            x_l=[]
            z_l=[]

            for points in target:
                V = points[size_Q[0]:2*size_Q[0]]
                E = points[2*size_Q[0]:2*size_Q[0]+size_R[0]]
                x_pred = self.F.xreplace({x:points[0],y:points[1],xd:points[2],yd:points[3],w:points[4]})
                x_pred = sympy.matrix2numpy(x_pred).flatten()+V
                z_pred = self.H@x_pred
                x_l.append(x_pred.reshape(-1,1).astype(np.float64))
                z_l.append((z_pred+E).reshape(-1,1).astype(np.float64))
            #print('sigma time: ' + str(time.time() - a) + ' sec')

            #ptime = ptime + time.time() - a
            m_kpr = np.zeros((size_X[0],1))
            z_kpr =np.zeros((size_Z[0],1))
            p_kpr = np.zeros(size_Q)
            S = np.zeros(size_R)
            T = np.zeros(np.shape(m_kpr@z_kpr.transpose()))

            for j in range(len(x_l)):
                m_kpr = m_kpr + W_m[i][j] * x_l[j]
                z_kpr = z_kpr + W_m[i][j] * z_l[j]
            for k in range(len(x_l)):
                p_kpr = p_kpr + W_c[i][k]* ((x_l[k]-m_kpr)@(x_l[k]-m_kpr).transpose())
                S = S + W_c[i][k]* ((z_l[k]-z_kpr)@(z_l[k]-z_kpr).transpose())
                T = T+ W_c[i][k]* ((x_l[k]-m_kpr)@(z_l[k]-z_kpr).transpose())


            ukf_k = T @ np.linalg.pinv(S)

           #p_kk = (np.eye(np.shape(p_kpr)[0])-ukf_k@T.transpose())@p_kpr
            p_kk = p_kpr - ukf_k@S@ukf_k.transpose()
           #p_kk = p_kpr - T @ np.linalg.pinv(S) @ T.transpose()
            UKF_K.append(ukf_k)
            M_kpr.append(m_kpr)
            Z_kpr.append(z_kpr)
            P_kk.append(p_kk)
            P_kpr.append(p_kpr)
            S_kpr.append(S)

        #print("pretime is " + str(pretime) + "sec")
        #print("pointtime is " + str(ptime) + "sec")
        v_copy = v.copy()
        weight = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P
        e = v_copy.e
        for j, z in enumerate(Z):
            Values = []
            for i in range(len(v.w)):
                values = self.p_d * v_copy.w[i] * multivariate_gaussian(z,Z_kpr[i].flatten().astype(float),S_kpr[i].astype(float))
                Values.append(values)
            normalization_factor = np.sum(Values) + self.clutter_density_func(z)
            for i in range(len(v.w)):
                error = z - Z_kpr[i].flatten()
                sat = v.saturate(error, self.G, [0, self.u])
                if (Values[i] == 0):
                    weigh = 0
                else:
                    weigh = Values[i] / normalization_factor
                if (weigh >= self.T):
                    if False:#(np.any(np.abs(sat)>=1.)):
                        M_svsf = M_kpr[i].ravel()
                        phi_12 = self.phi_12.copy()
                        #phi_12 = phi_12.subs(([(xd, v.m[i][2]), (yd, v.m[i][3]), (w, v.m[i][4])]))
                        phi_12 = phi_12.xreplace({xd:M_svsf[2],yd:M_svsf[3],w:M_svsf[4]})
                        phi_12 = sympy.matrix2numpy(phi_12, dtype=float)
                        inv_phi_12 = lin.pinv(phi_12)
                        phi_22 = self.phi_22.copy()
                        #phi_22 = phi_22.subs(([(xd, v.m[i][2]), (yd, v.m[i][3]), (w, v.m[i][4])]))
                        phi_22 = phi_22.xreplace({xd: M_svsf[2], yd: M_svsf[3], w: M_svsf[4]})
                        phi_22 = sympy.matrix2numpy(phi_22, dtype=float)

                        t_1 = phi_22 @ inv_phi_12 @ error
                        E_z = abs(error) + np.diagflat(np.full((self.u, 1), self.g)) @ abs(v.e[i])
                        k_u = self.H_1 @ np.diagflat((E_z) * v.saturate(error, self.G, [0, self.u - 1])) @ lin.pinv(
                            np.diagflat(error))

                        E_y = abs(t_1) + np.diagflat(np.full((self.l, 1), self.g)) @ abs(inv_phi_12 @ error)
                        k_l = np.diagflat(E_y) * v.saturate(t_1, self.G, [self.u - 1, np.shape(self.G)[0]]) @ lin.pinv(
                            np.diagflat(t_1)) @ phi_22 @ inv_phi_12

                        K = np.vstack((k_u, k_l))
                        x_po = M_kpr[i].flatten() + K @ error

                        # S_k = self.H_1 @ v.P[i][0:int(np.shape(v.P[i])[0]/2),0:int(np.shape(v.P[i])[0]/2)] @ self.H_1.T + self.R
                        S_k = S_kpr[i]
                        P_kpo = P_kpr[i] - K @ self.H @ P_kpr[i] - P_kpr[i] @ self.H.T @ K.T + K @ S_k @ K.T
                        # try:
                        #     P_kpo = v.P[i] - K@self.H@v.P[i] - v.P[i]@self.H.T@K.T + K@S_k@K.T
                        # except:
                        #     bk=0
                        e_kpo = z - self.H @ x_po
                        weight.append(weigh)
                        m.append(x_po)
                        P.append(P_kpo)
                        e.append(e_kpo)
                        # print("using svsf")
                        global SVSF_count
                        SVSF_count = SVSF_count + 1
                    else:
                        weight.append(Values[i] / normalization_factor)
                        # m.append(M_kpr[i].flatten() + UKF_K[i] @ (z - Z_kpr[i].flatten()))
                        # e.append(z - self.H @ (M_kpr[i].flatten() + UKF_K[i] @ (z - Z_kpr[i].flatten())))
                        # P.append(P_kk[i].copy())
                        m.append(M_kpr[i].flatten() + UKF_K[i] @ (z - Z_kpr[i].flatten()))
                        e.append(z - self.H @ (M_kpr[i].flatten() + UKF_K[i] @ (z - Z_kpr[i].flatten())))
                        P.append(P_kk[i].copy())
                        global UKF_count
                        UKF_count = UKF_count + 1


        return GaussianMixture(weight, m, P,e)
    def pruning(self, v: GaussianMixture) -> GaussianMixture:
        """
        See https://ieeexplore.ieee.org/document/7202905 for details
        """
        I = (np.array(v.w) > self.T).nonzero()[0]
        w = [v.w[i] for i in I]
        m = [v.m[i] for i in I]
        P = [v.P[i] for i in I]
        e = [v.e[i] for i in I]


        v = GaussianMixture(w, m, P,e)
        I = (np.array(v.w) > self.T).nonzero()[0].tolist()
        invP = get_matrices_inverses(v.P)
        vw = np.array(v.w)
        vm = np.array(v.m)
        ve = np.array(v.e)
        w = []
        m = []
        P = []
        e = []
        while len(I) > 0:
            j = I[0]
            for i in I:
                if vw[i] > vw[j]:
                    j = i
            L = []
            for i in I:
                if (vm[i] - vm[j]) @ invP[i] @ (vm[i] - vm[j]) <= self.U:
                    L.append(i)
            w_new = np.sum(vw[L])
            m_new = np.sum((vw[L] * vm[L].T).T, axis=0) / w_new
            P_new = np.zeros((m_new.shape[0], m_new.shape[0]))
            #e_new = np.sum(ve[L],0) / w_new
            e_new = np.max(ve[L],0)
            for i in L:
                P_new += (vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))).astype(float)
            P_new /= w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            e.append(e_new)
            I = [i for i in I if i not in L]

        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax:]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]
            e = [e[i] for i in L]

        return GaussianMixture(w, m, P, e)

    def state_estimation(self, v: GaussianMixture) -> List[np.ndarray]:
        X = []
        for i in range(len(v.w)):
            if v.w[i] >= 0.5:
                for j in range(int(np.round(v.w[i]))):
                    X.append(v.m[i])
        return X

    def filter_data(self, Z: List[List[np.ndarray]],Queue) -> List[List[np.ndarray]]:
        """
        Given the list of collections of measurements for each time step, perform filtering and return the
        estimated sets of tracks for each step.

        :param Z: list of observations(measurements) for each time step
        :return X:
        list of estimated track sets for each time step
        """
        X = []
        v = GaussianMixture([], [], [], [])
        global UKF_count
        global SVSF_count
        UKF_count = 0
        SVSF_count = 0
        a = time.time()
        for i, z in enumerate(Z):
            #print("timestep"+str(i))
            v = self.birth(v)
            a = time.time()
            v = self.unscented_predict_update(v, z)
            print('UKF correct time: ' + str(time.time() - a) + ' sec')
            v = self.pruning(v)
            print('number of components' + str(len(v.w)))
            x = self.state_estimation(v)
            X.append(x)
        print("UKF Count:" +str(UKF_count))
        print("SVSF Count:" + str(SVSF_count))
        print(' UKF filter time: ' + str(time.time() - a) + ' sec')
        Queue.put(X)
        return X
