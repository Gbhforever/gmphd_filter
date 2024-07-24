import numpy as np
import numpy.linalg as lin
from typing import List, Dict, Any

import scipy.special


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

class GmphdFilter_svsf:
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
        self.H_1 = lin.inv(self.H[:, 0:self.u])
        phi = self.t @ self.F @ lin.inv(self.t)
        [phi_x, phi_y] = np.shape(phi)
        self.phi_22 = phi[int(phi_x / 2):phi_x, int(phi_y / 2):phi_y]
        self.phi_12 = phi[0:int(phi_x / 2), int(phi_y / 2):phi_y]
        self.phi_12_inv = lin.inv(self.phi_12)


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

    def prediction(self, v: GaussianMixture) -> GaussianMixture:
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # v_pred = v_s + v_spawn +  v_new_born
        birth_copy = self.birth_GM.copy()
        # targets that survived v_s:
        v_s = thinning_and_displacement(v, self.p_s, self.F, self.Q)
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
        v_residual = thinning_and_displacement(v, self.p_d, self.H, self.R)
        detP = get_matrices_determinants(v_residual.P)
        invP = get_matrices_inverses(v_residual.P)
        v_residual.set_covariance_determinant_and_inverse_list(detP, invP)



        v_copy = v.copy()
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P
        P_orig = v.copy().P
        e = v_copy.e

        for z in Z:
            values = v_residual.mixture_component_values_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                error = z - self.H @ v.m[i]
                #sat = v.saturate(error,self.G,[0,self.u-1])
                if(self.l != 0): #not full measurement matrix

                    #H_1 = lin.inv(self.H[:, 0:self.u])

                    E_z = abs(error) + np.diagflat(np.full((self.u,1),self.g)) @ abs(v.e[i])
                    k_u = self.H_1 @ np.diagflat(E_z * v.saturate(error,self.G,[0,self.u-1])) @ lin.inv(np.diagflat(error))
                    #phi = self.t @ self.F @ lin.inv(self.t)
                    #[phi_x,phi_y] = np.shape(phi)
                    #phi_22 = phi[int(phi_x/2):phi_x,int(phi_y/2):phi_y]
                    #phi_12 = phi[0:int(phi_x/2),int(phi_y/2):phi_y]
                    E_y = abs(self.phi_22 @ lin.inv(self.phi_12) @ error) + np.diagflat(np.full((self.l,1),self.g)) @ abs(self.phi_12_inv@error)
                    k_l = np.diagflat(E_y) * v.saturate(self.phi_22@self.phi_12_inv@error,self.G,[self.u,np.shape(self.G)[0]]) @ lin.inv(np.diagflat(self.phi_22@self.phi_12_inv@error))@self.phi_22@self.phi_12_inv
                    K = np.vstack((k_u,k_l))
                    x_po = v.m[i] + K @ error

                    S_k = self.H_1 @ v.P[i][0:int(np.shape(v.P[i])[0]/2),0:int(np.shape(v.P[i])[0]/2)] @ self.H_1.T + self.R
                    P_kpo = v.P[i] - K@self.H@v.P[i] - v.P[i]@self.H.T@K.T + K@S_k@K.T
                    e_kpo = z - self.H @ x_po
                    w.append(values[i] / normalization_factor)
                    m.append(x_po)
                    P.append(P_kpo)
                    e.append(e_kpo)

                else:
                    t1 = (abs(error) + self.g * abs(v_residual.e[i])) * sat
                    t1 = np.identity(len(t1)) @ t1
                    t2 = lin.inv(np.identity(len(error))*error)
                    k = lin.inv(self.H) @ t1 @ t2

                    w.append(values[i] / normalization_factor)
                    x = v.m[i] + k @ error
                    m.append(x)

                    error_n = z - self.H.T @ x
                    e.append(error_n)

                    pt1 = (np.identity(len(k))-k @ self.H)
                    P_kk = pt1 @ v_residual.P[i] @ pt1.T + k @ self.R @ k.T
                    P.append(P_kk)

        return GaussianMixture(w, m, P,e)

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
            e_new = np.sum(ve[L],0) / w_new
            for i in L:
                P_new += vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))
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

    def filter_data(self, Z: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Given the list of collections of measurements for each time step, perform filtering and return the
        estimated sets of tracks for each step.

        :param Z: list of observations(measurements) for each time step
        :return X:
        list of estimated track sets for each time step
        """
        X = []
        v = GaussianMixture([], [], [], [])
        for z in Z:
            v = self.prediction(v)
            v = self.correction(v, z)
            v = self.pruning(v)
            x = self.state_estimation(v)
            X.append(x)
        return X
