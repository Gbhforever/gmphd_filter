import numpy as np
import numpy.linalg as lin
from typing import List, Dict, Any
import sympy
from scipy import stats

def multivariate_gaussian(x: np.ndarray, m: np.ndarray, P: np.ndarray) -> float:
    """
    Multivatiate Gaussian Distribution

    :param x: vector
    :param m: distribution mean vector
    :param P: Covariance matrix
    :return: probability density function at x
    """
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (lin.det(P) ** 0.5))
    second_part = -0.5 * (x - m.flatten()).transpose() @ lin.inv(P) @ (x - m.flatten())
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
    def __init__(self, w: List[np.float64], m: List[np.ndarray], P: List[np.ndarray]):
        """
        The Gaussian mixture class

        :param w: list of scalar weights
        :param m: list of np.ndarray means
        :param P: list of np.ndarray covariance matrices

        Note that constructor creates detP and invP variables which can be used instead of P list, for covariance matrix
        determinant and inverse. These lists cen be initialized with assign_determinant_and_inverse function, and
        it is useful in case we already have precalculated determinant and inverse earlier.
        """
        self.w = w
        self.m = m
        self.P = P
        self.detP = None
        self.invP = None

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

    def copy(self):
        w = self.w.copy()
        m = []
        P = []
        for m1 in self.m:
            m.append(m1.copy())
        for P1 in self.P:
            P.append(P1.copy())
        return GaussianMixture(w, m, P)


def get_matrices_inverses(P_list: List[np.ndarray]) -> List[np.ndarray]:
    inverse_P_list = []
    for P in P_list:
        inverse_P_list.append(lin.pinv(P.astype(float)))
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
    return GaussianMixture(w_s, m, P)
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
def unscented_transform(u:GaussianMixture,Q:np.ndarray,R:np.ndarray):
    sigma_points=[]
    W_m = []
    W_c =[]
    a = 1
    B = 2
    k = 0.1
    if not (len(u.w)==0):
        #L_s = np.shape(u.m[0])[0]
        #L_s=12
        #lamb = a**2 *(L_s+k)-L_s

        D = np.shape(u.m)[1]
        P_size = np.shape(u.P[0])[0]
        Q_size = np.shape(Q)[0]
        R_size = np.shape(R)[0]
        m_size = np.shape(u.m[0])[0]
        D_total = P_size+Q_size+R_size
        P_z = np.zeros((D_total-P_size,P_size))
        Q_z_1 = np.zeros((P_size,Q_size))
        Q_z_2 = np.zeros((R_size,Q_size))
        R_z = np.zeros((D_total-R_size,R_size))

        L_s = D_total
        lamb = a ** 2 * (L_s + k) - L_s

        for i in range(len(u.w)):
            P_l = np.vstack((u.P[i],P_z))
            Q_l = np.vstack((Q_z_1,Q,Q_z_2))
            R_l = np.vstack((R_z,R))
            C = np.hstack((P_l,Q_l,R_l))
            #L = np.linalg.cholesky(D_total*C)
            L = np.linalg.cholesky((lamb+D_total)* C)
            #L = np.linalg.cholesky((L_s+lamb)*u.P[i])
            m = np.hstack((u.m[i],np.zeros((1,D_total-m_size))[0])).transpose()
            u_s = []
            u_s.append(m)
            w_m =[]
            w_c= []
            w_m_0 = lamb/(L_s+lamb)
            #w_m_0 = 1/3
            w_c_0 = (lamb/(L_s+lamb)) + (1-a**2+B)
            w_m.append(w_m_0)
            w_c.append(w_c_0)
            for j in range(P_size+Q_size+R_size):
                x = m + L[:,j]
                w = 1/(2*(L_s+lamb))
                #w = (1-w_m_0)/(2*D_total)
                w_m.append(w)
                w_c.append(w)
                u_s.append(x)
            for j in range(P_size+Q_size+R_size):
                x = m - L[:,j]
                w = 1 / (2 * (L_s + lamb))
                w_m.append(w)
                w_c.append(w)
                u_s.append(x)
            sigma_points.append(u_s)
            W_m.append(w_m)
            W_c.append(w_c)
        # for component in sigma_points:
        #     for point in component:
        #
    return sigma_points,W_m,W_c


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

    def spawn_mixture(self, v: GaussianMixture) -> GaussianMixture:
        """
        Spawning targets in prediction step
        """
        w = []
        m = []
        P = []
        for i, w_v in enumerate(v.w):
            for j, w_spawn in enumerate(self.w_spawn):
                w.append(w_v * w_spawn)
                m.append(self.F_spawn[j] @ v.m[i] + self.d_spawn[j])
                P.append(self.Q_spawn[j] + self.F_spawn[j] @ v.P[i] @ self.F_spawn[j].T)
        return GaussianMixture(w, m, P)

    def prediction(self, v: GaussianMixture) -> GaussianMixture:
        """
        Prediction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture of the previous step
        """
        # v_pred = v_s + v_spawn +  v_new_born
        birth_copy = self.birth_GM.copy()
        # targets that survived v_s:
        D = np.shape(self.F)[0]
        W = 1/(2*D)

        u_s = unscented_transform(v,self.Q,self.R)
        v_s = non_linear_predict(v, self.p_s, self.F, self.Q,self.F_jacobian)

        # spawning targets
        v_spawn = self.spawn_mixture(v)
        # final phd of prediction
        return GaussianMixture(v_s.w + v_spawn.w + birth_copy.w, v_s.m + v_spawn.m + birth_copy.m,
                               v_s.P + v_spawn.P + birth_copy.P)

    def correction(self, v: GaussianMixture, Z: List[np.ndarray]) -> GaussianMixture:
        """
        Correction step of the GMPHD filter
        Inputs:
        - v: Gaussian mixture obtained from the prediction step
        - Z: Measurement set, containing set of observations
        """
        v_residual = linear_update(v, self.p_d, self.H, self.R)
        detP = get_matrices_determinants(v_residual.P)
        invP = get_matrices_inverses(v_residual.P)
        v_residual.set_covariance_determinant_and_inverse_list(detP, invP)

        K = []
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])

        v_copy = v.copy()
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P

        for z in Z:
            values = v_residual.mixture_component_values_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                w.append(values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - v_residual.m[i]))
                P.append(P_kk[i].copy())

        return GaussianMixture(w, m, P)
    def birth(self, v: GaussianMixture) -> GaussianMixture:
        birth_copy = self.birth_GM.copy()
        return GaussianMixture(v.w+birth_copy.w,v.m+birth_copy.m,v.P+birth_copy.P)


    def unscented_predict_update(self, v: GaussianMixture, Z: List[np.ndarray]) -> GaussianMixture:
        x, y, xd, yd, w = sympy.symbols("x,y,xd,yd,w")
        sigma_points,W_m,W_c = unscented_transform(v,self.Q,self.R)
        K=[]
        P_kk=[]
        M_kpr=[]
        Z_kpr = []
        S_kpr=[]
        for i,target in enumerate(sigma_points):
            x_l=[]
            z_l=[]
            for points in target:
                X = points[0:5]
                V = points[5:10]
                E = points[10:12]
                x_pred = self.F.subs([(x, X[0]), (y, X[1]), (xd, X[2]), (yd, X[3]), (w, X[4])])
                z_pred = self.H @ x_pred
                x_l.append(x_pred)
                z_l.append(z_pred)
            m_kpr = np.zeros((np.shape(x_l[0])[0],1))
            z_kpr =np.zeros((np.shape(z_l[0])[0],1))
            for j in range(len(x_l)):
                m_kpr = m_kpr + W_m[i][j] * sympy.matrix2numpy(x_l[j])
                z_kpr = z_kpr + W_m[i][j] * sympy.matrix2numpy(z_l[j])
            M_kpr.append(m_kpr)
            Z_kpr.append(z_kpr)
            P_kpr = np.zeros((np.shape(x_l[0])[0],np.shape(x_l[0])[0]))
            s_kpr = np.zeros((np.shape(z_l[0])[0],np.shape(z_l[0])[0]),dtype=float)
            G_k = np.zeros((np.shape(x_l[0])[0], np.shape(z_l[0])[0]))
            for j in range(len(x_l)):
                P_kpr = P_kpr + W_c[i][j] * (sympy.matrix2numpy(x_l[j])-m_kpr) @ (
                        sympy.matrix2numpy(x_l[j])-m_kpr).transpose()
                s_kpr = s_kpr + W_c[i][j] * (sympy.matrix2numpy(z_l[j]) - z_kpr) @ (
                            sympy.matrix2numpy(z_l[j]) - z_kpr).transpose()
                G_k = G_k + W_c[i][j] * (sympy.matrix2numpy(x_l[j]) - m_kpr) @ (
                        sympy.matrix2numpy(z_l[j]) - z_kpr).transpose()

            K.append(G_k @ np.linalg.inv(s_kpr.astype(float)))
            P_kpr = 0.5*P_kpr + 0.5*P_kpr.transpose()
            P_kpr = P_kpr + 1*10**-8 * np.eye(np.shape(x_l[0])[0])
            p_tmp = P_kpr - G_k @ np.linalg.inv(s_kpr.astype(float)) @ G_k.transpose()
            #p_tmp = np.around(p_tmp.astype(np.double),10)
            P_kk.append(p_tmp)
            S_kpr.append(s_kpr)

        v_copy = v.copy()
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P
        for j, z in enumerate(Z):
            Values = []
            for i in range(len(v.w)):
                values = self.p_d * v_copy.w[i] * multivariate_gaussian(z,Z_kpr[i].astype(float),S_kpr[i].astype(float))
                Values.append(values)
            normalization_factor = np.sum(Values) + self.clutter_density_func(z)
            for k in range(len(Values)):
                w.append(Values[k] / normalization_factor)
                m.append(M_kpr[k].flatten() + K[i] @ (z - Z_kpr[k].flatten()))
                P.append(P_kk[k].copy())

        return GaussianMixture(w, m, P)



    def pruning(self, v: GaussianMixture) -> GaussianMixture:
        """
        See https://ieeexplore.ieee.org/document/7202905 for details
        """
        I = (np.array(v.w) > self.T).nonzero()[0]
        w = [v.w[i] for i in I]
        m = [v.m[i] for i in I]
        P = [v.P[i] for i in I]
        v = GaussianMixture(w, m, P)
        I = (np.array(v.w) > self.T).nonzero()[0].tolist()
        invP = get_matrices_inverses(v.P)
        vw = np.array(v.w)
        vm = np.array(v.m)
        w = []
        m = []
        P = []
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
            for i in L:
                P_new += (vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))).astype(float)
            P_new /= w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            I = [i for i in I if i not in L]

        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax:]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]

        return GaussianMixture(w, m, P)

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
        v = GaussianMixture([], [], [])
        for z in Z:
            # v = self.prediction(v)
            # v = self.correction(v, z)
            # if not (len(X)==0):
            v = self.birth(v)
            v = self.unscented_predict_update(v, z)
            v = self.pruning(v)
            x = self.state_estimation(v)
            X.append(x)
        return X
