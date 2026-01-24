import numpy as np
from utils import kernel_rbf


class IncrementalWGP_RBF:

    def __init__(self, kernel_L, regularization, input_dim=1):
        self.kernel_L = kernel_L
        self.lambda_reg = regularization
        self.input_dim = input_dim
        self.reset()

    # ======================================================
    # RESET
    # ======================================================
    def reset(self):
        self.x_vect = None                 # unique inputs
        self.y_vect = None                 # averaged outputs
        self.n_f = []                      # multiplicity per input
        self.K_tilde = None                # kernel on unique inputs
        self.A_inv = None                  # inverse of D K D + λI
        self.log_det = 0.0                 # log det(I + D K D / λ)

    # ======================================================
    # ADD SAMPLE (CORE INCREMENTAL LOGIC)
    # ======================================================
    def add_sample(self, x, y):

        x = np.array(x).reshape(1, self.input_dim)
        y = np.array(y).reshape(1, 1)

        # --------------------------------------------------
        # FIRST SAMPLE
        # --------------------------------------------------
        if self.x_vect is None:

            self.x_vect = x
            self.y_vect = y
            self.n_f = [1]

            self.K_tilde = kernel_rbf(x, x, self.kernel_L)
            D = np.diag(np.sqrt(self.n_f))

            A = D @ self.K_tilde @ D + self.lambda_reg * np.eye(1)
            self.A_inv = np.linalg.inv(A)

            # log det(I + D K D / λ)
            self.log_det = np.log(1 + self.K_tilde[0, 0] / self.lambda_reg)
            return

        # --------------------------------------------------
        # CHECK IF SAMPLE EXISTS
        # --------------------------------------------------
        matches = np.where(np.all(self.x_vect == x, axis=1))[0]

        # ==================================================
        # CASE 1 — NEW SAMPLE  (BLOCK INVERSE + SCHUR DET)
        # ==================================================
        if len(matches) == 0:

            Z = len(self.n_f)

            # expand storage
            self.x_vect = np.vstack((self.x_vect, x))
            self.y_vect = np.vstack((self.y_vect, y))
            self.n_f.append(1)

            # kernel blocks
            k = kernel_rbf(self.x_vect[:-1], x, self.kernel_L)  # (Z x 1)
            k_xx = kernel_rbf(x, x, self.kernel_L)[0, 0]

            # update K_tilde
            self.K_tilde = np.block([
                [self.K_tilde, k],
                [k.T, np.array([[k_xx]])]
            ])

            # construct scaled vector B = k ⊙ sqrt(n_f)
            n_sqrt = np.sqrt(np.array(self.n_f[:-1])).reshape(-1, 1)
            B = k * n_sqrt

            # block inverse theorem
            A_inv_old = self.A_inv
            Dval = k_xx + self.lambda_reg

            s = Dval - B.T @ A_inv_old @ B
            s_inv = 1.0 / s

            TL = A_inv_old + A_inv_old @ B @ B.T @ A_inv_old * s_inv
            TR = -A_inv_old @ B * s_inv
            BL = TR.T
            BR = np.array([[s_inv]])

            self.A_inv = np.block([[TL, TR], [BL, BR]])

            # -------- Schur determinant update --------
            u = B / self.lambda_reg
            P = 1 + k_xx / self.lambda_reg
            schur_term = P - (u.T @ A_inv_old @ u)
            self.log_det += np.log(schur_term)

            return

        # ==================================================
        # CASE 2 — REPEATED SAMPLE (DOUBLE SHERMAN–MORRISON)
        # ==================================================
        j = matches[0]

        # update output average
        n_old = self.n_f[j]
        self.y_vect[j] = (self.y_vect[j] * n_old + y) / (n_old + 1)
        self.n_f[j] += 1

        Z = len(self.n_f)

        # kernel column
        k_j = self.K_tilde[:, j:j+1]
        n_sqrt = np.sqrt(np.array(self.n_f))

        delta = np.sqrt(n_old + 1) - np.sqrt(n_old)

        # vector u from paper (except u_j = k_xx)
        u = delta * (k_j * n_sqrt.reshape(-1, 1))
        u[j] = kernel_rbf(x, x, self.kernel_L)[0, 0]

        e_j = np.zeros((Z, 1))
        e_j[j] = 1

        # ---- Sherman–Morrison #1 ----
        A_inv = self.A_inv
        denom1 = 1 + e_j.T @ A_inv @ u
        A_inv = A_inv - (A_inv @ u @ e_j.T @ A_inv) / denom1

        # ---- Sherman–Morrison #2 ----
        v = delta * (k_j * n_sqrt.reshape(-1, 1))
        v[j] = 0

        denom2 = 1 + v.T @ A_inv @ e_j
        A_inv = A_inv - (A_inv @ e_j @ v.T @ A_inv) / denom2

        self.A_inv = A_inv

        # ==================================================
        # INFORMATION GAIN — DOUBLE DETERMINANT LEMMA
        # ==================================================
        # first lemma
        v1 = delta * (k_j * n_sqrt.reshape(-1, 1)) / self.lambda_reg
        v1[j] = kernel_rbf(x, x, self.kernel_L)[0, 0] / self.lambda_reg

        term1 = 1 + v1.T @ self.A_inv @ e_j
        self.log_det += np.log(term1)

        # second lemma
        u2 = delta * (k_j * n_sqrt.reshape(-1, 1)) / self.lambda_reg
        u2[j] = 0

        term2 = 1 + e_j.T @ self.A_inv @ u2
        self.log_det += np.log(term2)

    # ======================================================
    # GP PREDICTION
    # ======================================================
    def compute(self, X):

        X = np.atleast_2d(X)
        means = np.zeros(X.shape[0])
        vars_ = np.zeros(X.shape[0])

        for i in range(X.shape[0]):

            k_star = kernel_rbf(self.x_vect, X[i:i+1], self.kernel_L)

            means[i] = k_star.T @ self.A_inv @ self.y_vect

            vars_[i] = (
                kernel_rbf(X[i:i+1], X[i:i+1], self.kernel_L)
                - k_star.T @ self.A_inv @ k_star
            )

        return means, vars_

    # ======================================================
    # INFORMATION GAIN
    # ======================================================
    def get_info_gain(self):
        return 0.5 * self.log_det
