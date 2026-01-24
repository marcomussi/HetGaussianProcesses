import numpy as np
from utils import kernel_rbf


class IncrementalHeteroskedasticGaussianProcessRegressorRBF:

    def __init__(self, kernel_L, regularization, input_dim=1):
        self.kernel_L = kernel_L
        self.regularization = regularization
        self.input_dim = input_dim
        self.reset()

    def reset(self):
        self.x_vect = None
        self.y_vect = None
        self.num_samples = []
        self.K_matrix = None
        self.K_matrix_inv = None
        self.log_det_term = 0.0

    def add_sample(self, x, y):

        # ---------------------------
        # Input formatting
        # ---------------------------
        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim)
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1,)) or (y.ndim == 2 and y.shape == (1, 1))

        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)

        # ===========================
        # FIRST SAMPLE
        # ===========================
        if self.x_vect is None:

            self.x_vect = x
            self.y_vect = y
            self.num_samples = [1]

            K = kernel_rbf(x, x, self.kernel_L)
            self.K_matrix = K

            A = K + self.regularization * np.eye(1)
            self.K_matrix_inv = np.linalg.inv(A)

            # Initialize log-det for info gain
            self.log_det_term = np.log(1 + K[0, 0] / self.regularization)
            return

        # ===========================
        # CHECK IF SAMPLE EXISTS
        # ===========================
        matches = np.where(np.all(self.x_vect == x, axis=1))[0]

        # ==========================================================
        # CASE 1 — SAMPLE ALREADY PRESENT (RANK-1 UPDATE)
        # ==========================================================
        if len(matches) > 0:
            i = matches[0]

            # Update target mean
            n_i = self.num_samples[i]
            self.y_vect[i] = (self.y_vect[i] * n_i + y) / (n_i + 1)
            self.num_samples[i] += 1

            # Effective weight change
            w_old = 1 / np.sqrt(n_i)
            w_new = 1 / np.sqrt(n_i + 1)
            delta = (w_new**2 - w_old**2)

            # Unit vector at index i
            e = np.zeros((len(self.num_samples), 1))
            e[i] = 1

            # Sherman–Morrison update
            A_inv = self.K_matrix_inv
            denom = 1 + delta * (e.T @ A_inv @ e)

            self.K_matrix_inv = A_inv - (delta / denom) * (A_inv @ e @ e.T @ A_inv)

            # Incremental log-det update
            self.log_det_term += np.log(denom)

            return

        # ==========================================================
        # CASE 2 — NEW SAMPLE (BLOCK INVERSE UPDATE)
        # ==========================================================

        # Expand storage
        self.x_vect = np.vstack((self.x_vect, x))
        self.y_vect = np.vstack((self.y_vect, y))
        self.num_samples.append(1)

        # Kernel blocks
        k = kernel_rbf(self.x_vect[:-1], x, self.kernel_L)  # (n x 1)
        k_xx = kernel_rbf(x, x, self.kernel_L)[0, 0] + self.regularization

        A_inv = self.K_matrix_inv

        # Schur complement
        s = k_xx - k.T @ A_inv @ k
        s_inv = 1.0 / s

        # Block matrix inverse update
        top_left = A_inv + A_inv @ k @ k.T @ A_inv * s_inv
        top_right = -A_inv @ k * s_inv
        bottom_left = top_right.T
        bottom_right = np.array([[s_inv]])

        self.K_matrix_inv = np.block([
            [top_left,     top_right],
            [bottom_left,  bottom_right]
        ])

        # Expand kernel matrix cache
        self.K_matrix = np.block([
            [self.K_matrix, k],
            [k.T, kernel_rbf(x, x, self.kernel_L)]
        ])

        # Incremental log-det update
        self.log_det_term += np.log(s)

    # ==========================================================
    # GP PREDICTION
    # ==========================================================
    def compute(self, x):

        assert x.ndim == 2 and x.shape[1] == self.input_dim

        n = x.shape[0]
        postmean = np.zeros(n)
        postvariance = np.zeros(n)

        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            postmean[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            postvariance[i] = (
                kernel_rbf(x[i:i+1], x[i:i+1], self.kernel_L)
                - K_star.T @ self.K_matrix_inv @ K_star
            )

        return postmean, postvariance

    # ==========================================================
    # INFORMATION GAIN (INCREMENTAL)
    # ==========================================================
    def get_info_gain(self):
        return 0.5 * self.log_det_term
