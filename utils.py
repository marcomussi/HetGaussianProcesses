import numpy as np


def make_json_serializable(obj):
    """
    Makes every object serializable to be saved in json
    
    Args: 
        obj: object to serialize
    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return str(obj) 



def update_inverse_rank_one(Kinv, i, j, alpha):
    """
    Performs a rank-1 update to the inverse of a matrix using the
    Sherman-Morrison formula.

    This function assumes that the inverse of a square matrix K
    (given as Kinv = K^{-1}) is known, and that K is modified by adding
    a scalar 'alpha' to the (i, j) entry of K:
        K_new = K + alpha * e_i e_j^T

    Using the Sherman-Morrison rank-1 update identity, the updated
    inverse is computed efficiently without re-inverting the full matrix.

    Args:
        Kinv (numpy.ndarray): The inverse of the original matrix K.
                              Must be a square 2D array of shape (n, n).
        i (int): Row index of the updated entry in K.
        j (int): Column index of the updated entry in K.
        alpha (float): The scalar added to K[i, j].

    Returns:
        numpy.ndarray: The updated inverse matrix K_new^{-1}.
                       Shape (n, n).

    Raises:
        ValueError:
            - If Kinv is not a 2D square numpy array.
            - If i or j are not valid indices for Kinv.
            - If alpha is not a scalar.
        numpy.linalg.LinAlgError:
            - If the Sherman-Morrison denominator becomes zero,
              meaning the updated matrix is singular and the inverse
              does not exist.

    Note:
        This function assumes Kinv represents the exact inverse of a
        full-rank matrix K. It does not compute pseudo-inverses.
    """

    if not isinstance(Kinv, np.ndarray) or Kinv.ndim != 2:
        raise ValueError(f"'Kinv' must be a 2D numpy array. Got {type(Kinv)} with ndim={getattr(Kinv, 'ndim', None)}")

    n, m = Kinv.shape
    if n != m:
        raise ValueError(f"'Kinv' must be square. Got shape {Kinv.shape}")

    if not (isinstance(i, int) and 0 <= i < n):
        raise ValueError(f"Index 'i' out of range. Expected 0 <= i < {n}, got {i}")

    if not (isinstance(j, int) and 0 <= j < n):
        raise ValueError(f"Index 'j' out of range. Expected 0 <= j < {n}, got {j}")

    if not np.isscalar(alpha):
        raise ValueError(f"'alpha' must be a scalar. Got {type(alpha)}")

    u = np.zeros((n, 1))
    v = np.zeros((n, 1))
    u[i, 0] = alpha
    v[j, 0] = 1.0

    numerator = Kinv @ u @ (v.T @ Kinv)
    denom = 1.0 + (v.T @ Kinv @ u)[0, 0]

    if denom == 0:
        raise np.linalg.LinAlgError(
            "Sherman-Morrison update failed: denominator is zero: updated matrix is singular."
        )

    return Kinv - (numerator / denom)



def incr_inv(A_inv, B, C, D):
    """
    Calculates the inverse of a 2x2 block matrix using the block matrix
    inversion formula (Schur complement method).

    This function is "incremental" because it assumes the inverse of the
    top-left block 'A' (A_inv) is already known. It efficiently computes
    the inverse of the full-rank matrix M:
        M = [[A, B],
             [C, D]]
    
    Args:
        A_inv (numpy.ndarray): The inverse of the top-left block (A).
                               Shape (n, n).
        B (numpy.ndarray): The top-right block. Shape (n, n_add).
        C (numpy.ndarray): The bottom-left block. Shape (n_add, n).
        D (numpy.ndarray): The bottom-right block. Shape (n_add, n_add).

    Returns:
        numpy.ndarray: The inverse of the full block matrix M.
                       Shape (n + n_add, n + n_add).

    Raises:
        ValueError: If any input is not a 2D numpy array or if the matrix 
                    dimensions are inconsistent.
        numpy.linalg.LinAlgError: If the Schur complement (D - C @ A_inv @ B)
                                  is singular (not invertible).
    
    Note:
        This function assumes all input matrices are 2D and square where
        appropriate (A_inv is n x n, D is n_add x n_add). It does not
        compute pseudo-inverses.
    """

    n = A_inv.shape[0] # original number of rows/cols
    n_add = C.shape[0] # number of rows/cols to add
    
    if not isinstance(A_inv, np.ndarray) or A_inv.ndim != 2:
        raise ValueError(f"Input 'A_inv' must be a 2D numpy array. Got {type(A_inv)} with ndim={A_inv.ndim}")
    if not isinstance(B, np.ndarray) or B.ndim != 2:
        raise ValueError(f"Input 'B' must be a 2D numpy array. Got {type(B)} with ndim={B.ndim}")
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        raise ValueError(f"Input 'C' must be a 2D numpy array. Got {type(C)} with ndim={C.ndim}")
    if not isinstance(D, np.ndarray) or D.ndim != 2:
        raise ValueError(f"Input 'D' must be a 2D numpy array. Got {type(D)} with ndim={D.ndim}")
    if A_inv.shape != (n, n):
        raise ValueError(f"Input 'A_inv' must be square (n, n). Got {A_inv.shape}")
    if B.shape != (n, n_add):
        raise ValueError(f"Shape mismatch: 'B' must be (n, n_add). "
                         f"Got B.shape={B.shape}, expected ({n}, {n_add})")
    if C.shape != (n_add, n):
        raise ValueError(f"Shape mismatch: 'C' must be (n_add, n). "
                         f"Got C.shape={C.shape}, expected ({n_add}, {n})")
    if D.shape != (n_add, n_add):
        raise ValueError(f"Shape mismatch: 'D' must be (n_add, n_add). "
                         f"Got D.shape={D.shape}, expected ({n_add}, {n_add})")

    schur_complement = D - C @ A_inv @ B
    schur_complement_inv = np.linalg.solve(schur_complement, np.eye(n_add))

    block1 = A_inv + A_inv @ B @ schur_complement_inv @ C @ A_inv
    block2 = -A_inv @ B @ schur_complement_inv
    block3 = -schur_complement_inv @ C @ A_inv
    block4 = schur_complement_inv

    res1 = np.concatenate((block1, block2), axis=1)
    res2 = np.concatenate((block3, block4), axis=1)

    return np.concatenate((res1, res2), axis=0)



def kernel_rbf(a, b, L):
    """
    Calculates the Radial Basis Function (RBF) kernel matrix between two sets of vectors.
    The formula implemented is:
    K(a_i, b_j) = exp(-L * ||a_i - b_j||^2)

    Where:
    - a_i is the i-th row (vector) in matrix `a`.
    - b_j is the j-th row (vector) in matrix `b`.
    - ||...||^2 is the squared Euclidean distance.
    - L is the length-scale parameter (often related to gamma, e.g., L = gamma).
      A smaller 'L' results in a wider kernel, meaning points further
      apart are considered more similar (smoother function). A larger 'L'
      results in a narrower kernel, focusing more on local similarity.

    Args:
        a (numpy.ndarray): A 2D array of shape (n_samples_a, n_features)
                           representing the first set of vectors.
        b (numpy.ndarray): A 2D array of shape (n_samples_b, n_features)
                           representing the second set of vectors.
                           `a` and `b` must have the same number of features.
        L (float): The length-scale parameter (gamma) of the kernel. Must be
                   a positive value.

    Returns:
        numpy.ndarray: A 2D array (kernel matrix) of shape (n_samples_a, n_samples_b)
                       where the element (i, j) is the RBF kernel similarity
                       between `a[i]` and `b[j]`.

    Example:
        >>> import numpy as np
        >>> a = np.array([[0, 0], [1, 1]])
        >>> b = np.array([[0, 0], [1, 1], [2, 2]])
        >>> L = 0.5
        >>> kernel_rbf(a, b, L)
        array([[1. , 0.3, 0.0],
               [0.3, 1. , 0.3]])
    """
    if not isinstance(a, np.ndarray) or a.ndim != 2:
        raise ValueError(f"Input 'a' must be a 2D numpy array. Got shape {a.shape}")
    if not isinstance(b, np.ndarray) or b.ndim != 2:
        raise ValueError(f"Input 'b' must be a 2D numpy array. Got shape {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Inputs 'a' and 'b' must have the same number of features (columns). "
                         f"Got a.shape[1] = {a.shape[1]} and b.shape[1] = {b.shape[1]}")
    if not (isinstance(L, (int, float)) and L > 0):
        raise ValueError(f"Parameter 'L' must be a positive number. Got {L}")

    sq_dists = np.ones((a.shape[0], b.shape[0]))
    
    for i in range(a.shape[0]):
        
        for j in range(b.shape[0]):
            
            sq_dists[i, j] = np.power(np.linalg.norm(a[i, :] - b[j, :], 2), 2)
    
    return np.exp(-L * sq_dists)



def aggregate_dataset(x, y):
    """
    Aggregates duplicate entries in a dataset by calculating the mean of target values.

    Args:
        x (numpy.ndarray): A 2D array of shape (n_samples, n_features).
        y (numpy.ndarray): A 1D or 2D array of shape (n_samples,) or (n_samples, 1).

    Returns:
        tuple: (x_unique, y_mean, counts)

    Raises:
        ValueError: If x and y have different numbers of samples or if inputs are empty.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2 or y.ndim != 1:
        raise ValueError(
            f"Shape mismatch: x has {x.ndim} dimensions (2 needed), but y has {y.ndim} dimensions (1 needed)."
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Shape mismatch: x has {x.shape[0]} samples, but y has {y.shape[0]} samples."
        )

    x_unique, inverse_indices = np.unique(x, axis=0, return_inverse=True)

    y_sum = np.zeros(len(x_unique), dtype=float)
    np.add.at(y_sum, inverse_indices, y)

    counts = np.zeros(len(x_unique), dtype=int)
    np.add.at(counts, inverse_indices, 1)

    y_mean = y_sum / counts

    return x_unique, y_mean.reshape(-1, 1), counts.ravel()



def associate_nearest(x, xcover):
    """
    For each element in x, find the nearest element in xcover.

    Args:
        x : 1D array-like
            Points to be associated.
        xcover : 1D array-like
            Reference points.

    Return:
        values : np.ndarray
            Nearest xcover values for each x.
    """
    x = np.asarray(x)
    xcover = np.asarray(xcover)

    distances = np.abs(x[:, None] - xcover[None, :])

    indices = np.argmin(distances, axis=1)

    values = xcover[indices]

    return values
