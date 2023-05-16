import numpy as np
from scipy import optimize
from tqdm import tqdm


class KernelSVC:
    def __init__(self, C, kernel, epsilon=1e-3, precomputed=None, class_weight = "balanced"):
        self.type = "non-linear"
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.y = None
        self.X = None
        self.precomputed = precomputed
        self.class_weight = class_weight


    def fit(self, X, y, iterations):
        #### You might define here any variable needed for the rest of the code
        self.y = y
        self.X = X
        N = len(y)
        Y = np.outer(y, y.T)
        n = len(X)
        if self.precomputed is None:
            kernel_X = np.ones((n, n))
            for i in tqdm(range(n)):
                for j in range(i + 1, n):
                    similarity = self.kernel(X[i], X[j])
                    kernel_X[i, j] = similarity
                    kernel_X[j, i] = similarity
        else:
            kernel_X = self.precomputed
        n_class_0 = np.sum(y[y==-1])
        if self.class_weight == "balanced":
            weights = []
            for label in y:
                if label == -1:
                    weights.append(N/(2*n_class_0))
                else:
                    weights.append(N/(2*(N-n_class_0)))
            weights = np.array(weights)
        else:
            weights = np.ones(N)
        # Lagrange dual problem
        def loss(alpha):
            return np.dot(alpha.T, np.dot(kernel_X * Y, alpha)) / 2 - np.sum(alpha)

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return np.dot(kernel_X * Y, alpha) - np.ones(N)

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: np.dot(
            alpha, y
        )  # '''----------------function defining the equality constraint------------------'''
        jac_eq = (
            lambda alpha: y
        )  #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: np.concatenate(
            [alpha, self.C * weights - alpha]
        )  # '''---------------function defining the inequality constraint-------------------'''
        jac_ineq = lambda alpha: np.concatenate(
            [np.eye(N), -np.eye(N)]
        )  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''

        constraints = (
            {"type": "eq", "fun": fun_eq, "jac": jac_eq},
            {"type": "ineq", "fun": fun_ineq, "jac": jac_ineq},
        )

        optRes = optimize.minimize(
            fun=lambda alpha: loss(alpha),
            x0=np.ones(N),
            method="SLSQP",
            jac=lambda alpha: grad_loss(alpha),
            constraints=constraints,
        )
        self.alpha = optRes.x

        self.b = -np.mean(
            np.sum(np.tile(self.alpha * self.y, (N, 1)) * kernel_X, axis=1)
        )  #''' -----------------offset of the classifier------------------ '''
        self.y = y
        self.support_indices = np.where(self.alpha > self.epsilon)
        #print(self.support_indices)
        self.alpha = self.alpha[self.alpha > self.epsilon]

        # self.norm_f = np.sqrt(np.dot(self.alpha.T, np.dot(kernel_X*Y, self.alpha))) # '''------------------------RKHS norm of the function f ------------------------------'''

    ### Implementation of the separting function $f$
    def separating_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        N = len(self.support)
        n = len(x)
        K = np.ones((n, N))
        for i in tqdm(range(n)):
            for j in range(N):
                k = self.kernel(x[i], self.support[j])
                K[i, j] = k
        return np.sum(np.tile(self.alpha * self.y, (n, 1)) * K, axis=1).T

    def predict(self, K):
        return K[:,self.support_indices].dot(self.alpha.T)


class KernelPCA:
    def __init__(self, kernel, r=2):
        self.kernel = kernel  # <---
        self.alpha = None  # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.lmbda = None  # Vector of size d representing the top d eingenvalues
        self.support = None  # Data points where the features are evaluated
        self.r = r  ## Number of principal components

    def compute_PCA(self, X):
        # assigns the vectors
        X -= np.mean(X, axis=0)
        self.support = X
        N = len(X)
        K = self.kernel(X, X)
        eigenvalues, eigenvectors = np.linalg.eig(K)
        ## we sort the eigenvalues and keep top r indices
        top_r_eigenvalues = np.abs(eigenvalues).argsort()[::-1]
        self.lmbda = np.abs(eigenvalues[top_r_eigenvalues])
        self.alpha = np.real(eigenvectors[:, top_r_eigenvalues]) / np.sqrt(self.lmbda)

        # constraints = ({})
        # Maximize by minimizing the opposite

    def transform(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        N = len(x)
        K = self.kernel(x, x)
        return np.dot(K, self.alpha)
