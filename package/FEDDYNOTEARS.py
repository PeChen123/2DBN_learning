import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import warnings
from typing import Dict, List, Tuple

def admm_dynotears(X, Y, bnds, lambda_w, lambda_a, max_iter=200, h_tol=1e-8, rho_max=1e16, w_threshold=0.3, a_threshold=0.3):
    
    def _adj(w):
        """Convert the parameter vector back to original intra- and inter-slice matrices."""
        ''' [d, (p+1) * d] to [d, d] and [d, p*d] '''
        # Reshape w into intra-slice (W) and inter-slice (A) components
        w_tilde = w.reshape([2 * (p + 1) * d, d])

        # Extract positive and negative components for W
        w_plus = w_tilde[:d, :]
        w_minus = w_tilde[d:2 * d, :]
        W = w_plus - w_minus

        # Extract positive and negative components for A
        a_plus = w_tilde[2 * d:].reshape(2 * p, d**2)[::2].reshape(d * p, d)
        a_minus = w_tilde[2 * d:].reshape(2 * p, d**2)[1::2].reshape(d * p, d)
        A = a_plus - a_minus

        return W, A
    
    def _h(W):
        """Evaluate value and gradient of the acyclicity constraint for W."""
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h
    
    def loc_solver(X, Y, W, A, beta, gamma, rho): 
        """
        Compute matrices S, M, N, P, Q based on input data.
        """
        def compute_matrices(X, Y, rho):
            n = X.shape[0]
            S = (1 / n) * X.T @ X
            M = (1 / n) * X.T @ Y
            N = (1 / n) * Y.T @ Y
            P = S + rho * np.eye(S.shape[0])
            Q = N + rho * np.eye(N.shape[0])
            return S, M, N, P, Q

        def solve_linear_system(P, M, Q, b1, b2):
            """
            Solve the block linear system to find B_k and D_k.
            """
            # Eliminate D_k to solve for B_k
            Q_inv = np.linalg.inv(Q)
            P_mod = P - M @ Q_inv @ M.T
            b1_mod = b1 - M @ Q_inv @ b2
            B_k = np.linalg.solve(P_mod, b1_mod)

            # Eliminate B_k to solve for D_k
            P_inv = np.linalg.inv(P)
            Q_mod = Q - M.T @ P_inv @ M
            b2_mod = b2 - M.T @ P_inv @ b1
            D_k = np.linalg.solve(Q_mod, b2_mod)
            return B_k, D_k


        S, M, N, P, Q = compute_matrices(X, Y, rho)

        # Compute right-hand side vectors
        b1 = S - beta + rho * W
        b2 = M.T - gamma + rho * A

        # Solve for B_k and D_k
        B, D = solve_linear_system(P, M, Q, b1, b2)

        return B, D

        
    def glob_solver(K, B, D, alpha, beta, gamma, rho1, rho2, bnds, lambda_w, lambda_a):

        def _func_glo(w):
            W, A = _adj(w)
            h, _ = _h(W)
            admm_a = 0
            admm_w = 0
            for i in range(K):
                penalty_w = np.trace(beta[i] @ (B[i] - W).T) + 0.5 * rho2 * np.linalg.norm(B[i] - W, 'fro') ** 2
                admm_w += penalty_w
                penalty_a = np.trace(gamma[i] @ (D[i] - A).T) + 0.5 * rho2 * np.linalg.norm(D[i] - A, 'fro') ** 2
                admm_a += penalty_a

            return 0.5 * rho1 * h * h + alpha * h + lambda_w* (np.abs(W)).sum() + lambda_a * (np.abs(A)).sum() + admm_w + admm_a
        
        def _grad_glo(w):
            W, A = _adj(w)
            h, G_h = _h(W)
            G_W = np.zeros_like(W)
            G_A = np.zeros_like(A)

            # Sum the ADMM penalty gradients across all agents
            for i in range(K):
                penalty_w_grad = -1* beta[i] + rho2 * (W - B[i])
                penalty_a_grad = -1* gamma[i] + rho2 * (A - D[i])

                # Add penalty gradients to the global gradients
                G_W += penalty_w_grad
                G_A += penalty_a_grad

            # Add acyclicity gradient contributions
            G_W += (rho1 * h + alpha) * G_h
        
            # Flatten and concatenate gradients for optimization
            g_vec_w = np.append(G_W, -G_W, axis=0).flatten() + lambda_w * np.ones(2 * d**2) 
            g_vec_a = G_A.reshape(p, d**2)
            g_vec_a = np.hstack((g_vec_a, -g_vec_a)).flatten() + lambda_a * np.ones(2 * p * d**2)

            return np.append(g_vec_w, g_vec_a, axis=0)
        
        wa_est = np.zeros(2 * (p + 1) * d**2)
        wa_new = np.zeros(2 * (p + 1) * d**2)

        for n_iter in range(5):
            wa_new = sopt.minimize(_func_glo, wa_est, method='L-BFGS-B', jac=_grad_glo, bounds=bnds).x
            wa_est = wa_new
        
            # print(f"Objective function value at iteration {n_iter}: {_func_glo(wa_est)}")
            # print(f"Gradient norm at iteration {n_iter}: {np.linalg.norm(_grad_glo(wa_est))}")
        return wa_new

        
    def update_admm_params(alpha, beta, gamma, rho1, rho2, B_all, D_all, h, W, A ,rho_max):

        # Update the ADMM parameters
        alpha += rho1 * h
        for k in range(K):
            B,D = B_all[k], D_all[k]
            beta[k] += rho2 * (B - W) # for local W
            gamma[k] += rho2 * (D - A) # for local A 
        if rho1 < rho_max:
            rho1 *= 1.65
        if rho2 < rho_max:
            rho2 *= 1.1 # 1.0 to 1.2. 

        return alpha, beta, gamma, rho1, rho2

    K, n, d = X.shape
    p = Y.shape[2] // d
    print(p)
    # Lagrange multipliers
    beta, gamma = np.zeros((K, d, d)), np.zeros((K, p*d,  d))
    wa_est = np.zeros(2 * (p + 1) * d**2)
    wa_new = np.zeros(2 * (p + 1) * d**2)
    rho1, rho2, alpha = 1., 1., 0.0
    h, h_new = np.inf, np.inf
  

    # Augmented Lagrangian optimization loop
    for n_iter in range(max_iter):
        
        W, A = _adj(wa_est)
        # Local Update: Solve local subproblem for each agent
        B_all = []
        D_all = []

        for k in range(K):
            B_k, D_k = loc_solver(X[k], Y[k], W, A, beta[k], gamma[k], rho2)
            B_all.append(B_k)
            D_all.append(D_k)
        
        B_all = np.array(B_all)  # B_all now has the shape (K, d, d)
        D_all = np.array(D_all) 
        # # Save B_all and D_all as .npy files
        # np.save('B_all.npy', B_all)  # Save B_all to a file named 'B_all.npy'
        # np.save('D_all.npy', D_all)  # Save D_all to a file named 'D_all.npy'

        # Global Update: Solve global subproblem using updated local parameters
        wa_new= glob_solver(K, B_all, D_all, alpha, beta, gamma, rho1, rho2, bnds,lambda_w, lambda_a)

        # Update h for acyclicity
        h_new, _ = _h(_adj(wa_new)[0])
        h = h_new
        wa_est = wa_new
    
        # Update estimated parameters for the next iteration
        alpha, beta, gamma, rho1, rho2 = update_admm_params(alpha, beta, gamma, rho1, rho2, B_all, D_all, h, W, A ,rho_max)
        
        # Check convergence criterion
        if h <= h_tol:
            break
        if h> h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    # Final thresholding to enforce sparsity
    w_est, a_est = _adj(wa_est)
    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < a_threshold] = 0

    return w_est, a_est

def opt_boundary(X, Xlags, d_vars, tabu_edges: List[Tuple[int, int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    allow_self_loops: bool = False):

    _, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars
    # Bounds for intra-slice weights (W)

    bnds_w = 2 * [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges is not None and (0, i, j) in tabu_edges
        else (0, 0)
        if tabu_parent_nodes is not None and i in tabu_parent_nodes
        else (0, 0)
        if tabu_child_nodes is not None and j in tabu_child_nodes
        else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]

    bnds_a = []
    for k in range(1, p_orders + 1):
        bnds_a.extend(
            2
            * [
                (0, 0)
                if tabu_edges is not None and (k, i, j) in tabu_edges
                else (0, 0)
                if tabu_parent_nodes is not None and i in tabu_parent_nodes
                else (0, 0)
                if tabu_child_nodes is not None and j in tabu_child_nodes
                else (0, None)
                for i in range(d_vars)
                for j in range(d_vars)
            ]
        )

    bnds = bnds_w + bnds_a
    return bnds