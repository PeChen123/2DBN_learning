import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import warnings
from typing import Dict, List, Tuple
import random
import matplotlib.pyplot as plt

def PF_DBN(X, Y, bnds, mu, lambda_w, lambda_a, partial = None, max_iter=200, h_tol=1e-8, rho_max=1e16, w_threshold=0.3, a_threshold=0.1):

    def _loss(X, Y, W, A):
        """Evaluate value and gradient of loss."""
        # 1/2n ||X(I-W) - YA||_F^2
        loss = 0.5/ n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W) - Y.dot(A), "fro")) 
        G_loss_w = -1.0/ n*(X.T.dot(X.dot(np.eye(d, d) - W) - Y.dot(A)))
        G_loss_a = -1.0/ n*(Y.T.dot(X.dot(np.eye(d, d) - W) - Y.dot(A)))
        
        return loss, G_loss_w, G_loss_a

    
    def prox_operator(W, B, A, D, mu):

        pen= mu* (np.square(np.linalg.norm(W - B, 'fro')) + np.square(np.linalg.norm(A - D, 'fro')))
        # gradient for W_k and A_k 
        g_pen_w = 2*mu*(W - B) # gradient respect to W
        g_pen_a = 2*mu*(A - D) # gradient respect to A
        return pen, g_pen_w, g_pen_a
    
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

    def loc_solver_1(X, Y, B, D, rho1, lambda_a, lambda_w, mu, bnds): 

        def _func_loc(w):
            W, A = _adj(w)
            h, _ = _h(W)
            loss, _, _ = _loss(X, Y, W, A)
            prox,_, _ = prox_operator(W, B, A, D, mu)
            l1_penalty = lambda_w * (np.abs(W)).sum() + lambda_a * (np.abs(A)).sum()
            return loss + prox + l1_penalty + 0.5 * rho1 * h * h + alpha * h


        def _grad_loc(w):
            """Evaluate value and gradient of augmented Lagrangian."""
            W, A = _adj(w)
            _, G_W, G_A = _loss(X, Y, W, A)
            _, G_pen_w, G_pen_a = prox_operator(W, B, A, D, mu)
            h, G_h = _h(W)
            
            obj_grad_w = G_W + G_pen_w + (rho1 *  h + alpha) * G_h
            obj_grad_a = G_A + G_pen_a
            grad_vec_w = np.append(obj_grad_w, -obj_grad_w, axis=0).flatten() + lambda_w * np.ones(2 * d**2)
            grad_vec_a = obj_grad_a.reshape(p, d**2)
            grad_vec_a = np.hstack((grad_vec_a, -grad_vec_a)).flatten() + lambda_a * np.ones(2 * p * d**2)
            return np.append(grad_vec_w, grad_vec_a, axis=0)
        

        wak_est = np.zeros(2 * (p + 1) * d**2)
        wak_new = np.zeros(2 * (p + 1) * d**2)

        for n_iter in range(1):
            wak_new = sopt.minimize(_func_loc, wak_est, method='L-BFGS-B', jac=_grad_loc, bounds=bnds).x
            wak_est = wak_new
        return wak_est
    

    def loc_solver_2(WA_K, w, beta, gamma, rho, mu): 
        """ After updated B, D, """
        # 2 mu W_k +rho W - beta/ (2mu + rho)
        W_k, A_k = _adj(WA_K)
        W, A = _adj(w)
        B = 2 * mu * W_k + rho * W - beta / (2 * mu + rho)
        D = 2 * mu * A_k + rho * A - gamma / (2 * mu + rho)
        return B, D

        
    def glob_solver(K, B, D, beta, gamma, rho2, bnds):

        def _func_glo(w):
            W, A = _adj(w)
            # h, _ = _h(W)
            # print(h)
            admm_a = 0
            admm_w = 0
            for i in range(K):
                penalty_w = np.trace(beta[i] @ (B[i] - W).T) + 0.5 * rho2 * np.linalg.norm(B[i] - W, 'fro') ** 2
                penalty_a = np.trace(gamma[i] @ (D[i] - A).T) + 0.5 * rho2 * np.linalg.norm(D[i] - A, 'fro') ** 2
                admm_w += penalty_w
                admm_a += penalty_a

            # return 0.5 * rho1 * h * h + alpha * h + admm_w + admm_a
            return admm_w + admm_a
    
        def _grad_glo(w):
            W, A = _adj(w)
            # _, G_h = _h(W)
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
            # G_W += (rho1 * h + alpha) * G_h
        
            # Flatten and concatenate gradients for optimization
            g_vec_w = np.append(G_W, -G_W, axis=0).flatten()
            g_vec_a = G_A.reshape(p, d**2)
            g_vec_a = np.hstack((g_vec_a, -g_vec_a)).flatten()

            return np.append(g_vec_w, g_vec_a, axis=0)
        wag_est = np.zeros(2 * (p + 1) * d**2)
        wag_new = np.zeros(2 * (p + 1) * d**2)


        for n_iter in range(1):
            wag_new = sopt.minimize(_func_glo, wag_est, method='L-BFGS-B', jac=_grad_glo, bounds=bnds).x
            wag_est = wag_new
        
            # print(f"Objective function value at iteration {n_iter}: {_func_glo(wa_est)}")
            # print(f"Gradient norm at iteration {n_iter}: {np.linalg.norm(_grad_glo(wa_est))}")
        return wag_new

        
    def update_admm_params(alpha, beta, gamma, rho1, rho2, B_all, D_all, h, W, A ,rho_max):

        # Update the ADMM parameters
        alpha += rho1 * h
        for idx, k in enumerate(selected_agents):  # Use selected_agents, not range(K)
            B, D = B_all[idx], D_all[idx]  # Access using idx, not k
            beta[idx] += rho2 * (B - W)  # Update beta for local W
            gamma[idx] += rho2 * (D - A)  # Update gamma for local A
        if rho1 < rho_max:
            rho1 *= 1.6
        if rho2 < rho_max:
            rho2 *= 1.1 # 1.0 to 1.2. 

        return alpha, beta, gamma, rho1, rho2

    K, n, d = X.shape

    if partial is not None:
        num_agents = partial  # Adjust to the number of selected agents
    else:
        num_agents = K  # Total number of agents

    p = Y.shape[2] // d

    # Lagrange multipliers
    beta = np.zeros((num_agents, d, d))
    gamma = np.zeros((num_agents, p * d, d))
    wa_est = np.zeros(2 * (p + 1) * d**2)
    wa_new = np.zeros(2 * (p + 1) * d**2)
    rho1, rho2, alpha = 1.0, 1.0, 0.0
    h, h_new = np.inf, np.inf
    
    client_data = {k: {"W_k": None, "A_k": None} for k in range(K)}

    # Augmented Lagrangian optimization loop
    for n_iter in range(max_iter):
        W, A = _adj(wa_est)

        if partial is not None: 
            selected_agents = random.sample(range(K), partial)  # Select a subset of agents 
        else:
            selected_agents = range(K)  # Use all agents

        # Local Update: Solve local subproblem for each agent in `selected_agents`
        B_all, D_all = np.zeros((K, d, d)), np.zeros((K, p * d, d))
        W_k, A_k = np.zeros((K, d, d)), np.zeros((K, p * d, d))
        WA_all = []

        for idx, k in enumerate(selected_agents):
            WA_K = loc_solver_1(X[k], Y[k], B_all[idx], D_all[idx], rho1, lambda_a, lambda_w, mu, bnds)
            B_new, D_new = loc_solver_2(WA_K, wa_est, beta[idx], gamma[idx], rho2, mu)
            B_all[idx] = B_new
            D_all[idx] = D_new
            WA_all.append(WA_K)

        # Global Update: Solve global subproblem using updated local parameters
        wa_new = glob_solver(num_agents, B_all, D_all, beta, gamma, rho2, bnds)

        for idx, k in enumerate(selected_agents):
            h_new, _ = _h(_adj(WA_all[idx])[0])
            h_new += h_new

        h = h_new / K
        wa_est = wa_new

        # Update estimated parameters for the next iteration
        alpha, beta, gamma, rho1, rho2 = update_admm_params(
            alpha, beta, gamma, rho1, rho2, B_all, D_all, h, W, A, rho_max
        )

        for idx, k in enumerate(selected_agents):
                W_k, A_k = _adj(WA_all[idx])
                client_data[k]["W_k"] = W_k  # Update W_k for this client
                client_data[k]["A_k"] = A_k  # Update A_k for this client


        # # Check convergence criterion
        if h <= h_tol:
            break
        if h > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    # Final thresholding to enforce sparsity
    w_est, a_est = _adj(wa_est)
    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < a_threshold] = 0

    W_k_all = []
    A_k_all = []
    
    for k in range(K):
        if client_data[k]["W_k"] is not None and client_data[k]["A_k"] is not None:
            # Apply thresholds
            client_data[k]["W_k"][np.abs(client_data[k]["W_k"]) < w_threshold] = 0
            client_data[k]["A_k"][np.abs(client_data[k]["A_k"]) < a_threshold] = 0

            # Append to global lists if needed
            W_k_all.append(client_data[k]["W_k"])
            A_k_all.append(client_data[k]["A_k"])

    return w_est, a_est, W_k_all, A_k_all