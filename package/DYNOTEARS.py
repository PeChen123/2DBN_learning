import numpy as np
from scipy.optimize import minimize
import networkx as nx
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union


def dynotears(X, Y, bnds,lambda_w, lambda_a, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3, a_threshold=0.3):
    """
    Solve min_W L(W, A; X, Y) + lambda_W ||W||_1 + lambda_A ||A||_1 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        Y (np.ndarray): [n, p * d] lagged matrix
        lambda_W (float): L1 penalty parameter for intra-slice weights
        lambda_A (float): L1 penalty parameter for inter-slice weights
        max_iter (int): max number of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold for intra-slice
        a_threshold (float): drop edge if |weight| < threshold for inter-slice

    Returns:
        W_est (np.ndarray): [d, d] intra-slice estimated DAG
        A_est (np.ndarray): [p * d, d] inter-slice estimated weights
    """

    n, d = X.shape # n = X.shape[0], d = X.shape[1]
    p = Y.shape[1] // d

    def _loss(W, A):
        """Evaluate value and gradient of loss."""
        # M = X @ W + Y @ A
        # R = X - M
        # loss = 0.5 / n * (R ** 2).sum()
        # G_loss_w= -1/n * X.T @ R
        # G_loss_a = -1/n * Y.T @ R
        loss = 0.5/ n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W) - Y.dot(A), "fro"))
        G_loss_w = -1.0/ n*(X.T.dot(X.dot(np.eye(d, d) - W) - Y.dot(A)))
        G_loss_a = -1.0/ n*(Y.T.dot(X.dot(np.eye(d, d) - W) - Y.dot(A)))
        return loss, G_loss_w, G_loss_a

    def _h(W):
        """Evaluate value and gradient of the acyclicity constraint for W."""
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

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

    def _func(w):
        W, A = _adj(w)
        loss, _, _ = _loss(W, A)
        h, _ = _h(W)
        l1_penalty = lambda_w* (np.abs(W)).sum() + lambda_a * (np.abs(A)).sum()
        return loss + 0.5 * rho * h * h + alpha * h + l1_penalty


    def _grad(w):
        """Evaluate value and gradient of augmented Lagrangian."""
        W, A = _adj(w)
        _, G_W, G_A = _loss(W, A)
        h, G_h = _h(W)
        obj_grad_w = G_W + (rho *  h + alpha) * G_h
        obj_grad_a = G_A
        grad_vec_w = np.append(obj_grad_w, -obj_grad_w, axis=0).flatten() + lambda_w * np.ones(2 * d**2)
        grad_vec_a = obj_grad_a.reshape(p, d**2)
        grad_vec_a = np.hstack((grad_vec_a, -grad_vec_a)).flatten() + lambda_a * np.ones(2 * p * d**2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    # Initialize intra-slice and inter-slice matrices
    wa_est = np.zeros(2 * (p + 1) * d**2)
    wa_new = np.zeros(2 * (p + 1) * d**2)

    # Augmented Lagrangian parameters
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

    # Augmented Lagrangian optimization loop
    for n_iter in range(max_iter):
        while (rho < rho_max) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = sopt.minimize(_func, wa_est, method='L-BFGS-B', jac=_grad, bounds=bnds).x
            h_new, _ = _h(_adj(wa_new)[0])
            a,b = _adj(wa_new)
            # print("a: ", b[:5,:5])
            if h_new > 0.25 * h_value:
                rho *= 10

        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value

        if h_value <= h_tol:
            break
        if h_value > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    w_est, a_est = _adj(wa_est)
    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < a_threshold] = 0

    return w_est, a_est


def opt_bnds(X, Xlags, d_vars, tabu_edges: List[Tuple[int, int, int]] = None,
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