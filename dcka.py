# Deconfounded Centered Kernel Alignment (dCKA)
# 'https://github.com/ykumards/simtorch/tree/main'

import torch

remove_negative_eig: bool = True


def remove_negative_eigenvalues(A):
    L, V = torch.linalg.eigh(A)
    L[L < 0] = 0

    return V @ torch.diag_embed(L) @ V.T


def normalize(X):
    X = X - X.mean(0, keepdims=True)
    return X / torch.linalg.norm(X)


def _centering(K: torch.tensor, device):
    n = K.shape[0]
    unit = torch.ones([n, n], device=device)
    identity = torch.eye(n, device=device)
    H = identity - unit / n

    return torch.matmul(torch.matmul(H, K), H)


def _linear_residual(input_y, input_dist):
    input_y = input_y.view(-1, 1)

    inv_term = torch.inverse(input_dist.T @ input_dist)
    beta = inv_term @ input_dist.T @ input_y
    y_pred = input_dist @ beta

    res_sim = (input_y - y_pred).view(-1)
    pve = (1 - res_sim.var() / input_y.var()).item()

    return res_sim, pve


def linear_HSIC(
    L_X: torch.tensor,
    L_Y: torch.tensor,
    input_confounders: torch.tensor,
    device,
):
    input_grammian = (
        torch.matmul(input_confounders, input_confounders.T).view(-1, 1)
    )
    input_distance_1 = torch.cat(
        (torch.ones(input_grammian.shape[0], 1, device=device), input_grammian), 1
    )

    input_distance_2 = torch.cat(
        (torch.ones(input_grammian.shape[0], 1, device=device), input_grammian), 1
    )

    residuals1, pve_1 = _linear_residual(L_X, input_distance_1)
    residuals2, pve_2 = _linear_residual(L_Y, input_distance_2)

    residuals1 = residuals1.view(L_X.shape)
    residuals2 = residuals2.view(L_Y.shape)
    if remove_negative_eig:
        residuals1 = remove_negative_eigenvalues(residuals1)
        residuals2 = remove_negative_eigenvalues(residuals2)

    return (torch.sum(_centering(residuals1, device) * _centering(residuals2, device)), pve_1, pve_2)


def linear_CKA(
    L_X: torch.tensor,
    L_Y: torch.tensor,
    input_confounders: torch.tensor,
    device,
):
    input_confounders = normalize(
        input_confounders.to(device))

    hsic, pve_1, pve_2 = linear_HSIC(L_X, L_Y, input_confounders, device)
    var1 = torch.sqrt(linear_HSIC(L_X, L_X, input_confounders)[0], device)
    var2 = torch.sqrt(linear_HSIC(L_Y, L_Y, input_confounders)[0], device)

    # return ((hsic) / ((var1 * var2))).detach().cpu(), pve_1, pve_2
    return ((hsic + 1e-15) / ((var1 * var2) + 1e-15)).detach().cpu(), pve_1, pve_2