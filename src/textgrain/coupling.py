"""Block coupling of textGrain: Sinkhorn with KL regularization calibrated to an entropy budget."""

from dataclasses import dataclass

import numpy as np

EULER_GAMMA = 0.5772156649015329
LAMBDA_RANGE = (1e-2, 1e2)


@dataclass
class Coupling:
    """Coupling pi (B, m) between blocks and columns, with the fraction of entropy it removes and its regularization."""

    pi: np.ndarray
    beta: float
    lam: float


def entropy(probs: np.ndarray) -> float:
    """Takes a distribution and returns its entropy in nats."""
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def block_probs(probs: np.ndarray, blocks: np.ndarray, n_blocks: int) -> np.ndarray:
    """Takes a token distribution and the block of each token and returns the probability of each block, rho (B,)."""
    return np.bincount(blocks, weights=probs, minlength=n_blocks)


def standardized_gumbel(uniforms: np.ndarray) -> np.ndarray:
    """Takes uniforms U and returns G = (Z - gamma) / (pi / sqrt(6)) with Z = -ln(-ln U)."""
    return (-np.log(-np.log(uniforms)) - EULER_GAMMA) / (np.pi / np.sqrt(6))


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    top = x.max(axis=axis, keepdims=True)
    return top + np.log(np.exp(x - top).sum(axis=axis, keepdims=True))


def sinkhorn(rho: np.ndarray, gains: np.ndarray, lam: float, n_iter: int = 300, tol: float = 1e-8) -> np.ndarray:
    """Takes positive block probabilities rho (B,), gains G (B, m) and lambda and returns the coupling maximizing <pi, G> - lam * KL(pi || rho x uniform)."""
    m = gains.shape[1]
    log_rho = np.log(rho)[:, None]
    log_pi = log_rho - np.log(m) + gains / lam
    for _ in range(n_iter):
        log_pi += log_rho - _logsumexp(log_pi, axis=1)
        log_pi += -np.log(m) - _logsumexp(log_pi, axis=0)
        if np.abs(np.exp(_logsumexp(log_pi, axis=1)) - np.exp(log_rho)).sum() < tol:
            break
    return np.exp(log_pi)


def correct_marginals(table: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Takes a nonnegative table and target marginals and returns the table with exact marginals (rounding of Altschuler et al., 2017)."""
    row_sums = table.sum(axis=1)
    x = np.where(row_sums > 0, np.minimum(rows / np.where(row_sums > 0, row_sums, 1), 1), 1)
    table = table * x[:, None]
    column_sums = table.sum(axis=0)
    y = np.where(column_sums > 0, np.minimum(columns / np.where(column_sums > 0, column_sums, 1), 1), 1)
    table = table * y[None, :]
    row_deficit = rows - table.sum(axis=1)
    column_deficit = columns - table.sum(axis=0)
    missing = row_deficit.sum()
    return table + np.outer(row_deficit, column_deficit) / missing if missing > 0 else table


def kl_from_independence(pi: np.ndarray, rho: np.ndarray) -> float:
    """Takes a coupling (B, m) and its row marginal rho and returns KL(pi || rho x uniform), the entropy it removes on average."""
    m = pi.shape[1]
    reference = rho[:, None] / m * np.ones_like(pi)
    mask = pi > 0
    return float((pi[mask] * np.log(pi[mask] / reference[mask])).sum())


def couple(rho: np.ndarray, gains: np.ndarray, token_entropy: float, beta: float, n_bisection: int = 40, rel_tol: float = 1e-3) -> Coupling:
    """Takes block probabilities, gains G, the entropy H(P) and a budget beta and returns the coupling removing beta * H(P) nats, up to rel_tol."""
    m = gains.shape[1]
    independent = Coupling(np.outer(rho, np.ones(m) / m), 0.0, np.inf)
    active = rho > 0
    rho_active, gains_active = rho[active], gains[active]
    block_entropy = entropy(rho_active)
    if beta <= 0 or token_entropy <= 0 or block_entropy <= 0:
        return independent

    def solve(lam: float) -> np.ndarray:
        return correct_marginals(sinkhorn(rho_active, gains_active, lam), rho_active, np.ones(m) / m)

    target = min(beta * token_entropy, (1 - 1e-8) * block_entropy, np.log(m))
    low, high = np.log(LAMBDA_RANGE[0]), np.log(LAMBDA_RANGE[1])
    pi_active = solve(LAMBDA_RANGE[0])
    kl = kl_from_independence(pi_active, rho_active)
    if kl <= target:
        return _embed(pi_active, active, kl / token_entropy, LAMBDA_RANGE[0])
    for _ in range(n_bisection):
        middle = (low + high) / 2
        pi_active = solve(np.exp(middle))
        kl = kl_from_independence(pi_active, rho_active)
        if abs(kl - target) <= rel_tol * target:
            break
        if kl > target:
            low = middle
        else:
            high = middle
    return _embed(pi_active, active, kl / token_entropy, float(np.exp(middle)))


def _embed(pi_active: np.ndarray, active: np.ndarray, beta: float, lam: float) -> Coupling:
    pi = np.zeros((len(active), pi_active.shape[1]))
    pi[active] = pi_active
    return Coupling(pi, beta, lam)


def token_probs(probs: np.ndarray, blocks: np.ndarray, rho: np.ndarray, pi: np.ndarray, column: int) -> np.ndarray:
    """Takes P, the partition, rho, the coupling and a column j and returns Q(w | j) = m * pi(g(w), j) * P_w / rho_g(w)."""
    m = pi.shape[1]
    block_given_column = m * pi[:, column]
    ratio = np.divide(block_given_column, rho, out=np.zeros_like(rho), where=rho > 0)
    return probs * ratio[blocks]
