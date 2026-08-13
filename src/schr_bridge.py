import math

import geotorch
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class Schr_Bridge(nn.Module):
    """Schrödinger bridge with an Ornstein-Uhlenbeck reference process.

    The shape matrices ``S_k`` of the Gaussian potentials are parameterised so
    that ``S_k - S_lower_bound * I`` is positive definite at every optimisation
    step: the eigenvalues are ``S_lower_bound + exp(.)`` and, in the
    non-diagonal case, the eigenvectors are kept orthogonal by ``geotorch``.
    The default ``S_lower_bound=1`` constrains the potentials to ``S_k > I``;
    ``S_lower_bound=0`` recovers the plain constraint ``S_k > 0``.

    The bound is one-sided by construction, but ``get_log_C_tilde`` also needs
    the Gaussian integral to converge, i.e. ``A_k > 0``, which for the OU
    reference means every eigenvalue of ``S_k`` must stay below
    ``sigma_T^2 * exp(2 * b * T)``. That ceiling is astronomically large for the
    ``(b, T)`` used in the experiments, so a lower bound of 1 is well inside it.
    """

    def __init__(self, dim=2, n_potentials=5, epsilon=1, is_diagonal=True,
                 sampling_batch_size=1, S_diagonal_init=1.1, b=0.0, m=0.0, T=1.0,
                 S_lower_bound=1.0):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials
        self.sampling_batch_size = sampling_batch_size
        self.T = T

        if not S_diagonal_init > S_lower_bound:
            raise ValueError(
                f"S_diagonal_init={S_diagonal_init} must be strictly greater than "
                f"S_lower_bound={S_lower_bound}: the initial S_k has to satisfy the "
                "constraint it is optimised under."
            )

        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer("S_lower_bound", torch.tensor(float(S_lower_bound)))

        self.log_alpha_raw = nn.Parameter(
            self.epsilon * torch.log(torch.ones(n_potentials) / n_potentials)
        )
        self.r = nn.Parameter(torch.randn(n_potentials, dim))
        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log((S_diagonal_init - S_lower_bound) * torch.ones(n_potentials, self.dim))
        )
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(n_potentials, self.dim, self.dim)
        )
        geotorch.orthogonal(self, "S_rotation_matrix")

    def init_r_by_samples(self, samples):
        assert samples.shape[0] == self.r.shape[0]
        self.r.data = torch.clone(samples.to(self.r.device))

    def get_r(self):
        return self.r

    def get_S(self):
        # eigenvalues of S_k, bounded below by S_lower_bound; in the non-diagonal
        # branch the rotation is orthogonal, so S_k - S_lower_bound * I keeps them
        S_eigenvalues = self.S_lower_bound + torch.exp(self.S_log_diagonal_matrix)
        if self.is_diagonal:
            S = S_eigenvalues
        else:
            S = ((self.S_rotation_matrix * S_eigenvalues[:, None, :])
                 @ torch.permute(self.S_rotation_matrix, (0, 2, 1)))
        return S

    def get_log_alpha(self):
        return (1 / self.epsilon) * self.log_alpha_raw

    def ou_sigma(self, b, t=1.0):
        bt = b * t
        if abs(bt) < 1e-1:
            return t * (1 - bt + (2/3) * bt**2 - (1/3) * bt**3 + (2/15) * bt**4)
        else:
            return (1 - torch.exp(-2*b*t)) / (2*b)

    @torch.no_grad()
    def forward(self, x):
        S = self.get_S()
        r = self.get_r()
        b = self.b
        m = self.m
        epsilon = self.epsilon

        sigma_sq = self.ou_sigma(b, self.T)
        log_alpha = self.get_log_alpha()
        decay_factor = torch.exp(-b * self.T)

        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size // sampling_batch_size if batch_size % sampling_batch_size == 0
            else (batch_size // sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]

            m_x = (1 - decay_factor) * m + decay_factor * sub_batch_x

            if self.is_diagonal:
                x_S_x = (m_x[:, None, :] * S[None, :, :] * m_x[:, None, :]).sum(dim=-1)
                x_r = (m_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + S[None, :] * m_x[:, None, :] / sigma_sq
            else:
                x_S_x = (m_x[:, None, None, :] @ (S[None, :, :, :] @ m_x[:, None, :, None]))[:, :, 0, 0]
                x_r = (m_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + (S[None, :, :, :] @ m_x[:, None, :, None])[:, :, :, 0] / sigma_sq

            exp_argument = (x_S_x / sigma_sq + 2 * x_r) / (2 * epsilon * sigma_sq) + log_alpha[None, :]

            exp_argument = torch.nan_to_num(exp_argument)
            r_x = torch.nan_to_num(r_x)
            S = torch.nan_to_num(S)

            mix = Categorical(logits=exp_argument)
            if self.is_diagonal:
                comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon * S)[None, :, :]), 1)
            else:
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon * S)
            gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        return torch.cat(samples, dim=0)

    def get_log_potential(self, x):
        S = self.get_S()
        r = self.get_r()
        log_alpha = self.get_log_alpha()
        epsilon = self.epsilon

        mix = Categorical(logits=log_alpha)
        if self.is_diagonal:
            comp = Independent(Normal(loc=r, scale=torch.sqrt(epsilon * S)), 1)
        else:
            comp = MultivariateNormal(loc=r, covariance_matrix=epsilon * S)
        gmm = MixtureSameFamily(mix, comp)

        return gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)

    def get_log_C_tilde(self, t):
        m = self.m
        epsilon = self.epsilon
        b = self.b
        log_alpha = self.get_log_alpha()
        r = self.get_r()
        S = self.get_S()
        D = self.dim

        m_vec = m.expand(D).float() if m.dim() == 0 else m.float()

        sigma_T_sq = torch.clamp((1 - torch.exp(-2*b*t)) / (2*b), min=1e-8)
        a = 1.0 / (2 * epsilon * sigma_T_sq)

        if self.is_diagonal:
            Sigma_inv = (2.0 / epsilon) * torch.ones(D, device=m.device)
            log_det_Sigma = D * torch.log(epsilon / 2)

            S_k_inv = 1.0 / S
            A_k_diag = b * Sigma_inv[None, :] + (1.0/epsilon) * S_k_inv - 2*a
            b_k = b * Sigma_inv[None, :] * m_vec[None, :] + (1.0/epsilon) * S_k_inv * r
            c_k = (-0.5 * b * torch.sum(m_vec**2 * Sigma_inv)
                   - 0.5 / epsilon * torch.sum(r**2 * S_k_inv, dim=-1))

            A_k_inv_diag = 1.0 / A_k_diag
            log_det_A_k = torch.sum(torch.log(torch.abs(A_k_diag)), dim=-1)
            quadratic_term = 0.5 * torch.sum(b_k**2 * A_k_inv_diag, dim=-1)
            log_det_S_k = torch.sum(torch.log(S), dim=-1)
        else:
            Sigma_inv = ((2.0 / epsilon) * torch.eye(D, device=m.device)).float()
            log_det_Sigma = D * torch.log(epsilon / 2)

            S_k_inv = torch.linalg.inv(S).float()
            A_k = (b * Sigma_inv[None, :, :]
                   + (1.0/epsilon) * S_k_inv
                   - 2*a * torch.eye(D, device=m.device)[None, :, :])
            A_k_inv = torch.linalg.inv(A_k)
            b_k = (b * torch.einsum('ij,j->i', Sigma_inv, m_vec)[None, :]
                   + (1.0/epsilon) * torch.einsum('kij,kj->ki', S_k_inv, r))
            c_k = (-0.5 * b * torch.einsum('i,ij,j->', m_vec, Sigma_inv, m_vec)
                   - 0.5 / epsilon * torch.einsum('ki,kij,kj->k', r, S_k_inv, r))

            log_det_A_k = torch.logdet(A_k)
            quadratic_term = 0.5 * torch.einsum('ki,kij,kj->k', b_k, A_k_inv, b_k)
            log_det_S_k = torch.logdet(S)

        log_prefactor = (log_alpha
                         - 0.5 * D * math.log(2 * math.pi)
                         + 0.5 * D * torch.log(b)
                         - 0.5 * log_det_Sigma
                         - 0.5 * D * math.log(epsilon)
                         - 0.5 * log_det_S_k
                         - 0.5 * log_det_A_k)

        return torch.logsumexp(log_prefactor + quadratic_term + c_k, dim=-1)
