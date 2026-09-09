from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp

import gfnx
from gfnx.base import TEnvParams
from gfnx.metrics.base import (
    BaseMetricsModule,
    BaseProcessArgs,
    EmptyInitArgs,
    EmptyUpdateArgs,
    MetricsState,
)
from gfnx.utils.bitseq import detokenize, tokenize
from gfnx.utils.rollout import TPolicyFn, TPolicyParams, forward_rollout

# --------------------------------------------------------------------------------
# 3, 4, 5. Sinkhorn distance, magnetisation error, 2-point correlation error
# --------------------------------------------------------------------------------
def _corr_curve(spins: chex.Array, axis: int) -> chex.Array:
    """C_row/C_col(r) from §C.1.3: 2-point correlation averaged over all sites, as a
    function of lattice distance r along `axis` (1=row-direction, 2=col-direction).
    `spins`: (batch, L, L) in {-1, +1}."""
    L = spins.shape[axis]
    M = jnp.mean(spins, axis=0)  # (L, L)
    m_axis = axis - 1  # M has no batch dimension

    def at_shift(r):
        shifted = jnp.roll(spins, -r, axis=axis)
        shifted_M = jnp.roll(M, -r, axis=m_axis)
        pair_mean = jnp.mean(spins * shifted, axis=0)
        return jnp.mean(pair_mean - M * shifted_M)

    return jnp.stack([at_shift(r) for r in range(L)])


def sinkhorn_distance(x: chex.Array, y: chex.Array, reg: float = 0.05, n_iters: int = 200) -> chex.Array:
    """Entropy-regularised OT distance (Cuturi, 2013) with Hamming ground cost,
    computed in log domain for numerical stability. x: (n, d), y: (m, d), binary.

    Note: the paper uses reg=0.001; with Hamming costs up to L*L that is extremely
    peaked and needs many more Sinkhorn iterations to converge without under/overflow.
    Start with reg=0.05-0.1 to get something stable and tighten later if you need to
    match their exact numbers.
    """
    x_sum = jnp.sum(x, axis=-1)
    y_sum = jnp.sum(y, axis=-1)
    cost = x_sum[:, None] + y_sum[None, :] - 2.0 * (x @ y.T)  # (n, m)
    n, m = cost.shape
    log_mu = -jnp.log(n) * jnp.ones(n)
    log_nu = -jnp.log(m) * jnp.ones(m)

    def body(_, carry):
        f, g = carry
        f = reg * (log_mu - jax.scipy.special.logsumexp((g[None, :] - cost) / reg, axis=1))
        g = reg * (log_nu - jax.scipy.special.logsumexp((f[:, None] - cost) / reg, axis=0))
        return f, g

    f, g = jax.lax.fori_loop(0, n_iters, body, (jnp.zeros(n), jnp.zeros(m)))
    plan = jnp.exp((f[:, None] + g[None, :] - cost) / reg)
    return jnp.sum(plan * cost)


@chex.dataclass
class IsingPhysicsMetricState(MetricsState):
    mag_error: jnp.ndarray
    corr_error: jnp.ndarray
    sinkhorn: jnp.ndarray


class IsingPhysicsMetricsModule(BaseMetricsModule):
    """Magnetisation error, 2-point correlation error, and Sinkhorn distance between
    on-policy GFlowNet samples and ground-truth Wolff samples -- the Mag./Corr./Sink.
    columns of Table 1 in the paper.

    Ground-truth reference statistics (M_row_true, M_col_true, C_row_true, C_col_true,
    and the raw GT spins for Sinkhorn) are computed once at construction time from a
    fixed batch of Wolff samples, exactly like EUBOMetricsModule's fixed test set.
    """

    def __init__(
        self,
        env,
        L: int,
        k: int,
        fwd_policy_fn: TPolicyFn,
        gt_spins: chex.Array,  # (n_gt, L, L) in {-1, +1}, from gfnx.utils.wolff_sampler
        n_rounds: int,
        batch_size: int,
        sinkhorn_reg: float = 0.05,
        sinkhorn_iters: int = 200,
    ):
        self.env = env
        self.L = L
        self.k = k
        self.fwd_policy_fn = fwd_policy_fn
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters

        self.gt_spins = gt_spins
        M = jnp.mean(gt_spins, axis=0)
        self.M_row_true = jnp.mean(M, axis=1)
        self.M_col_true = jnp.mean(M, axis=0)
        self.C_row_true = _corr_curve(gt_spins, axis=1)
        self.C_col_true = _corr_curve(gt_spins, axis=2)

    InitArgs = EmptyInitArgs

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> IsingPhysicsMetricState:
        return IsingPhysicsMetricState(
            mag_error=jnp.array(jnp.inf, dtype=jnp.float32),
            corr_error=jnp.array(jnp.inf, dtype=jnp.float32),
            sinkhorn=jnp.array(jnp.inf, dtype=jnp.float32),
        )

    UpdateArgs = EmptyUpdateArgs

    def update(self, metrics_state, rng_key, args=None):
        return metrics_state  # computed on demand in process(), like ELBO/EUBO

    def get(self, metrics_state: IsingPhysicsMetricState) -> Dict[str, Any]:
        return {
            "mag_error": metrics_state.mag_error,
            "corr_error": metrics_state.corr_error,
            "sinkhorn": metrics_state.sinkhorn,
        }

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        policy_params: TPolicyParams
        env_params: TEnvParams

    def process(
        self, metrics_state: IsingPhysicsMetricState, rng_key: chex.PRNGKey, args: ProcessArgs
    ) -> IsingPhysicsMetricState:
        def process_round(carry_rng_key, _):
            rng_key, roll_key = jax.random.split(carry_rng_key)
            _, aux_info = forward_rollout(
                rng_key=roll_key,
                num_envs=self.batch_size,
                policy_fn=self.fwd_policy_fn,
                policy_params=args.policy_params,
                env=self.env,
                env_params=args.env_params,
            )
            tokens = aux_info["final_env_state"].tokens  # (batch_size, max_length)
            bits = jax.vmap(lambda t: detokenize(t, self.k))(tokens)
            spins = (2.0 * bits.astype(jnp.float32) - 1.0).reshape(self.batch_size, self.L, self.L)
            return rng_key, spins

        _, spins_per_round = jax.lax.scan(process_round, rng_key, None, length=self.n_rounds)
        spins = spins_per_round.reshape(-1, self.L, self.L)  # (n_rounds * batch_size, L, L)

        M = jnp.mean(spins, axis=0)
        M_row = jnp.mean(M, axis=1)
        M_col = jnp.mean(M, axis=0)
        mag_error = (
            jnp.sum(jnp.abs(M_row - self.M_row_true)) + jnp.sum(jnp.abs(M_col - self.M_col_true))
        ) / (2 * self.L)

        C_row = _corr_curve(spins, axis=1)
        C_col = _corr_curve(spins, axis=2)
        corr_error = (
            jnp.sum(jnp.abs(C_row - self.C_row_true)) + jnp.sum(jnp.abs(C_col - self.C_col_true))
        ) / (2 * self.L)

        model_bits = ((spins + 1) / 2).reshape(spins.shape[0], -1)
        gt_bits = ((self.gt_spins + 1) / 2).reshape(self.gt_spins.shape[0], -1)
        sinkhorn = sinkhorn_distance(model_bits, gt_bits, reg=self.sinkhorn_reg, n_iters=self.sinkhorn_iters)

        return metrics_state.replace(mag_error=mag_error, corr_error=corr_error, sinkhorn=sinkhorn)
