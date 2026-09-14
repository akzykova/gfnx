"""Evaluation metrics for the GFNx Ising-Bitseq experiment.

The target-side evaluation pool is fixed outside these metric modules.  Each
metric module samples a fresh subset from that pool on every evaluation round,
so one can use a large fixed SW pool (typically 16384 samples) while evaluating
on smaller batches (typically 2048 total, or smaller per round).

The physics metrics follow the reference implementation used by the authors:
  - magnetization error
  - two-point correlation error
  - Sinkhorn distance with Hamming cost

The EUBO module performs the GFNx backward-rollout estimator on a sampled
subset of the fixed reference-state pool.
"""

from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp

from gfnx.base import TEnvParams
from gfnx.metrics.base import (
    BaseMetricsModule,
    BaseProcessArgs,
    EmptyInitArgs,
    EmptyUpdateArgs,
    MetricsState,
)
from gfnx.utils.bitseq import detokenize
from gfnx.utils.rollout import (
    TPolicyFn,
    TPolicyParams,
    backward_rollout,
    backward_trajectory_log_probs,
    forward_rollout,
)


# -----------------------------------------------------------------------------
# Ising physics helpers
# -----------------------------------------------------------------------------


def _corr_curve(spins: chex.Array, axis: int) -> chex.Array:
    """Connected two-point correlation C(r) for a batch of Ising spins.

    This mirrors Ising2D.two_point_correlation() in the reference repository:
    for every distance r, compute

        <S_i S_{i+r}> - <S_i><S_{i+r}>

    averaged over all sites. Periodic boundary conditions are implemented by
    ``jnp.roll``.

    Args:
        spins: Array of shape (batch, L, L), values in {-1, +1}.
        axis: 1 for vertical/row direction, 2 for horizontal/column direction.

    Returns:
        Array of shape (L,) containing the correlation for r = 0, ..., L-1.
    """
    L = spins.shape[axis]
    mean_spins = jnp.mean(spins, axis=0)  # (L, L)
    mean_axis = axis - 1  # remove batch dimension

    def at_shift(r: int) -> chex.Array:
        shifted = jnp.roll(spins, -r, axis=axis)
        shifted_mean = jnp.roll(mean_spins, -r, axis=mean_axis)
        pair_mean = jnp.mean(spins * shifted, axis=0)
        return jnp.mean(pair_mean - mean_spins * shifted_mean)

    return jnp.stack([at_shift(r) for r in range(L)])


def sinkhorn_distance(
    x: chex.Array,
    y: chex.Array,
    reg: float = 1e-3,
    n_iters: int = 100,
) -> chex.Array:
    """Entropy-regularized OT distance with Hamming ground cost.

    This implements the same uniform empirical measures and Hamming cost as
    the reference ``eval_metrics.sinkhorn_distance``. Inputs are binary
    vectors of shape (n, d) and (m, d).

    The implementation is intentionally written in log-domain form for
    numerical stability. It is intended for the per-round subsets used by
    ``IsingPhysicsMetricsModule``; do not call it on very large 16384 x 16384
    sets unless the batch is chunked externally.
    """
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)

    x_sum = jnp.sum(x, axis=-1)
    y_sum = jnp.sum(y, axis=-1)
    cost = x_sum[:, None] + y_sum[None, :] - 2.0 * (x @ y.T)

    n = cost.shape[0]
    m = cost.shape[1]
    log_mu = -jnp.log(jnp.asarray(n, dtype=cost.dtype))
    log_nu = -jnp.log(jnp.asarray(m, dtype=cost.dtype))

    def body(_, carry):
        f, g = carry
        f = reg * (
            log_mu
            - jax.scipy.special.logsumexp(
                (g[None, :] - cost) / reg,
                axis=1,
            )
        )
        g = reg * (
            log_nu
            - jax.scipy.special.logsumexp(
                (f[:, None] - cost) / reg,
                axis=0,
            )
        )
        return f, g

    f, g = jax.lax.fori_loop(
        0,
        n_iters,
        body,
        (jnp.zeros(n, dtype=cost.dtype), jnp.zeros(m, dtype=cost.dtype)),
    )

    plan = jnp.exp((f[:, None] + g[None, :] - cost) / reg)
    return jnp.sum(plan * cost)


# -----------------------------------------------------------------------------
# Physics metrics
# -----------------------------------------------------------------------------


@chex.dataclass
class IsingPhysicsMetricState(MetricsState):
    mag_error: jnp.ndarray
    corr_error: jnp.ndarray
    sinkhorn: jnp.ndarray


class IsingPhysicsMetricsModule(BaseMetricsModule):
    """Magnetization error, two-point correlation error, and Sinkhorn.

    ``gt_spins`` is a fixed reference pool, normally 16384 SW samples.  On
    every evaluation round a fresh subset of size ``batch_size`` is drawn from
    that pool and used for *all three* metrics in that round. The round values
    are then averaged over ``n_rounds``.

    This matches the reference metric definitions. In particular, for the
    Ising setup used here (h=0), the reference magnetization target is exactly
    zero; it is NOT estimated from the sampled GT batch.
    """

    def __init__(
        self,
        env,
        L: int,
        k: int,
        fwd_policy_fn: TPolicyFn,
        gt_spins: chex.Array,
        n_rounds: int,
        batch_size: int,
        sinkhorn_reg: float = 1e-3,
        sinkhorn_iters: int = 100,
    ):
        self.env = env
        self.L = L
        self.k = k
        self.fwd_policy_fn = fwd_policy_fn
        self.gt_spins = gt_spins
        self.pool_size = gt_spins.shape[0]
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters

        if self.batch_size > self.pool_size:
            raise ValueError(
                f"batch_size={self.batch_size} cannot exceed GT pool size={self.pool_size}"
            )

    InitArgs = EmptyInitArgs

    def init(
        self,
        rng_key: chex.PRNGKey,
        args: InitArgs,
    ) -> IsingPhysicsMetricState:
        del rng_key, args
        return IsingPhysicsMetricState(
            mag_error=jnp.array(jnp.inf, dtype=jnp.float32),
            corr_error=jnp.array(jnp.inf, dtype=jnp.float32),
            sinkhorn=jnp.array(jnp.inf, dtype=jnp.float32),
        )

    UpdateArgs = EmptyUpdateArgs

    def update(self, metrics_state, rng_key, args=None):
        del rng_key, args
        return metrics_state

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
        self,
        metrics_state: IsingPhysicsMetricState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> IsingPhysicsMetricState:

        def process_round(carry_rng_key, _):
            rng_key, gt_key, model_key = jax.random.split(carry_rng_key, 3)

            # Fresh reference subset from the fixed SW pool.
            gt_idx = jax.random.choice(
                gt_key,
                self.pool_size,
                shape=(self.batch_size,),
                replace=False,
            )
            gt_spins = self.gt_spins[gt_idx]

            # Fresh model samples of the same size.
            _, aux_info = forward_rollout(
                rng_key=model_key,
                num_envs=self.batch_size,
                policy_fn=self.fwd_policy_fn,
                policy_params=args.policy_params,
                env=self.env,
                env_params=args.env_params,
            )

            # final_env_state is the canonical final state returned by GFNx's
            # forward_rollout API. It is not an ad-hoc reconstruction.
            final_state = aux_info["final_env_state"]
            tokens = final_state.tokens
            bits = jax.vmap(lambda t: detokenize(t, self.k))(tokens)
            spins = (
                2.0 * bits.astype(jnp.float32) - 1.0
            ).reshape(self.batch_size, self.L, self.L)

            # ------------------------------------------------------------
            # Magnetization error.
            # Reference Ising2D uses zero target magnetization for h == 0.
            # ------------------------------------------------------------
            model_mean = jnp.mean(spins, axis=0)
            row_model = jnp.mean(model_mean, axis=1)
            col_model = jnp.mean(model_mean, axis=0)

            # This experiment has h=0, so the reference target is zero.
            row_gt = jnp.zeros_like(row_model)
            col_gt = jnp.zeros_like(col_model)

            mag_error = (
                jnp.sum(jnp.abs(row_model - row_gt))
                + jnp.sum(jnp.abs(col_model - col_gt))
            ) / (2.0 * self.L)

            # ------------------------------------------------------------
            # Two-point correlation error.
            # Reference divides the combined row+column error by 4L.
            # ------------------------------------------------------------
            row_model_corr = _corr_curve(spins, axis=1)
            col_model_corr = _corr_curve(spins, axis=2)
            row_gt_corr = _corr_curve(gt_spins, axis=1)
            col_gt_corr = _corr_curve(gt_spins, axis=2)

            corr_error = (
                jnp.sum(jnp.abs(row_model_corr - row_gt_corr))
                + jnp.sum(jnp.abs(col_model_corr - col_gt_corr))
            ) / (4.0 * self.L)

            # ------------------------------------------------------------
            # Sinkhorn distance between the two binary sample batches.
            # Reference uses Hamming cost and epsilon=1e-3.
            # ------------------------------------------------------------
            model_bits = ((spins + 1.0) / 2.0).reshape(self.batch_size, -1)
            gt_bits = ((gt_spins + 1.0) / 2.0).reshape(self.batch_size, -1)

            sinkhorn = sinkhorn_distance(
                gt_bits,
                model_bits,
                reg=self.sinkhorn_reg,
                n_iters=self.sinkhorn_iters,
            )

            return rng_key, (mag_error, corr_error, sinkhorn)

        _, (mag_rounds, corr_rounds, sinkhorn_rounds) = jax.lax.scan(
            process_round,
            rng_key,
            None,
            length=self.n_rounds,
        )

        return metrics_state.replace(
            mag_error=jnp.mean(mag_rounds),
            corr_error=jnp.mean(corr_rounds),
            sinkhorn=jnp.mean(sinkhorn_rounds),
        )


# -----------------------------------------------------------------------------
# EUBO
# -----------------------------------------------------------------------------


@chex.dataclass
class IsingEUBOMetricState(MetricsState):
    eubo: jnp.ndarray


class IsingEUBOMetricsModule(BaseMetricsModule):
    """EUBO evaluated from a fixed reference-state pool.

    ``gt_states`` must contain terminal BitseqEnvState objects corresponding to
    the same fixed SW reference pool as the physics metrics. Each round draws a
    fresh subset from that pool, then runs the GFNx backward rollout from those
    terminal states. The per-round EUBO estimates are averaged.
    """

    def __init__(
        self,
        env,
        bwd_policy_fn: TPolicyFn,
        gt_states,
        n_rounds: int,
        batch_size: int,
    ):
        self.env = env
        self.bwd_policy_fn = bwd_policy_fn
        self.gt_states = gt_states
        self.pool_size = gt_states.tokens.shape[0]
        self.n_rounds = n_rounds
        self.batch_size = batch_size

        if self.batch_size > self.pool_size:
            raise ValueError(
                f"batch_size={self.batch_size} cannot exceed GT pool size={self.pool_size}"
            )

        if env.is_normalizing_constant_tractable:
            self.logZ = jnp.log(env.get_normalizing_constant(None))
        else:
            self.logZ = jnp.array(0.0, dtype=jnp.float32)

    InitArgs = EmptyInitArgs

    def init(self, rng_key, args):
        del rng_key, args
        return IsingEUBOMetricState(
            eubo=jnp.array(jnp.inf, dtype=jnp.float32)
        )

    UpdateArgs = EmptyUpdateArgs

    def update(self, metrics_state, rng_key, args=None):
        del rng_key, args
        return metrics_state

    def get(self, metrics_state: IsingEUBOMetricState):
        return {"eubo": metrics_state.eubo}

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        policy_params: TPolicyParams
        env_params: TEnvParams

    def process(self, metrics_state, rng_key, args):
        def process_round(carry_rng_key, _):
            rng_key, gt_key, rollout_key = jax.random.split(carry_rng_key, 3)

            # Fresh reference subset: analogous to target.cached_sample(batch_size)
            # from a fixed larger cache.
            idx = jax.random.choice(
                gt_key,
                self.pool_size,
                shape=(self.batch_size,),
                replace=False,
            )
            round_states = jax.tree_util.tree_map(
                lambda x: x[idx],
                self.gt_states,
            )

            bwd_traj_data, _ = backward_rollout(
                rng_key=rollout_key,
                init_state=round_states,
                policy_fn=self.bwd_policy_fn,
                policy_params=args.policy_params,
                env=self.env,
                env_params=args.env_params,
            )

            log_rewards = self.env.reward_module.log_reward(
                round_states,
                args.env_params,
            )
            log_pf_traj, log_pb_traj = backward_trajectory_log_probs(
                self.env,
                bwd_traj_data,
                args.env_params,
            )

            eubo = log_pb_traj - log_pf_traj + log_rewards
            chex.assert_shape(eubo, (self.batch_size,))

            return rng_key, eubo

        _, eubo_per_round = jax.lax.scan(
            process_round,
            rng_key,
            None,
            length=self.n_rounds,
        )

        eubo = jnp.mean(eubo_per_round) - self.logZ
        return metrics_state.replace(eubo=eubo)
