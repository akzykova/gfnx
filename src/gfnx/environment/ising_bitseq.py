import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp

from .bitseq import BitseqEnvironment
from .bitseq import EnvState as BitseqEnvState
from gfnx.utils.bitseq import tokenize
from gfnx.utils import wolff_sampler

from gfnx.utils.bitseq import tokenize

class IsingBitseqEnvironment(BitseqEnvironment):
    """BitseqEnvironment (k-bit-packed encoding of an L x L torus Ising lattice)
    with ground-truth sampling wired to the Wolff cluster algorithm, so that
    gfnx.metrics.EUBOMetricsModule works out of the box."""

    def __init__(
        self,
        reward_module,
        L: int,
        k: int,
        beta: float,
        wolff_burn_in: int = 2000,
        wolff_sweeps_per_sample: int = 5,
    ):
        super().__init__(reward_module, n=L * L, k=k)
        self.L = L
        self.beta = beta
        self.wolff_burn_in = wolff_burn_in
        self.wolff_sweeps_per_sample = wolff_sweeps_per_sample

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        return True

    def get_ground_truth_sampling(self, rng_key, batch_size, env_params):
        gt_spins, _, _ = wolff_sampler(
            key=rng_key,
            N=self.L,
            sigma=1.0,
            alpha=self.beta / 2.0,
            num_samples=batch_size,
            burn_in=self.wolff_burn_in,
            sweeps_per_sample=self.wolff_sweeps_per_sample,
        )
        gt_bits = ((gt_spins + 1) // 2).reshape(batch_size, -1).astype(jnp.int32)
        gt_tokens = jax.vmap(lambda b: tokenize(b, self.k))(gt_bits)
        return BitseqEnvState(
            tokens=gt_tokens,
            is_terminal=jnp.ones((batch_size,), dtype=jnp.bool),
            is_initial=jnp.zeros((batch_size,), dtype=jnp.bool),
            is_pad=jnp.zeros((batch_size,), dtype=jnp.bool),
        )
