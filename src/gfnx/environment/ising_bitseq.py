import jax
import jax.numpy as jnp

from .bitseq import BitseqEnvironment
from .bitseq import EnvState as BitseqEnvState
from gfnx.utils.bitseq import tokenize
from gfnx.utils import swendsen_wang_sampler


class IsingBitseqEnvironment(BitseqEnvironment):
    """16x16 Ising model using the GFNx BitSeq representation.

    The generative environment remains the standard BitSeq environment.
    Swendsen-Wang is used only for generating ground-truth Ising samples.
    """

    def __init__(
        self,
        reward_module,
        L: int,
        k: int,
        beta: float,
        sw_burn_in: int = 1000,
        sw_sweeps_per_sample: int = 100,
    ):
        super().__init__(
            reward_module,
            n=L * L,
            k=k,
        )

        self.L = L
        self.beta = beta
        self.sw_burn_in = sw_burn_in
        self.sw_sweeps_per_sample = sw_sweeps_per_sample

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        return True

    def get_ground_truth_sampling(
        self,
        rng_key,
        batch_size,
        env_params,
    ):
        """Generate equilibrium Ising samples using Swendsen-Wang."""

        gt_spins, _, _ = swendsen_wang_sampler(
            key=rng_key,
            L=self.L,
            beta=self.beta,
            J=1.0,
            num_samples=batch_size,
            batch_size=256,
            burn_in=self.sw_burn_in,
            collect_every=self.sw_sweeps_per_sample,
        )

        # {-1,+1} -> {0,1}
        gt_bits = (
            (gt_spins + 1) // 2
        ).reshape(batch_size, -1).astype(jnp.int32)

        # Pack 8 bits -> one BitSeq token.
        gt_tokens = jax.vmap(
            lambda bits: tokenize(bits, self.k)
        )(gt_bits)

        return BitseqEnvState(
            tokens=gt_tokens,
            is_terminal=jnp.ones(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            is_initial=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
            is_pad=jnp.zeros(
                (batch_size,),
                dtype=jnp.bool_,
            ),
        )