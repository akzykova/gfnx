import numpy as np
import jax
import jax.numpy as jnp

from .bitseq import BitseqEnvironment
from .bitseq import EnvState as BitseqEnvState
from gfnx.utils.bitseq import tokenize


class IsingBitseqEnvironment(BitseqEnvironment):
    def __init__(
        self,
        reward_module,
        L: int,
        k: int,
        beta: float,
        gt_samples_path: str | None = None,
    ):
        super().__init__(
            reward_module,
            n=L * L,
            k=k,
        )

        self.L = L
        self.beta = beta

        self.gt_bits = None
        self.gt_pool_size = None

        if gt_samples_path is not None:
            data = np.load(gt_samples_path)

            bits = data["bits"].astype(np.int32)
            file_L = int(data["L"])
            file_beta = float(data["beta"])

            if file_L != L:
                raise ValueError(
                    f"GT file has L={file_L}, but environment has L={L}"
                )

            if not np.isclose(file_beta, beta):
                raise ValueError(
                    f"GT file has beta={file_beta}, but environment has beta={beta}"
                )

            if bits.ndim != 2 or bits.shape[1] != L * L:
                raise ValueError(
                    f"Expected GT bits shape (N, {L * L}), got {bits.shape}"
                )

            if not np.all((bits == 0) | (bits == 1)):
                raise ValueError("GT samples must contain only 0/1 bits")

            self.gt_bits = jnp.asarray(bits, dtype=jnp.int32)
            self.gt_pool_size = bits.shape[0]

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        return self.gt_bits is not None

    def get_ground_truth_sampling(
        self,
        rng_key,
        batch_size,
        env_params,
    ):
        """Sample a fresh batch from the fixed reference GT pool."""

        if self.gt_bits is None:
            raise RuntimeError(
                "gt_samples_path was not provided, so ground-truth "
                "sampling is unavailable."
            )

        if batch_size > self.gt_pool_size:
            raise ValueError(
                f"batch_size={batch_size} cannot exceed "
                f"GT pool size={self.gt_pool_size}"
            )

        # Fresh random subset every call.
        indices = jax.random.choice(
            rng_key,
            self.gt_pool_size,
            shape=(batch_size,),
            replace=False,
        )

        bits = self.gt_bits[indices]

        # {0,1} bits -> bitseq tokens expected by BitseqEnvState.
        tokens = jax.vmap(
            lambda x: tokenize(x, self.k)
        )(bits)

        return BitseqEnvState(
            tokens=tokens,
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