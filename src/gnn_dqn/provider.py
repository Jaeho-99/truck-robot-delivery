"""Instance providers for training and testing.

The final instance generator (Ulsan Nam-gu based, congestion
scenarios) is a separate task; until then the training distribution is
drawn from the existing scaling generator (src.instance): master pools
with different seeds, customer counts sampled from a configured list.
The test set (master seed 1) is never used for training.
"""

import random

from .. import instance
from ..heuristics import Params

TEST_MASTER_SEED = 1


class ScalingInstanceProvider:
    """Samples Params from pre-built master pools (training) and
    exposes the master-seed-1 instances as the test set."""

    def __init__(self, train_master_seeds=(2, 3, 4),
                 sizes=(5, 10, 15, 20), num_trucks=5, num_robots=3,
                 num_parking_copies=2, beta_robot=3, seed=0):
        self.sizes = list(sizes)
        self.kw = dict(num_trucks=num_trucks, num_robots=num_robots,
                       num_parking_copies=num_parking_copies,
                       beta_robot=beta_robot)
        self.rng = random.Random(seed)
        self.masters = [instance.build_master(s)
                        for s in train_master_seeds]

    def _params(self, master, n):
        inst, e_c, l_c = instance.build_scaling_instance(
            master, n, **self.kw)
        return Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])

    def sample(self):
        return self._params(self.rng.choice(self.masters),
                            self.rng.choice(self.sizes))

    def test_set(self, sizes=None):
        master = instance.build_master(TEST_MASTER_SEED)
        return [(f"n{n}_s{TEST_MASTER_SEED}", self._params(master, n))
                for n in (sizes or self.sizes)]
