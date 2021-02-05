import sys

import scipy.signal
import numpy as np

from typing import List

import tensorflow as tf


class SampleBatch:
    OBS = "obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    DONES = "dones"

    RETURNS = "returns"
    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    # Value function predictions emitted by the behaviour policy.
    VF_PREDS = "vf_preds"

    ACTION_DIST_INPUTS = "action_dist_inputs"
    ACTION_PROB = "action_prob"
    ACTION_LOGP = "action_logp"

    # Uniquely identifies an episode.
    EPS_ID = "eps_id"

    def __init__(self, *args, **kwargs):
        self.data = dict(*args, **kwargs)

        lengths = []
        for k, v in self.data.copy().items():
            assert isinstance(k, str), self
            lengths.append(len(v))
            if isinstance(v, list):
                self.data[k] = np.array(v).squeeze()
                if len(self.data[k].shape) == 1:
                    self.data[k] = self.data[k][..., None]
        if not lengths:
            raise ValueError("Empty sample batch")
        assert (
            len(set(lengths)) == 1
        ), "Data columns must be same length, but lens are {}".format(lengths)

        self.count = len(next(iter(self.data.values())))
        self.new_columns = []

    def __getitem__(self, key: str):
        """Returns one column (by key) from the data.

        Args:
            key (str): The key (column name) to return.

        Returns:
            TensorType: The data under the given key.
        """
        return self.data[key]

    def __setitem__(self, key, item) -> None:
        """Inserts (overrides) an entire column (by key) in the data buffer.

        Args:
            key (str): The column name to set a value for.
            item (TensorType): The data to insert.
        """
        if key not in self.data:
            self.new_columns.append(key)
        self.data[key] = item

    def __str__(self):
        return "SampleBatch({})".format(str(self.data))

    def __repr__(self):
        return "SampleBatch({})".format(str(self.data))

    def __iter__(self):
        return self.data.__iter__()

    def __contains__(self, x):
        return x in self.data

    def __len__(self):
        return len(list(self.data.values())[0])

    def concat(self, other: "SampleBatch") -> "SampleBatch":
        """Returns a new SampleBatch with each data column concatenated.

        Args:
            other (SampleBatch): The other SampleBatch object to concat to this
                one.

        Returns:
            SampleBatch: The new SampleBatch, resulting from concating `other`
                to `self`.

        Examples:
            >>> b1 = SampleBatch({"a": [1, 2]})
            >>> b2 = SampleBatch({"a": [3, 4, 5]})
            >>> print(b1.concat(b2))
            {"a": [1, 2, 3, 4, 5]}
        """

        if self.keys() != other.keys():
            raise ValueError(
                "SampleBatches to concat must have same columns! {} vs {}".format(
                    list(self.keys()), list(other.keys())
                )
            )
        out = {}
        for k in self.keys():
            out[k] = np.concatenate([self[k], other[k]])
        return SampleBatch(out)

    @staticmethod
    def concat_samples(samples: List["SampleBatch"]):
        """Concatenates n data dicts or MultiAgentBatches.

        Args:
            samples (List[Dict[TensorType]]]): List of dicts of data (numpy).

        Returns:
            Union[SampleBatch, MultiAgentBatch]: A new (compressed)
                SampleBatch or MultiAgentBatch.
        """
        concat_samples = []
        for s in samples:
            if s.count > 0:
                concat_samples.append(s)

        out = {}
        for k in concat_samples[0].keys():
            out[k] = np.concatenate([s[k] for s in concat_samples], axis=0)
        return SampleBatch(out)

    def keys(self):
        """
        Returns:
            Iterable[str]: The keys() iterable over `self.data`.
        """
        return self.data.keys()

    def items(self):
        """
        Returns:
            Iterable[TensorType]: The values() iterable over `self.data`.
        """
        return self.data.items()

    def get(self, key: str):
        """Returns one column (by key) from the data or None if key not found.

        Args:
            key (str): The key (column name) to return.

        Returns:
            Optional[TensorType]: The data under the given key. None if key
                not found in data.
        """
        return self.data.get(key)


def compute_advantages(
    rollout: SampleBatch,
    last_r: float,
    gamma: float = 0.9,
    lambda_: float = 1.0,
    use_gae: bool = True,
    use_critic: bool = True,
):
    """
    Given a rollout, compute its value targets and the advantages.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory.
        last_r (float): Value estimation for last observation.
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE.
        use_gae (bool): Using Generalized Advantage Estimation.
        use_critic (bool): Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    rollout_size = len(rollout[SampleBatch.ACTIONS])

    assert (
        SampleBatch.VF_PREDS in rollout or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS], np.array([[last_r]])])
        delta_t = rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[SampleBatch.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
        rollout[SampleBatch.VALUE_TARGETS] = (
            rollout[SampleBatch.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS], np.array([last_r])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if use_critic:
            rollout[SampleBatch.ADVANTAGES] = (
                discounted_returns - rollout[SampleBatch.VF_PREDS]
            )
            rollout[SampleBatch.VALUE_TARGETS] = discounted_returns
        else:
            rollout[SampleBatch.ADVANTAGES] = discounted_returns
            rollout[SampleBatch.VALUE_TARGETS] = np.zeros_like(
                rollout[SampleBatch.ADVANTAGES]
            )

    rollout[SampleBatch.ADVANTAGES] = rollout[SampleBatch.ADVANTAGES].astype(np.float32)

    assert all(
        val.shape[0] == rollout_size for key, val in rollout.items()
    ), "Rollout stacked incorrectly!"
    return rollout


def discount_cumsum(x: np.ndarray, gamma: float) -> float:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma (float): The discount factor gamma.

    Returns:
        float: The discounted cumulative sum over the reward sequence `x`.
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def np_standardized(array):
    """Normalize the values in an array.

    Args:
        array (np.ndarray): Array of values to normalize.

    Returns:
        array with zero mean and unit standard deviation.
    """
    return (array - array.mean()) / max(1e-4, array.std())

@tf.function
def tf_standardized(tensor):
    return tensor - tf.reduce_mean(tensor, 0) / (tf.maximum(1e-7, tf.math.reduce_std(tensor, 0)))
