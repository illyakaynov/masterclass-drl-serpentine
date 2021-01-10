import sys

import scipy.signal
import numpy as np

def aligned_array(size, dtype, align=64):
    """Returns an array of a given size that is 64-byte aligned.

    The returned array can be efficiently copied into GPU memory by TensorFlow.
    """

    n = size * dtype.itemsize
    empty = np.empty(n + (align - 1), dtype=np.uint8)
    data_align = empty.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    if n == 0:
        # stop np from optimising out empty slice reference
        output = empty[offset:offset + 1][0:0].view(dtype)
    else:
        output = empty[offset:offset + n].view(dtype)

    assert len(output) == size, len(output)
    assert output.ctypes.data % align == 0, output.ctypes.data
    return output

def concat_aligned(items, time_major=None):
    """Concatenate arrays, ensuring the output is 64-byte aligned.

    We only align float arrays; other arrays are concatenated as normal.

    This should be used instead of np.concatenate() to improve performance
    when the output array is likely to be fed into TensorFlow.

    Args:
        items (List(np.ndarray)): The list of items to concatenate and align.
        time_major (bool): Whether the data in items is time-major, in which
            case, we will concatenate along axis=1.

    Returns:
        np.ndarray: The concat'd and aligned array.
    """

    if len(items) == 0:
        return []
    elif len(items) == 1:
        # we assume the input is aligned. In any case, it doesn't help
        # performance to force align it since that incurs a needless copy.
        return items[0]
    elif (isinstance(items[0], np.ndarray)
          and items[0].dtype in [np.float32, np.float64, np.uint8]):
        dtype = items[0].dtype
        flat = aligned_array(sum(s.size for s in items), dtype)
        if time_major is not None:
            if time_major is True:
                batch_dim = sum(s.shape[1] for s in items)
                new_shape = (
                    items[0].shape[0],
                    batch_dim,
                ) + items[0].shape[2:]
            else:
                batch_dim = sum(s.shape[0] for s in items)
                new_shape = (
                    batch_dim,
                    items[0].shape[1],
                ) + items[0].shape[2:]
        else:
            batch_dim = sum(s.shape[0] for s in items)
            new_shape = (batch_dim, ) + items[0].shape[1:]
        output = flat.reshape(new_shape)
        assert output.ctypes.data % 64 == 0, output.ctypes.data
        np.concatenate(items, out=output, axis=1 if time_major else 0)
        return output
    else:
        return np.concatenate(items, axis=1 if time_major else 0)

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

class SampleBatch:
    OBS = "obs"
    CUR_OBS = "obs"
    NEXT_OBS = "new_obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    PREV_ACTIONS = "prev_actions"
    PREV_REWARDS = "prev_rewards"
    DONES = "dones"
    INFOS = "infos"

    ACTION_DIST_INPUTS = "action_dist_inputs"
    ACTION_PROB = "action_prob"
    ACTION_LOGP = "action_logp"

    # Uniquely identifies an episode.
    EPS_ID = "eps_id"

    # Value function predictions emitted by the behaviour policy.
    VF_PREDS = "vf_preds"

    def __init__(self, *args, **kwargs):
        self.data = dict(*args, **kwargs)

        lengths = []
        for k, v in self.data.copy().items():
            assert isinstance(k, str), self
            lengths.append(len(v))
            if isinstance(v, list):
                self.data[k] = np.array(v)
                if len(self.data[k].shape) == 1:
                    self.data[k] = self.data[k][..., None]
        if not lengths:
            raise ValueError("Empty sample batch")
        assert (
            len(set(lengths)) == 1
        ), "Data columns must be same length, but lens are {}".format(lengths)

        self.new_columns = []

    def size_bytes(self) -> int:
        """
        Returns:
            int: The overall size in bytes of the data buffer (all columns).
        """
        return sum(sys.getsizeof(d) for d in self.data.values())

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
                "SampleBatches to concat must have same columns! {} vs {}".
                format(list(self.keys()), list(other.keys())))
        out = {}
        for k in self.keys():
            out[k] = np.concatenate([self[k], other[k]])
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

