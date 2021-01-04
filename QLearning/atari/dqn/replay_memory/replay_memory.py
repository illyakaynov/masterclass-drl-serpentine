import numpy as np

import collections

ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))


class CircularBufferReplayMemory:
    def __init__(self,
                 observation_shape,
                 stack_size=1,
                 replay_capacity=1000,
                 batch_size=32,
                 add_last_samples=3,
                 gamma=0.99,
                 observation_dtype=np.uint8,
                 teminal_shape=(),
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32, ):
        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._state_shape = self._observation_shape + (self._stack_size,)
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_shape = teminal_shape
        self._terminal_dtype = terminal_dtype
        self._add_last_samples = add_last_samples

        self.store = {}

        self.add_count = 0

        self._create_storage()

    def cursor(self):
        return self.add_count % self._replay_capacity

    def _get_storage_signature(self):
        storage_elements = [
            ReplayElement('state', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('next_state', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('terminal', self._terminal_shape, self._terminal_dtype)
        ]
        return storage_elements

    def _create_storage(self):
        for elem in self._get_storage_signature():
            self.store[elem.name] = np.empty(shape=((self._replay_capacity,) + elem.shape), dtype=elem.type)

    def _sample_idxs(self, batch_size):
        upper = min(self.add_count, self._replay_capacity)
        idxs = np.random.randint(upper, size=(batch_size,), dtype='int32')

        random_idxs = np.random.randint(batch_size, size=self._add_last_samples, dtype='int32')

        index = self.cursor()

        for i in random_idxs:
            idx = index - i - 1
            if idx < 0:
                idx += self._replay_capacity
            idxs[i] = idx

        return idxs

    def sample_memories(self, idxs=None):
        if idxs is None:
            idxs = self._sample_idxs(self._batch_size)

        transition_elems = []
        for elem in self._get_storage_signature():
            elem_tensor = self.store[elem.name][idxs]
            transition_elems.append(elem_tensor)

        transition_elems.append(idxs)
        return transition_elems

    def insert(self, transition):

        for transition_elem, storage_elem in zip(transition, self._get_storage_signature()):
            self.store[storage_elem.name][self.cursor()] = transition_elem
        self.add_count += 1


from dqn.replay_memory.sum_tree import SumTree


class PrioritisedCircularBufferReplayMemory(CircularBufferReplayMemory):
    def __init__(self,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 gamma=0.99,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 teminal_shape=(),
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32,
                 alpha=0.6,
                 beta=0.4,
                 beta_decay=1.00005,
                 beta_max=1.,
                 is_stratified_sampling=True,
                 uniform_priorities=False):
        super().__init__(observation_shape,
                         stack_size,
                         replay_capacity,
                         batch_size,
                         gamma,
                         max_sample_attempts,
                         extra_storage_types,
                         observation_dtype,
                         teminal_shape,
                         terminal_dtype,
                         action_shape,
                         action_dtype,
                         reward_shape,
                         reward_dtype)
        self.alpha = alpha
        self.beta = beta
        self.is_stratified_sampling = is_stratified_sampling
        self.beta_decay = beta_decay
        self.beta_max = beta_max
        self.uniform_priorities = uniform_priorities

        self.sum_tree = SumTree(replay_capacity)

    def _get_storage_signature(self):
        """The signature of the add function.

        The signature is the same as the one for OutOfGraphReplayBuffer, with an
        added priority.

        Returns:
          list of ReplayElements defining the type of the argument signature needed
            by the add function.
        """
        parent_add_signature = super()._get_storage_signature()
        add_signature = parent_add_signature + [
            ReplayElement('priority', (1,), np.float32)
        ]
        return add_signature

    def insert(self, transition):
        priority = np.expand_dims(np.array(self.sum_tree.max_recorded_priority), 0)
        super().insert(transition + priority)

    def _sample_idxs(self, batch_size):
        indices = np.asarray(self.sum_tree.stratified_sample(batch_size))
        return indices

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.

        For any memory location not yet used, the corresponding priority is 0.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).

        Returns:
          priorities: float, the corresponding priorities.
        """

        assert indices.dtype == np.int32 or indices.dtype == np.int64, ('Indices must be integers, '
                                                                        'given: {}'.format(indices.dtype))
        batch_size = len(indices)
        priority_batch = np.empty((batch_size,), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        return priority_batch

    def sample_memories(self, idxs=None):
        if idxs is None:
            idxs = self._sample_idxs(self._batch_size)

        transition_elems = []
        for elem in self._get_storage_signature():

            if elem.name == 'priority':
                priorities = self.get_priority(idxs)
                transition_elems.append(priorities)
                continue

            elem_tensor = self.store[elem.name][idxs]
            transition_elems.append(elem_tensor)

        transition_elems.append(idxs)
        return transition_elems

    def update_priorities(self, indices, priorities):
        if self.uniform_priorities:
            priorities = [1 for _ in range(len(priorities))]

        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)
