import numpy as np


class ReplayBuffer(object):
    """ Cyclic buffer
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.pointer = 0

    @property
    def size(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.pointer += 1
        else:
            if self.pointer == self.max_size:
                self.pointer = 0

            self.buffer[self.pointer] = experience
            self.pointer += 1

    def sample(self, size):
        replace_mode = size > len(self.buffer)
        index = np.random.choice(self.size, size=size, replace=replace_mode)
        return [self.buffer[idx] for idx in index]
