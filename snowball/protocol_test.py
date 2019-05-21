import unittest

from snowball.protocol import SnowballProtocol

VERBOSE = False


class BasicSnowballTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(BasicSnowballTest, self).__init__(*args, **kwargs)

    def _one_test(self, num_participants, prob):
        proto = SnowballProtocol(num_participants, 0., -1, prob, .8, 120, 10, float('inf'))

        done = False
        while not done:
            done = proto.step()

        self.assertTrue(proto.consensus)

        if VERBOSE:
            print("Iterations:", proto.iteration)
            print("Snowball Map:", proto.snowball_map)

    def test_sanity(self):
        self._one_test(100, 0.5)
        self._one_test(1000, 0.5)
        self._one_test(2000, 0.5)
        self._one_test(100, 0.7)
        self._one_test(1000, 0.7)
        self._one_test(2000, 0.7)
        self._one_test(100, 0.9)
        self._one_test(1000, 0.9)
        self._one_test(2000, 0.9)


if __name__ == "__main__":
    unittest.main()
