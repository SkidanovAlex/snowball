import random
import time

SNOWBALL_SAMPLE_SIZE = 10
SNOWBALL_ALPHA = 0.8
SNOWBALL_BETA = 120


class SnowballParticipant:
    def __init__(self, self_id, participants, alpha=SNOWBALL_ALPHA, beta=SNOWBALL_BETA, k=SNOWBALL_SAMPLE_SIZE):
        assert len(participants) > k

        self.color = None
        self.lastcolor = None
        self.self_id = self_id
        self.participants = participants
        self.d = {True: 0, False: 0}
        self.count = 0

        self.alpha = alpha
        self.beta = beta
        self.k = k

    @property
    def confidence(self):
        assert self.d[self.color] >= self.d[not self.color]
        return abs(self.d[True] - self.d[False])

    def respond_to_query(self, from_id, color, participants_objects=None):
        if self.color is None:
            self.color = color
        return self.color

    def is_finished(self):
        return self.count >= self.beta

    def get_subset(self):
        subset = random.sample(self.participants, self.k)
        while self.self_id in subset:
            subset = random.sample(self.participants, self.k)
        return subset

    def snowball_iteration(self, query_method):
        if self.color is None:
            return
        if self.is_finished():
            return

        subset = self.get_subset()
        return query_method(self.self_id, subset, self.color, lambda votes: self.snowball_iteration_post(subset, votes))

    def snowball_iteration_post(self, participants, votes):
        current_votes = {False: 0, True: 0}

        for v in votes:
            if v is not None:  # None stands for a node that didn't answer on timeout
                current_votes[v] += 1

        winning_vote = list(filter(
            lambda x: current_votes[x] >= self.k * self.alpha,
            current_votes.keys()))

        if len(winning_vote) == 0:
            self.count = 0
            return participants, votes

        vote = winning_vote[0]

        self.d[vote] += 1

        if self.d[vote] > self.d[self.color]:
            self.color = vote

        if vote == self.lastcolor:
            self.count += 1
        else:
            self.count = 0
            self.lastcolor = vote

        return participants, votes

    def snowball(self, query_method):
        count = 0
        while not self.is_finished():
            self.snowball_iteration(query_method)
            time.sleep(0.1)
