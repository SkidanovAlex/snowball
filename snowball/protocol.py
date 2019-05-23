import random

from adversary import AugmentedSnowballParticipant


class SnowballProtocol:
    def __init__(self, args):
        self.participants = list(range(args.num_participants))

        self.adversaries_num = int(args.num_participants * args.adversary_percent)
        self.good_num = args.num_participants - self.adversaries_num

        self.alpha = args.snowball_alpha
        self.beta = args.snowball_beta
        self.k = args.snowball_k

        self.adversary_strategy = args.adversary_strategy
        self.balance = args.balance
        self.top_iterations = args.part_iterations * args.num_participants

        self.query_method = None
        self.participant_objects = None
        self.running_participants = None
        self.iteration = 0

        self.history = []
        self.record = args.record

        self.net_name = args.net_name

        self.reset()

    def reset(self):
        self.history.clear()
        self.iteration = 0

        self.participant_objects = [
            AugmentedSnowballParticipant(i, self.participants, i >= self.good_num, self.adversary_strategy, self.alpha,
                                         self.beta, self.k, self.net_name)
            for i in self.participants]

        for part in self.participant_objects:
            part.color = random.uniform(0, 1) > self.balance

        def query_method(part_id, participants, color, callback):
            return callback(
                [self.participant_objects[participant_id].respond_to_query(part_id, color, self.participant_objects, iteration=self.iteration) for
                 participant_id in participants])

        self.query_method = query_method
        self.running_participants = list(self.participant_objects)

    @property
    def snowball_map(self):
        sb = {True: 0, False: 0}
        for part in self.participant_objects:
            if not part.adversary:
                sb[part.color] += 1
        return sb

    @property
    def consensus(self):
        sb = self.snowball_map
        return sb[True] == 0 or sb[False] == 0

    def remove_adversaries(self):
        self.running_participants = list(filter(lambda x: not x.adversary, self.running_participants))

        par_obj = []
        par_id = []

        for ix, par in zip(self.participants, self.participant_objects):
            if not par.adversary:
                par_id.append(ix)
                par_obj.append(par)

        self.participant_objects = par_obj
        self.participants = par_id
        for par in self.participant_objects:
            par.participants = par_id

    def log(self, from_id, queried_participants, votes):
        if self.record:
            self.history.append({
                'from': from_id,
                'q_participants': queried_participants,
                'votes': votes
            })

    def step(self):
        part = random.choice(self.running_participants)
        self.iteration += 1

        # Adversarial queries affect in no way the state but it must be counted toward number of iterations
        if part.adversary:
            self.log(part.self_id, None, None)
            return self.iteration == self.top_iterations

        q_participants, votes = part.snowball_iteration(self.query_method)
        self.log(part.self_id, q_participants, votes)

        if part.is_finished():
            self.running_participants.remove(part)

        if len(self.running_participants) == self.adversaries_num:
            # All good participants are committed to some value
            return True

        return self.iteration == self.top_iterations
