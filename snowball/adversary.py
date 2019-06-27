from learning.model import pick_action, load_net
from participant import SnowballParticipant


class Strategy:
    TRY_BALANCE = 0
    INCREASE_CONFIDENCE = 1
    EQUAL_SPLIT = 2
    RL = 3
    NON_ANSWER = 4
    BREAK_SAFETY = 5
    BREAK_LIVENESS = 6


class AugmentedSnowballParticipant(SnowballParticipant):
    def __init__(self, self_id, participants, adversary, strategy, alpha, beta, k, net_name='nn'):
        super().__init__(self_id, participants, alpha, beta, k)
        self.adversary = adversary

        if self.adversary:
            self.strategy = strategy
            if strategy == Strategy.RL:
                self.net = load_net(net_name)

    def is_finished(self):
        if self.adversary:
            # Adversaries never reach consensus
            return False
        else:
            return super().is_finished()

    def respond_to_query(self, from_id, color, participants_objects=None, iteration=None):
        if self.adversary:
            # Use private information stored on `self.participants`

            if self.strategy == Strategy.TRY_BALANCE:
                # Return less frequent color
                # This strategy tries greedily to balance participants
                snowball_map = {False: 0, True: 0}
                for par_id in self.participants:
                    par = participants_objects[par_id]
                    snowball_map[par.color] += 1

                return snowball_map[True] < snowball_map[False]

            elif self.strategy == Strategy.INCREASE_CONFIDENCE:
                # Return same color as received
                # This strategy increase each node confidence
                return color

            elif self.strategy == Strategy.EQUAL_SPLIT:
                # Split agents into two groups.
                return from_id % 2 == 0

            elif self.strategy == Strategy.RL:
                # Use policy learned through RL
                return pick_action(self.net, participants_objects, from_id)
            elif self.strategy == Strategy.NON_ANSWER:
                # Returning `None` must be interpreted as an adversary simulating a timeout
                # A correct node must be prepared to this situation as every node can crash at any point
                snowball_map = {False: 0, True: 0}
                for par_id in self.participants:
                    par = participants_objects[par_id]
                    snowball_map[par.color] += 1

                least_frequent = snowball_map[True] < snowball_map[False]

                if least_frequent == color:
                    return color
                else:
                    return None
            elif self.strategy == Strategy.BREAK_LIVENESS:
                if iteration < 100000:
                    # Return less frequent color
                    # This strategy tries greedily to balance participants
                    snowball_map = {False: 0, True: 0}
                    for par_id in self.participants:
                        par = participants_objects[par_id]
                        snowball_map[par.color] += 1

                    return snowball_map[True] < snowball_map[False]
                else:
                    return color

            elif self.strategy == Strategy.BREAK_SAFETY:
                # Split agents into two groups.
                assert iteration is not None
                if iteration < 100000:
                    return from_id % 200 >= 100
                else:
                    has_finished = False
                    for par_id in self.participants[:10]:
                        par = participants_objects[par_id]
                        if par.is_finished():
                            has_finished = True
                    if not has_finished:
                        return from_id % 200 >= 110
                    else:
                        print('.')
                        return True
            else:
                raise AssertionError(self.strategy)
        else:
            return super().respond_to_query(from_id, color)
