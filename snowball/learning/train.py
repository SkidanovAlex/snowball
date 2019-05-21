from snowball.adversary import Strategy
from snowball.learning.model import capture_state
from snowball.protocol import SnowballProtocol


def rollout(net, num_participants, adversary_percent, prob, alpha, beta, k, part_iterations):
    proto = SnowballProtocol(num_participants, adversary_percent, -1, prob, alpha, beta, k, part_iterations,
                             record=True)

    # Provide policy network to adversaries
    for par in proto.participant_objects:
        if par.adversary:
            par.strategy = Strategy.RL
            par.net = net

    states = []
    done = False
    while not done:
        states.append(capture_state(proto))
        done = proto.step()

        if proto.iteration % 100 == 0:
            print(proto.iteration)

    states.append(capture_state(proto))

    experience = []  # tuples of the form (state, action, reward, done, next_state)

    previous_state = None
    previous_votes = []

    for num, info in enumerate(proto.history):
        if info['votes'] is None:  # Step performed by adversary. Skip.
            continue

        part_id, votes = info['q_participants'], info['votes']
        adv_votes = [v for pid, v in zip(part_id, votes) if proto.participant_objects[pid].adversary]

        if adv_votes:  # Skip steps where not adversary was queried
            for v in previous_votes:
                sender = previous_state[info['from']]
                experience.append(((previous_state, sender), v, +1, False, states[num]))

            # Remember last state where an adversary answer a query
            previous_state = states[num]
            previous_votes = adv_votes

    if proto.consensus:
        for v in previous_votes:
            # This is the most relevant experience to learn useful policies since `done` = True
            experience.append((previous_state, v, +1, True, states[-1]))

    return experience
