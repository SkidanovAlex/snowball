import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNetwork(nn.Module):
    H = 100

    def __init__(self):
        super().__init__()

        self.participant0 = nn.Linear(12, self.H)
        self.participant1 = nn.Linear(self.H, self.H)

        self.state0 = nn.Linear(self.H, self.H)
        self.state1 = nn.Linear(self.H, self.H)

        self.value_head = nn.Linear(self.H, 1)
        self.action_head = nn.Linear(2 * self.H, 2)

    def participants_embed(self, participants):
        participants = F.relu(self.participant0(participants))
        participants = F.relu(self.participant1(participants))
        return participants

    def forward(self, participants, sender=None):
        # Use DeepSet idea from https://arxiv.org/abs/1703.06114 to encode current state
        # since the order of participants is not relevant.

        participants = self.participants_embed(participants)
        state = participants.sum(1)
        state = F.relu(self.state0(state))
        state = F.relu(self.state1(state))

        value = self.value_head(state)

        if sender is not None:
            sender = self.participants_embed(sender)
            encode = torch.cat([sender, state], 1)
            action_prob = self.action_head(encode)  # Return logits
        else:
            action_prob = None

        return value, action_prob


def path(folder):
    return os.path.join(os.path.split(__file__)[0], folder)


def reset_net(name):
    net = PolicyValueNetwork()
    return save_net(net, name)


def load_net(name='nn'):
    try:
        with open(os.path.join(path('model'), f'{name}.pth'), 'rb') as f:
            net = torch.load(f)
    except FileNotFoundError:
        print(f"Warning! Model not found. Creating new neural network at model/{name}.pth")
        net = reset_net(name)
    return net


def save_net(net, name='nn'):
    if not os.path.exists(path('model')):
        os.mkdir(path('model'))

    with open(os.path.join(path('model'), f'{name}.pth'), 'wb') as f:
        torch.save(net, f)
    return net


def part2feat(participant):
    # participant 2 feature
    # features: true_count, true_confidence, false_count, false_confidence, adversary, bits

    num_bits = 5
    feat = np.zeros(7 + num_bits, dtype=np.float32)

    if participant.adversary:
        feat[6] = 1
    else:
        assert participant.color is not None
        if participant.color:
            feat[0] = participant.count
            feat[1] = participant.confidence
            feat[2] = 1.
        else:
            feat[3] = participant.count
            feat[4] = participant.confidence
            feat[5] = 1.

    for i in range(num_bits):
        feat[7 + i] = participant.self_id >> i & 1

    return feat


def capture_state(proto):
    return [part2feat(part) for part in proto.participant_objects]


def pick_action(net, participants, from_id):
    participants_feat = [part2feat(part) for part in participants]

    participants_feat = torch.Tensor(participants_feat).unsqueeze(0)
    from_participant_feat = participants_feat[:, from_id]

    _, action_prob = net(participants_feat, from_participant_feat)

    return bool(action_prob[0, 1] > action_prob[0, 0])  # Pick action greedy


if __name__ == '__main__':
    net = load_net('supervised')
    save_net(net)
