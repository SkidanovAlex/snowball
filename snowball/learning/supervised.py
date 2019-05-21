import multiprocessing
import os
import pickle
import random

import numpy as np
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from learning.model import load_net, save_net, capture_state, path
from protocol import SnowballProtocol

VERBOSE = True
DISCOUNT = .99


def get_samples(proto, take_percent=.01):
    state, sender, action, value = [], [], [], []

    proto.reset()
    done = False

    if VERBOSE:
        print("Start simulation")
        print(proto.snowball_map)

    while not done:
        # Save only 1% of the sample to avoid bias
        save = random.uniform(0, 1) < take_percent

        if save:
            s = capture_state(proto)

        done = proto.step()

        if save:
            info = proto.history[-1]

            if info['votes'] is not None:
                f = s[info['from']]

                part_id, votes = info['q_participants'], info['votes']

                for pid, v in zip(part_id, votes):
                    if proto.participant_objects[pid].adversary:
                        state.append(s)
                        sender.append(f)
                        action.append(v)
                        value.append(proto.iteration)

        if VERBOSE and proto.iteration % 10000 == 0:
            print(proto.iteration)

    last_it = proto.iteration
    for i in range(len(value)):
        n = last_it - value[i] + 1
        value[i] = (1 - DISCOUNT ** n) / (1 - DISCOUNT)

    return state, sender, action, np.array(value, dtype=np.float32)


def train(args):
    states, sender, action, value = [], [], [], []
    with open(os.path.join(path('dataset'), f'supervised-{args.adversary_strategy}.pkl'), 'rb') as f:
        num_sim = 0
        while True:
            try:
                states_x, sender_x, action_x, value_x = pickle.load(f)
                num_sim += 1
                states.extend(states_x)
                sender.extend(sender_x)
                action.extend(action_x)
                value.extend(value_x)
            except EOFError:
                print("Number of simulations loaded:", num_sim)
                print("Dataset entries:", len(states))
                break

    states = torch.Tensor(states)
    dataset = torch.utils.data.DataLoader(list(zip(states, sender, action, value)), batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    net = load_net(f'supervised-{args.adversary_strategy}')
    net.to(device)

    optim = torch.optim.Adam(net.parameters())

    for epoch in range(args.num_epochs):
        for s, f, a, v in dataset:
            s = s.to(device)
            f = f.to(device)
            vp, ap = net(s, f)

            # value_loss = F.mse_loss(vp, v.unsqueeze(1))
            a = a.to(device)
            action_loss = F.cross_entropy(ap, a)

            # loss = action_loss + value_loss

            optim.zero_grad()
            action_loss.backward()
            optim.step()

            print("Action loss:", action_loss.cpu().data.numpy())
            # print(loss, value_loss, action_loss)

    save_net(net, f'supervised-{args.adversary_strategy}')


def create_dataset(args):
    def build(tasks, lock, count):
        for _ in iter(tasks.get, '<>'):
            args.record = True
            proto = SnowballProtocol(args)

            states, sender, action, value = get_samples(proto)

            lock.acquire()

            if not os.path.exists(path('dataset')):
                os.mkdir(path('dataset'))

            count.value += 1

            print("Saving simulation: ", count.value)
            with open(os.path.join(path('dataset'), f'supervised-{args.adversary_strategy}.pkl'), 'ab') as f:
                pickle.dump((states, sender, action, value), f)

            lock.release()

    num_process = multiprocessing.cpu_count()
    print("Number of cores:", num_process)
    tasks = multiprocessing.Queue()
    [tasks.put(None) for _ in range(20)]
    [tasks.put('<>') for _ in range(num_process)]

    lock = multiprocessing.Lock()
    count = multiprocessing.Value('i', 0)

    for i in range(num_process):
        multiprocessing.Process(target=build, args=(tasks, lock, count)).start()
