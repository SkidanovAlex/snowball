# A3C from http://arxiv.org/abs/1602.01783

import selectors

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import adversary
import learning.model
import learning.supervised
import protocol


def worker(conn: "mp.Connection", worker_id, args):
    def log(*args, **kwargs):
        print(f"Worker[{worker_id}]:", *args, **kwargs)

    log(worker_id)

    net = learning.model.load_net('a3c')

    while True:
        sd = conn.recv()

        if sd is None:  # Flag to terminate run
            break

        net.load_state_dict(sd)

        args.record = True
        args.adversary_strategy = 1  # Different from RL (to avoid loading useless net)

        proto = protocol.SnowballProtocol(args)

        # Update policy network to adversaries
        for part in proto.participant_objects:
            if part.adversary:
                part.strategy = adversary.Strategy.RL
                part.net = net

        # Run simulation
        log("Run simulation")
        states, sender, action, values = learning.supervised.get_samples(proto, take_percent=1.)
        log("Simulation over")

        # Compute gradient
        net.zero_grad()
        loss, value = 0., torch.zeros(1, 1)

        for st, s, a, v in reversed(list(zip(states, sender, action, values))):
            vp, ap = net(torch.Tensor(st).unsqueeze(0), torch.Tensor(s).unsqueeze(0))
            value = 1. + args.discount * value
            value_loss = F.mse_loss(vp, value)
            action_loss = F.cross_entropy(ap, torch.LongTensor([a]))
            loss += value_loss + action_loss

        loss.backward()

        grad = [param.grad for param in net.parameters()]
        conn.send(grad)


def train(args):
    num_workers = mp.cpu_count()
    print("Number of workers:", num_workers)

    net = learning.model.load_net('a3c')
    optim = torch.optim.Adam(net.parameters())

    start_sel = selectors.DefaultSelector()
    active = 0

    for i in range(num_workers):
        conn0, conn1 = mp.Pipe(True)
        mp.Process(target=worker, args=(conn1, i, args)).start()
        start_sel.register(conn0, selectors.EVENT_WRITE)
        active += 1

    iteration = 0

    running_sell = selectors.DefaultSelector()

    while active > 0:
        events = start_sel.select()

        for key, _ in events:
            conn = key.fileobj
            conn.send(net.state_dict())
            iteration += 1
            active -= 1

            start_sel.unregister(conn)
            running_sell.register(conn, selectors.EVENT_READ)

    active = num_workers

    while active > 0:
        print("Iteration:", iteration, "Active:", active)
        events = running_sell.select()

        for key, _ in events:
            conn = key.fileobj
            gradient = conn.recv()

            for param, grad in zip(net.parameters(), gradient):
                param.grad = grad

            optim.step()

            if iteration == args.rl_updates:
                conn.send(None)
                running_sell.unregister(conn)
                active -= 1
            else:
                conn.send(net.state_dict())
                iteration += 1

        learning.model.save_net(net, 'a3c')
