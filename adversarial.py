""" Run avalanche in an adversarial environment
"""
from avalanche import AvalancheMaster, BasicNode
from avalanche import Block, Transaction, rand

class Settings:
    # Environment parameters
    node_count = 100
    byzantine_percent = 0.15

    # Safety parameters
    k = 8
    alpha = .6
    success = int(k * alpha + .5)
    beta1 = 10
    beta2 = 10

    # Other
    transaction_spawn = .001
    conflict_transaction_spawn = .001

class Adversary(BasicNode):
    def __init__(self, settings):
        super().__init__(-1, settings)

        self.node_count = settings.node_count
        self.byz_node_count = int(self.node_count * settings.byzantine_percent)
        self.honest_node_count = self.node_count - self.byz_node_count

        self.started = False

        # How many conflicts have been created so far
        self.conflict_count = 0

        # Map: (A: conflictid) -> B: blockid[]
        # List all blocks in conflict.
        self.conflict_id = {}

        # Map: (A: blockid, B: conflictid) -> C: blockid
        # Keep track of which side of the conflict `B` block `A` support.
        self.conflict_support = {}

    def query(self, blockid):
        # Answer adaptive answer depending on `self.active_node`
        return 0

    def on_receive(self, block, adversary=False):
        super().on_receive(block)

        # if adversary:
        #     pass

        # else:
        #     pass

    def generate_conflict(self):
        # TODO: Use different parent selection for adversaries
        # Rigth now I'm creating a single fork so this is parent
        # selection algorithm should be ok
        parents = self.parent_selection()

        sender = rand()

        # Two transaction with the same sender form a fork
        tx0 = Transaction(sender, rand())
        tx1 = Transaction(sender, rand())

        blk0 = Block(tx0, parents)
        blk1 = Block(tx1, parents)

        self.log("Create-Block", blk0.id, blk0.parents)
        self.log("Create-Block", blk1.id, blk1.parents)
        self.log("Create-Conflict", blk0.id, blk1.id)

        conflict_count = self.conflict_count
        self.conflict_id[conflict_count] = [blk0, blk1]

        self.conflict_support[(blk0.id, conflict_count)] = blk0.id
        self.conflict_support[(blk1.id, conflict_count)] = blk1.id

        self.conflict_count += 1

        self.on_receive(blk0, True)
        self.on_receive(blk1, True)

        return conflict_count, [blk0, blk1]

    def step(self, active_node):
        """ active_node: Current honest participant index which is going to ask query next
        """
        self.active_node = active_node

        if not self.started:
            _, conflict = self.generate_conflict()

            # Send message perfectly balanced among honest nodes in the network!
            # On this synchronous environment we can split network in two halfs.
            #
            # NOTE: Is it ok for adversaries to gossip a message to every honest participant?
            # The protocol says you can only gossip to k random participants, in an asyncrhonous
            # settings might be harder for adversaries to achieve perfect splits.
            for i in range(self.honest_node_count):
                # Gossip part of the conflict depending on parity
                cid = i & 1

                part = self.participants[i]
                part.sync(conflict[cid].id, self)
                # For adversaries there is no need to query.
                # part.query(conflict[cid])

            self.started = True


def main():
    master = AvalancheMaster(Settings, Adversary)
    master.run()

if __name__ == '__main__':
    main()