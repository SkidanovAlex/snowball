"""
    Implementation of Avalanche [1]

    [1] Snowflake to Avalanche: A Novel Metastable Consensus Protocol Family for Cryptocurrencies
"""

import random
import copy
from queue import Queue
from io import StringIO

import logger
logger_avalanche = logger.get_logger_avalanche()

ACTIVE_LOG = True

# Turn on for reproductible experiments
random.seed(0)

class Settings:
    # Environment parameters
    node_count = 100
    byzantine_percent = 0.

    # Safety parameters
    k = 8
    alpha = .6
    success = int(k * alpha + .5)
    beta1 = 10
    beta2 = 10

    # Other
    transaction_spawn = .001

def rand():
    return random.randint(0, 2**64 - 1)

def sample(n, k, u):
    """ Sample k integers from [0..n-1] without repetetion ignoring `u`
    """
    return [x + int(x >= u) for x in random.sample(range(n - 1), k)]

class Transaction:
    def __init__(self, sender, receiver):
        self.id = rand()
        self.sender = sender
        self.receiver = receiver

class Block:
    genesis_id = 0

    def __init__(self, body, parents):
        self.id = rand()
        self.body = body

        # List with id of parent blocks
        self.parents = parents

        # Whether this block has been already accepted or not
        self.accepted = False

    def __repr__(self):
        name = 'TX' if isinstance(self.body, Transaction) else 'NoP'
        return f"Block({name},{str(self.id)},{len(self.parents)},{int(self.accepted)})"

    @classmethod
    def fromblock(cls, block):
        nblock = copy.deepcopy(block)
        nblock.accepted = False
        return nblock

    @classmethod
    def genesis(cls):
        block = Block(None, [])
        block.id = Block.genesis_id
        return block


class Snowball:
    """ There is a snowball instance per conflict
    """
    def __init__(self, txid):
        self.pref = txid

        # Snowball internal state
        self.last = txid
        self.cnt = 0

        # In [1] confidence is referred as d_u(t)
        self.confidence = {txid : 0}

    def add(self, txid):
        # A transaction can be reissued without causing a (new) conflict
        self.confidence[txid] = self.confidence.get(txid, 0)

    @property
    def size(self):
        return len(self.confidence)


class BasicNode:
    def __init__(self, ix, settings):
        self.settings = settings
        self.index = ix

        # Participants list is initialized in `set_participants`
        # It will be used by honest nodes only to call `query` method
        self.participants = None

        # DAG information. (block_id -> block)
        self.blocks = {}

        # Id of blocks without children
        self.roots = set()

        # "Two transactions conflict if they consume the same UTXO and yield different outputs." [1]
        # Conflict sets are a Snowball instance per blocks(transactions) in conflict
        # Conflict is a transitive relation. Each block is directed to a single snowball instance
        #
        # Map: sender -> snowball_instance
        self.conflict_set = {}

        # Transactions that haven't been processed yet.
        self.pending_blocks = Queue()

        # Initialize DAG
        self.init()

        # Statistic variables
        self.accepted_blocks_count = 1 # Only genesis in the begining

    def log(self, *args, **kwargs):
        """ Implement print interface
        """

        if ACTIVE_LOG:
            output = StringIO()
            print(*args, **kwargs, file=output, end="")
            message = output.getvalue()

            logger_avalanche.info(f"NODE {self.index}: {message}")

    def init(self):
        """ Create genesis block
        """
        genesis = Block.genesis()
        self.on_receive(genesis)

    def set_participants(self, participants):
        self.participants = participants

    def on_receive(self, block):
        """ procedure onReceiveTx(T)
            Figure 5: Avalanche: transaction generation. [1]
        """
        if not block.id in self.blocks:
            self.blocks[block.id] = block

            if isinstance(block.body, Transaction):
                txid = block.body.id
                utxo = block.body.sender

                if utxo not in self.conflict_set:
                    self.conflict_set[utxo] = Snowball(txid)
                else:
                    self.conflict_set[utxo].add(txid)

                self.pending_blocks.put(block)

            # Update current roots
            for parid in block.parents:
                assert parid in self.blocks

                if parid in self.roots:
                    self.roots.remove(parid)

            self.roots.add(block.id)
            self.log("Receive-Block", block.id, block.parents)

    def get(self, blockid):
        # assert blockid in self.blocks
        return Block.fromblock(self.blocks[blockid])

    def sync(self, blockid, participant):
        """ Mechanism that allow two participants sync their DAG blocks.
        """
        if blockid in self.blocks:
            return

        block = participant.get(blockid)
        for parid in block.parents:
            self.sync(parid, participant)

        # Call on_receive after processing parents
        self.on_receive(block)

    def get_conflict_set(self, block):
        """ Return snowball instance of blocks that contains a transaction
            and None for noop blocks
        """
        if isinstance(block.body, Transaction):
            return self.conflict_set[block.body.sender]
        elif block.body is None:
            return None

    def is_preferred(self, block):
        """ procedure isPreferred(T)
            Figure 6: Avalanche: voting and decision primitives [1]
        """
        if block.body is None:
            # Nop transaction
            return True
        else:
            utxo = block.body.sender
            txid = block.body.id
            return self.conflict_set[utxo].pref == txid

    def dag_head(self, block):
        """ Iterate over active vertices in the DAG reachables from `block`. Non accepted transactions

            Quote from *optimizations*. page 15. [1]
              > Since the search path can be pruned at accepted vertices,
              > the cost for an update is constant if the rejected vertices
              > have limited number of descendants and the undecided
              > region of the DAG stays at constant size.
        """
        queue = [block]
        visited = {block.id}

        while queue:
            cur_block = queue.pop()
            yield cur_block

            for parid in cur_block.parents:
                par_block = self.blocks[parid]

                # Prune at accepted nodes
                if par_block.accepted:
                    continue

                par_block_id = par_block.id
                if par_block_id in visited:
                    continue

                visited.add(par_block_id)
                queue.append(par_block)

    def is_strongly_preferred(self, block):
        """ procedure isStronglyPreferred(T)
            Figure 6: Avalanche: voting and decision primitives [1]
        """
        if block.accepted:
            return True

        if not self.is_preferred(block):
            return False

        return all(map(self.is_preferred, self.dag_head(block)))

    def query(self, blockid):
        """ procedure onQuery(j, T)
            Figure 6: Avalanche: voting and decision primitives [1]
        """
        block = self.blocks[blockid]
        value = int(self.is_strongly_preferred(block))
        return value

    def topological_sort(self):
        order = []
        degree = {}

        def update_degree(block):
            for parid in block.parents:
                degree[parid] = degree.get(parid, 0) + 1

        for blockid in self.roots:
            block = self.blocks[blockid]

            order.append(block)
            update_degree(block)

        i = 0
        while i < len(order):
            block = order[i]
            i += 1

            if block.accepted:
                continue

            for parid in block.parents:
                degree[parid] -= 1

                if degree[parid] == 0:
                    nblock = self.blocks[parid]
                    order.append(nblock)
                    update_degree(nblock)

        return order

    def parent_selection(self):
        """
            Quote from *Parent Selection*. page 15. [1]
                > The adaptive parent selection algorithm chooses par-
                > ents by starting at the DAG frontier and retreating towards
                > the genesis vertex until finding an eligible parent.

                > Otherwise, the algorithm tries the parents of the trans-
                > actions in E, thus increasing the chance of finding more
                > stabilized transactions as it retreats. The retreating search
                > is guaranteed to terminate when it reaches the genesis
                > vertex.

            TODO: Retreat parent selection as previous selection fails
        """
        order = list(reversed(self.topological_sort()))

        parents = set()
        strongly_preferred = set()

        for block in order:
            if block.accepted:
                parents.add(block.id)
                strongly_preferred.add(block.id)
            else:
                if not self.is_preferred(block):
                    continue

                sp = True # strongly preferred
                for parid in block.parents:
                    if parid not in strongly_preferred:
                        sp = False
                        break

                if sp:
                    strongly_preferred.add(block.id)

                    if isinstance(block.body, Transaction):
                        snowball = self.get_conflict_set(block)
                        ok = snowball.size == 1 or snowball.confidence[block.body.id] > 0
                    else:
                        # This is a noop block (conflict set is always of size 1)
                        ok = True

                    if ok:
                        parents.add(block.id)

                        for parid in block.parents:
                            if parid in parents:
                                parents.remove(parid)

        assert len(parents) > 0

        return list(parents)

    def generate_tx(self):
        parents = self.parent_selection()
        tx = Transaction(rand(), rand())
        block = Block(tx, parents)
        self.log("Create-Block", block.id, block.parents)
        self.on_receive(block)

    def is_accepted(self, block):
        """ procedure isAccepted(T)
            Figure 6: Avalanche: voting and decision primitives. [1]
        """
        if block.accepted:
            return True

        snowball = self.get_conflict_set(block)

        # consecutive counter
        if snowball is not None and\
            snowball.pref == block.body.id and snowball.cnt >= self.settings.beta2:
            block.accepted = True

        # safe early commitment
        elif snowball is None or\
            snowball.size == 1 and snowball.confidence[block.body.id] >= self.settings.beta1:

            for parid in block.parents:
                parblock = self.blocks[parid]

                if not parblock.accepted:
                    break
            else:
                block.accepted = True

        if block.accepted:
            self.accepted_blocks_count += 1
            self.log("Accept-Block", block.id)

        return block.accepted

    def step(self):
        """ Figure 4: Avalanche: the main loop. [1]
        """
        # Create new transaction at fixed rate
        if random.random() < self.settings.transaction_spawn:
            self.generate_tx()

        if not self.pending_blocks.empty():
            block = self.pending_blocks.get()
            parts = sample(self.settings.node_count, self.settings.k, self.index)

            value = 0

            for pix in parts:
                part = self.participants[pix]

                # Sync block view before making a query
                part.sync(block.id, self)

                value += part.query(block.id)

                # If block will collect the chit, break without asking remaining participants
                if value >= self.settings.success:
                    break

            if value >= self.settings.success:
                for headblock in self.dag_head(block):

                    # Ignore nop blocks
                    if isinstance(headblock.body, Transaction):
                        txid = headblock.body.id
                        utxo = headblock.body.sender
                        snowball = self.conflict_set[utxo]

                        snowball.confidence[txid] += 1
                        cur_confidence = snowball.confidence[txid]
                        pref = snowball.pref
                        pref_confidence = snowball.confidence[pref]

                        if cur_confidence > pref_confidence:
                            snowball.pref = txid

                        if txid != snowball.last:
                            snowball.last = txid
                            snowball.cnt = 0
                        else:
                            snowball.cnt += 1

                # Accept blocks
                for block in reversed(self.topological_sort()):
                    self.is_accepted(block)


class DummyAdversary:
    """ From the point of view of honest participants there are several indistiguishible hidden
        adversaries but in practice there is a single adversarial instance.
    """
    def __init__(self, settings):
        pass

    def set_participants(self, participants):
        pass

    def sync(self, blockid, participant):
        pass

    def query(self, blockid):
        return 0

    def step(self, active_node):
        """ active_node: Current honest participant index which is going to ask query next
        """
        pass


class AvalancheMaster:
    def __init__(self, settings, adversarial_cls=DummyAdversary):
        self.settings = settings

        self.node_count = settings.node_count
        self.byz_node_count = int(self.node_count * settings.byzantine_percent)
        self.honest_node_count = self.node_count - self.byz_node_count
        self.adversary = adversarial_cls(settings)

        # Adversaries behave as a single entity, so a single instance is used for all of them
        self.participants = [BasicNode(i, self.settings) for i in range(self.honest_node_count)] +\
                            [self.adversary] * self.byz_node_count

        for part in self.participants:
            part.set_participants(self.participants)

    def adversary_update(self, active_node):
        self.adversary.step(active_node)

    def run(self):
        """ Using global scheduler to run avalanche
        """
        while True:
            u = random.choice(range(self.honest_node_count))
            self.adversary_update(u)
            self.participants[u].step()


def main():
    master = AvalancheMaster(Settings)
    master.run()


if __name__ == '__main__':
    main()
