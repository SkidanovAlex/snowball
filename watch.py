#!/usr/bin/env python
import sys
import random
import time
from itertools import count

import pylab
from matplotlib.pyplot import pause
import matplotlib.colors as col
import networkx as nx
from tqdm import tqdm


pylab.ion()

EMPTY = 0
ACCEPTED = 0b01
CONFLICT = 0b10

colormask = col.ListedColormap([
    col.BASE_COLORS['r'],   # 0(00) Red
    col.BASE_COLORS['g'],   # 1(01) Green (accepted)
    col.BASE_COLORS['b'],   # 2(10) Blue (conflict)
    col.BASE_COLORS['g'],   # 3(11) Green (conflict accepted)
])

class DAGHandler:
    # Point of view of the DAG. Index of a participant on the network.
    # Use -1 to set global point of view (in this case a block is accepted if at least one participant accepted it)
    POV = 0

    def __init__(self):
        self.blocks = {}
        self.position = 0

        self.graph = nx.DiGraph()

    def receiveBlock(self, blockId, parents):
        if blockId in self.blocks:
            return False

        self.blocks[blockId] = parents

        x = self.position * 3
        y = self.position * 21 % 100
        self.position += 1

        self.graph.add_node(blockId, position=(x, y), color=EMPTY)

        for parent in parents:
            self.graph.add_edge(blockId, parent)

        return True

    def accept(self, blockId):
        self.graph.node[blockId]['color'] |= ACCEPTED
        return True

    def update(self, line):
        # Parse line
        node, report = line.split(':')
        nodeId = int(node.split()[1])
        command, *_args = report.split()
        args = ' '.join(_args)

        # Proccess line only depending on target point of view
        if (DAGHandler.POV >= 0 and nodeId == DAGHandler.POV) or\
            DAGHandler.POV == -1 or\
            nodeId == -1:

            if command == 'Accept-Block':
                # Mark this block as accepted
                blockId = int(args)
                return self.accept(blockId)

            elif command == 'Receive-Block' or command == 'Create-Block':
                # Add this block to the DAG
                blockId, *_parents = args.split()
                blockId = int(blockId)
                parents = eval(' '.join(_parents))
                return self.receiveBlock(blockId, parents)

            elif command == 'Create-Conflict':
                conflict_set = list(map(int, args.split()))

                for blockid in conflict_set:
                    self.graph.node[blockid]['color'] |= CONFLICT

                return True

        return False

    def draw(self):
        positions = nx.get_node_attributes(self.graph, 'position')
        colors = [self.graph.node[u]['color'] for u in self.graph.node]
        nx.draw_networkx(self.graph, with_labels=False, pos=positions, node_color=colors, cmap=colormask, vmin=0, vmax=3, node_size=100, width=.1)
        pause(0.00001)


def watch(path):
    handler = DAGHandler()

    idle = 0
    changes = 0

    pylab.show()

    with open(path) as f:
        for _ in tqdm(count()):
            if idle > 20:
                break
            line = f.readline()
            if line:
                idle = 0
                _, log = line.split('|')
                out = handler.update(log.strip())

                if out:
                    changes += 1
                    if changes == 10:
                        handler.draw()
                        changes = 0
            else:
                idle += 1
                time.sleep(.5)


def main():
    if len(sys.argv) > 1:
        try:
            logid = int(sys.argv[1])
            path = f'logs/avalanche-{logid}.log'
        except ValueError:
            path = sys.argv[1]
    else:
        from logger import find_id
        logid = find_id() - 1
        path = f'logs/avalanche-{logid}.log'

    print("Watching:", path)
    watch(path)

if __name__ == '__main__':
    main()