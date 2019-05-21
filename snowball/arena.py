import logging
import sys

import adversary
import experiment
import run

import multiprocessing as mp
from itertools import repeat, product


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s| %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

    return logger


def execute_snowball(inputs):
    args, (p, s) = inputs
    args.adversary_percent = p
    args.adversary_strategy = getattr(adversary.Strategy, s)
    proto = experiment.snowball(args)

    logger.info(f"Percent: {p} Strategy: {s}")
    logger.info(f"Consensus: {int(proto.consensus)} Iteration: {proto.iteration}")
    logger.info(str(proto.snowball_map))


def main(args, logger):
    S = adversary.Strategy
    percents = [0., .05, .1, .15, .17, .18, .19, .20]
    strategies = ["INCREASE_CONFIDENCE", "EQUAL_SPLIT", "NON_ANSWER"]

    logger.info("Start run")
    num_proc = mp.cpu_count() + 2

    for i in range(5):
        params = list(zip(repeat(args), product(percents, strategies)))

        with mp.Pool(num_proc) as p:
            p.map(execute_snowball, params)


if __name__ == '__main__':
    parser = run.get_arg_parser()
    args = parser.parse_args()

    logger = setup_logger('measure', 'log/test09172018.log')
    main(args, logger)
