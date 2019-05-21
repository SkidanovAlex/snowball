import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Snowball Framework')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    snowball_parameters = parser.add_argument_group('parameters')
    snowball_parameters.add_argument('--num_participants', type=int, default=2000)
    snowball_parameters.add_argument('--adversary_percent', type=float, default=.19)
    snowball_parameters.add_argument('--adversary_strategy', type=str, default='INCREASE_CONFIDENCE')
    snowball_parameters.add_argument('--balance', type=float, default=.5)
    snowball_parameters.add_argument('--snowball_alpha', type=float, default=.8)
    snowball_parameters.add_argument('--snowball_beta', type=int, default=120)
    snowball_parameters.add_argument('--snowball_k', type=int, default=10)
    snowball_parameters.add_argument('--part_iterations', type=int, default=1000)
    snowball_parameters.add_argument('--net_name', type=str, default='nn')
    snowball_parameters.set_defaults(record=False)

    subparser = parser.add_subparsers()

    experiment = subparser.add_parser('experiment')
    experiment.set_defaults(action='experiment')
    experiment.add_argument('--no_plt', action='store_true')
    experiment.add_argument('--verbose_every', type=int, default=5000)
    experiment.add_argument('--iterations_per_frame', type=int, default=5000)
    experiment.add_argument('--remove_after', type=int, default=None)

    learning = subparser.add_parser('learning')
    learning.set_defaults(action='learning')
    learning.add_argument('--create_dataset', action='store_true', default=False)
    learning.add_argument('--train_supervised', action='store_true', default=False)
    learning.add_argument('--num_epochs', type=int, default=32)

    rl = subparser.add_parser('rl')
    rl.set_defaults(action='rl')
    rl.add_argument('--rl_updates', type=int, default=1024)
    rl.add_argument('--discount', type=float, default=.99)

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not hasattr(args, 'action'):
        parser.print_help()
        exit(0)

    if args.action == 'experiment':
        import experiment
        import adversary

        args.adversary_strategy = getattr(adversary.Strategy, args.adversary_strategy)

        if args.no_plt:
            experiment.snowball(args)
        else:
            experiment.snowball_plt(args)

    elif args.action == 'learning':
        import learning.supervised
        import adversary

        args.adversary_strategy = getattr(adversary.Strategy, args.adversary_strategy)

        if args.create_dataset:
            learning.supervised.create_dataset(args)

        elif args.train_supervised:
            learning.supervised.train(args)

    elif args.action == 'rl':
        import learning.rl

        learning.rl.train(args)
