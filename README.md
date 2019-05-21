# Avalanche

## Snowball

From `avalanche/snowball` directory, run:

    python run.py --adversary_percent=0.19 --adversary_strategy EQUAL_SPLIT experiment

Observe 4 plots that show nodes color assignment and their confidence in that color.
If at the start already one color is prevalent - restart.

Example of snowball being very confident in both colors for 1000 iterations:

![Example of diverging snowball](img/showball_diverge.png?raw=true "Example of diverging snowball")

See `adversary.py` for more adversary strategies.

### Supervised framework pipeline

From `avalanche/snowball` directory:

Create target dataset

    python run.py --adversary_strategy TRY_BALANCE learning --create_dataset

Train model using this dataset

    python run.py --adversary_strategy TRY_BALANCE learning --train

Test model performance on protocol simulation

    python run.py --adversary_strategy RL --net_name supervised-0 experiment --iterations_per_frame 100

Consider using low number of `--iterations_per_frame` or `--no_plt`.
Evaluating the network at every adversary step is expensive.

### Train with reinforcement learning

Current training algorithm is A3C. To start training

    python run.py rl

# Credits
This code is copied from a private branch, and is initially built by [Marcelo Fornet](https://github.com/mfornet).
