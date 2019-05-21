import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from matplotlib.animation import FuncAnimation

from protocol import SnowballProtocol


def snowball(args):
    proto = SnowballProtocol(args)

    verbose = args.verbose_every is not None

    if verbose:
        print("Running snowball")
        print(proto.snowball_map)

    done = False
    while not done:
        done = proto.step()

        if verbose and proto.iteration % args.verbose_every == 0:
            print(f"Iteration: {proto.iteration}")
            print(f"Remaining participants:", len(proto.running_participants) - proto.adversaries_num)
            print("Snowball map:", proto.snowball_map)

    if verbose:
        print("Consensus:", proto.consensus)
        print("Snowball iterations:", proto.iteration)
        print(proto.snowball_map)

    return proto


def snowball_plt(args):
    proto = SnowballProtocol(args)

    snowball_map = proto.snowball_map
    fig = plt.figure()

    col_dist = fig.add_subplot(221)
    pnts = col_dist.scatter([0, 1], [snowball_map[True], snowball_map[False]], color="red blue".split())
    col_dist.set_xlim(-1, 2)
    col_dist.set_ylim(0, proto.good_num)

    corr_ax = fig.add_subplot(222)
    corr_ax.set_title("Count confidence correlation")

    count_ax = fig.add_subplot(223)
    count_ax.set_title("Participants count distribution")
    count_true_line, = count_ax.plot([], [], color="red")
    count_false_line, = count_ax.plot([], [], color="blue")

    confidence_ax = fig.add_subplot(224)
    confidence_ax.set_title("Participants confidence distribution")
    confidence_true_line, = confidence_ax.plot([], [], color="red")
    confidence_false_line, = confidence_ax.plot([], [], color="blue")

    proto._removed = False

    def update(frame):
        for i in range(args.iterations_per_frame):
            done = proto.step()
            if done:
                break

        if args.remove_after is not None and not proto._removed:
            min_confidence = min(x.confidence for x in proto.participant_objects if not x.adversary)

            # Remove adversaries after all participants has confidence greater than some threshold
            if min_confidence >= args.remove_after:
                proto._removed = True
                proto.remove_adversaries()

        snowball_map = {False: 0, True: 0}
        count_map = {False: [], True: []}
        confidence_map = {False: [], True: []}

        for part in proto.participant_objects:
            # Only check for color of non adversarial participants
            if not part.adversary:
                snowball_map[part.color] += 1
                count_map[part.color].append(part.count)
                confidence_map[part.color].append(part.confidence)

        def kde_helper(data, x, line):
            max_y = 0
            try:
                kde = ss.gaussian_kde(data)
                y = kde(x)
                max_y = max(y)
                line.set_data(x, y)
            # Exceptions to handle degenerated data
            except np.linalg.linalg.LinAlgError:
                pass
            except ValueError:
                pass
            finally:
                return max_y

        # Count graphic update
        max_count = max(count_map[True] + count_map[False])
        xcount = np.linspace(0, max_count, 1000)
        count_ax.set_xlim(0, max_count)
        count_ax_ylim = 0

        count_ax_ylim = max(count_ax_ylim, kde_helper(count_map[True], xcount, count_true_line))
        count_ax_ylim = max(count_ax_ylim, kde_helper(count_map[False], xcount, count_false_line))

        count_ax.set_ylim(0, count_ax_ylim)

        # Confidence graphic update

        max_confidence = max(confidence_map[True] + confidence_map[False])
        xconfidence = np.linspace(0, max_confidence, 1000)
        confidence_ax.set_xlim(0, max_confidence)
        confidence_ax_ylim = 0
        confidence_ax_ylim = max(confidence_ax_ylim,
                                 kde_helper(confidence_map[True], xconfidence, confidence_true_line))
        confidence_ax_ylim = max(confidence_ax_ylim,
                                 kde_helper(confidence_map[False], xconfidence, confidence_false_line))

        confidence_ax.set_ylim(0, confidence_ax_ylim)

        pnts.set_offsets([[0, snowball_map[True]], [1, snowball_map[False]]])

        corr_ax.clear()
        corr_ax.set_title("Count confidence correlation")
        corr_ax.scatter(count_map[True], confidence_map[True], color='red')
        corr_ax.scatter(count_map[False], confidence_map[False], color='blue')
        corr_ax.set_ylim(0, None)

        col_dist.set_title(
            f"Iteration: {proto.iteration} Red: {snowball_map[True]} Blue: {snowball_map[False]}"
            f" {'OFF' if proto._removed else 'ON'}")

        return pnts, count_true_line, count_false_line

    print("Snowball")

    _ = FuncAnimation(fig, update, interval=1, repeat=False)
    plt.show()
