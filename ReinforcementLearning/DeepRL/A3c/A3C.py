from Networks import ActorCriticModel
from Worker import Worker
import torch.multiprocessing as mp
from SharedAdam import SharedAdam


def run(i, *args):
    args[0].run()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', True)

    debugging = False

    num_workers = mp.cpu_count()
    global_iter = mp.Value('i', 0)
    max_total_steps = 5e6
    colours = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey", ]

    # network params
    image_dim = [84, 84]
    color_dim = 4
    n_classes = 4
    lr = 1e-4
    C = 0.01

    global_network = ActorCriticModel(image_dim, color_dim, n_classes, C)
    global_network.share_memory()
    global_optimiser = SharedAdam(global_network.parameters(), lr=lr)

    # Worker Params
    n_steps = 15
    frame_dimensions = [84, 84]
    gamma = 0.99
    env_name = "Breakout-v0"

    print("Starting to create {} workers...".format(num_workers))
    workers = []
    for i in range(num_workers):
        color = colours[i % len(colours)]
        worker_id = "worker_{}".format(i + 1)
        local_network = ActorCriticModel(image_dim, color_dim, n_classes, C)
        worker = Worker(worker_id, env_name, global_network, local_network,global_optimiser, global_iter, max_total_steps, n_steps, gamma,
                        color)
        workers.append(worker)

    if not debugging:

        print("Starting each worker for total of {} steps".format(max_total_steps))
        threads = []
        for worker in workers:
            t = mp.spawn(run, args=(worker,), nprocs=1, join=False, daemon=False, start_method='spawn')
            threads.append(t)

        print("Awaiting workers...")
        for t in threads:
            t.join()

        print("DONE!")

    else:
        workers[0].run()
