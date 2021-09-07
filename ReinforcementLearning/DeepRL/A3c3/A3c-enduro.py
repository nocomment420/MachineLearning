from Network import ActorCriticModel
import torch.multiprocessing as mp
from Worker import Worker
import torch
import numpy as np
import os
import pathlib

def run(i, *args):
    args[0].run()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', True)

    num_workers = 1#mp.cpu_count()
    colours = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey", ]

    # make result directory
    base_path = pathlib.Path(__file__).parent.absolute()
    path = "{}/run-07-06-21-pacman".format(base_path)
    try:
        os.mkdir(path)
    except:
        answer = input("Are you sure you want to overide run {} ? [y/n]".format(path.replace("{}/".format(base_path),"")))
        if answer == "y":
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        else:
            exit(1)
    # network params
    image_dim = [84, 84]
    color_dim = 4
    n_classes = 4
    lr = 0.001
    C = 0.01

    network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr)
    network.share_memory()

    benchmark_network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr)
    benchmark_network.load_state_dict(network.state_dict())
    best_score = 0
    benchmark_freq = 600

    grad_queue = mp.Queue()
    rewards_queue = mp.Queue(maxsize=benchmark_freq)
    rewards_list = []

    # Worker Params
    n_steps = 5
    frame_dimensions = [84, 84]
    gamma = 0.99
    env_name = "MsPacman-v0"

    print("Starting to create {} workers...".format(num_workers))
    workers = []
    for i in range(num_workers):
        color = colours[i % len(colours)]
        worker_id = "worker_{}".format(i + 1)
        worker = Worker(worker_id, env_name, network, n_steps, gamma, color, grad_queue, rewards_queue)
        workers.append(worker)


    threads = []
    for worker in workers:
        t = mp.spawn(run, args=(worker,), nprocs=1, join=False, daemon=False, start_method='spawn')
        threads.append(t)

    grads = None
    grad_count = 0
    batch_size = 64
    benchmark_count = 0

    while True:
        worker_grads = grad_queue.get()
        if grads is None:
            grads = worker_grads
            grad_count += 1

        elif grad_count % batch_size == 0:
            network.apply_grads(grads)
            grads = None
            grad_count = 0

        else:
            for (glob, worker) in zip(grads, worker_grads):
                glob += worker
            grad_count += 1

        # check if network has gotten worse
        if rewards_queue.full():
            benchmark_count += 1
            sum = 0
            for _ in range(benchmark_freq):
                reward = rewards_queue.get()
                sum += reward
                rewards_list.append(reward)

            av_reward = sum / benchmark_freq
            if av_reward < best_score:
                network.load_state_dict(benchmark_network.state_dict())
                result = "reverted to checkpoint"
            else:
                best_score = av_reward
                benchmark_network.load_state_dict(network.state_dict())
                result = "saved new checkpoint"

            print("Total episodes: {} | av_reward:{} | best_reward: {} -> {}".format(benchmark_count * benchmark_freq,
                                                                                     round(av_reward, 2),
                                                                                     round(best_score, 2), result))

            torch.save(network, "{}/a3c-{}.pt".format(path, benchmark_count))
            np.save("{}/a3c-rewards.npy".format(path), np.array(rewards_list))





