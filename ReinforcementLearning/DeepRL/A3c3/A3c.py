from Network import ActorCriticModel
import torch.multiprocessing as mp
from Worker import Worker, GameImageProcessor
import torch
import numpy as np
import os
import pathlib
import datetime
import gym


def run(i, *args):
    args[0].run(print_freq=20)


if __name__ == "__main__":
    ENV_NAME = "Breakout-v0"
    # ENV_NAME = "MsPacman-v0"

    mp.freeze_support()
    mp.set_start_method('spawn', True)

    num_workers = mp.cpu_count()
    colours = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey", ]

    # make result directory
    base_path = pathlib.Path(__file__).parent.absolute()
    now = datetime.datetime.now()
    folder_name = "{} - {}-{}-{}".format(ENV_NAME, now.day, now.month, now.year)
    path = "{}/{}".format(base_path, folder_name)
    try:
        os.mkdir(path)
    except:
        answer = input(
            "Are you sure you want to override run {} ? [y/n]".format(path.replace("{}/".format(base_path), "")))
        if answer == "y":
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        else:
            exit(1)

    # network params
    image_dim = [84, 84]
    color_dim = 4
    env = gym.envs.make(ENV_NAME)
    n_classes = env.action_space.n
    # lr = 0.005
    lr = 0.001
    C = 0.01

    network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr)
    network.share_memory()

    benchmark_network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr)
    benchmark_network.load_state_dict(network.state_dict())
    best_score = 0
    benchmark_freq = 700

    grad_queue = mp.Queue()
    rewards_queue = mp.Queue(maxsize=benchmark_freq)
    rewards_list = []

    # Worker Params
    n_steps = 5
    frame_dimensions = [84, 84]
    gamma = 0.99

    if ENV_NAME == "Breakout-v0":
        image_crop = [31, 195]
    elif ENV_NAME == "MsPacman-v0":
        image_crop = [0, -50]
    else:
        image_crop = [0, -0]

    print("Starting to create {} workers...".format(num_workers))
    workers = []
    for i in range(num_workers):
        color = colours[i % len(colours)]
        worker_id = "worker_{}".format(i + 1)
        image_processor = GameImageProcessor(image_crop, frame_dimensions)

        worker = Worker(worker_id, ENV_NAME, network, n_steps, gamma, color, grad_queue, rewards_queue, image_processor)
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
