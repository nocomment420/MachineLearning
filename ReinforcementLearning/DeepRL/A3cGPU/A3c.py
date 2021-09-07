from Network import ActorCriticModel
import torch.multiprocessing as mp
from Worker import Worker
import torch
import numpy as np
import ctypes


def run(i, *args):
    args[0].run()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', True)
    env_name = "Breakout-v0"

    # network params
    cpu_device = torch.device('cpu')
    gpu_device = torch.device('cuda')

    image_dim = [84, 84]
    color_dim = 4
    n_classes = 4
    lr = 0.001
    C = 0.01
    weight_repository = ActorCriticModel(image_dim, color_dim, n_classes, C, lr, cpu_device, None)
    weight_repository.share_memory()

    global_network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr, gpu_device, weight_repository)
    global_network.to(gpu_device)
    global_network.share_weights()

    # Worker Params
    num_workers = mp.cpu_count()
    colours = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey", ]
    n_steps = 15

    frame_dimensions = [84, 84]
    gamma = 0.99
    step_queue = mp.Queue(maxsize=1000)


    print("Starting to create {} workers...".format(num_workers))
    workers = []
    for i in range(num_workers):
        color = colours[i % len(colours)]
        worker_id = "worker_{}".format(i + 1)
        worker_network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr, cpu_device, weight_repository)
        worker = Worker(worker_id, env_name, worker_network, n_steps, gamma, color, step_queue)
        workers.append(worker)

    print("Running Workers".format(num_workers))
    threads = []
    for worker in workers:
        t = mp.spawn(run, args=(worker,), nprocs=1, join=False, daemon=False, start_method='spawn')
        threads.append(t)

    steps = []
    losses = []
    batch_size = 1028
    while True:
        worker_steps = step_queue.get()

        if len(steps) > batch_size:
            states = []
            actions = []
            rewards = []
            for step in steps:
                states.append(step.state)
                actions.append(step.action)
                rewards.append(step.reward)

            loss = global_network.calc_loss(states, actions, rewards)
            losses.append(loss)
            global_network.share_weights()
            steps = []

            if len(losses) > 20:
                av_loss = sum(losses) / len(losses)
                print("Applying grads - {}".format(av_loss))
                losses = []
        else:
            steps += worker_steps


