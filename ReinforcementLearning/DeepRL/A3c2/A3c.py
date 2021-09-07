from Network import ActorCriticModel
import torch.multiprocessing as mp
from Worker import Worker
import torch

def run(i, *args):
    args[0].run()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', True)

    num_workers = mp.cpu_count()
    colours = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey", ]

    # network params
    image_dim = [84, 84]
    color_dim = 4
    n_classes = 4
    lr = 0.002
    C = 0.01

    network = ActorCriticModel(image_dim, color_dim, n_classes, C, lr)
    network.share_memory()

    grad_queue = mp.Queue()
    # Worker Params
    n_steps = 5
    frame_dimensions = [84, 84]
    gamma = 0.99
    env_name = "Breakout-v0"

    print("Starting to create {} workers...".format(num_workers))
    workers = []
    for i in range(num_workers):
        color = colours[i % len(colours)]
        worker_id = "worker_{}".format(i + 1)
        worker = Worker(worker_id, env_name, network, n_steps, gamma, color, grad_queue)
        workers.append(worker)


    threads = []
    for worker in workers:
        t = mp.spawn(run, args=(worker,), nprocs=1, join=False, daemon=False, start_method='spawn')
        threads.append(t)

    grads = None
    grad_count = 0
    batch_size = 64
    trainings = 0
    while True:
        worker_grads = grad_queue.get()
        if grads is None:
            grads = worker_grads
            grad_count += 1
        elif grad_count % batch_size == 0:
            # print("Applying grads")
            network.apply_grads(grads)
            grads = None
            grad_count = 0
            trainings += 1
            if trainings % 50 == 0:
                torch.save(network, "a3c-{}.pt".format(trainings))

        else:
            for (glob, worker) in zip(grads, worker_grads):
                glob += worker
            grad_count += 1




