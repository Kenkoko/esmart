def seed_python(seed):
    import random

    random.seed(seed)


def seed_torch(seed):
    import torch

    torch.manual_seed(seed)


def seed_numpy(seed):
    import numpy.random

    numpy.random.seed(seed)

def seed_tf(seed):
    import tensorflow as tf
    tf.random.set_seed(seed)

def get_seed(config, what):
    seed = config.get(f"random_seed.{what}")
    if seed < 0 and config.get(f"random_seed.default") >= 0:
        import hashlib

        # we add an md5 hash to the default seed so that different PRNGs get a
        # different seed
        seed = (
            config.get(f"random_seed.default")
            + int(hashlib.md5(what.encode()).hexdigest(), 16)
        ) % 0xFFFF  # stay 32-bit

    return seed


def seed_from_config(config):
    seed = get_seed(config, "python")
    if seed > -1:
        seed_python(seed)

    seed = get_seed(config, "torch")
    if seed > -1:
        seed_torch(seed)

    seed = get_seed(config, "numpy")
    if seed > -1:
        seed_numpy(seed)

    seed = get_seed(config, "numba")
    # if seed > -1:
    #     seed_numba(seed)

    seed = get_seed(config, "tensorflow")
    if seed > -1:
        seed_tf(seed)

def seed_all(default_seed, python=-1, torch=-1, numpy=-1, numba=-1, tensorflow=-1):
    from esmart import Config

    config = Config()
    config.set("random_seed.default", default_seed)
    config.set("random_seed.python", python)
    config.set("random_seed.torch", torch)
    config.set("random_seed.numpy", numpy)
    config.set("random_seed.tensorflow", tensorflow)
    seed_from_config(config)