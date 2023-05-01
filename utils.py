"""
This module provides the function to detect the hardware environment and set up the appropriate
TensorFlow distribution strategy.

Functions
---------
detect_hardware(tpu_name):
    Detects the hardware environment and sets up the appropriate TensorFlow distribution strategy.
    Returns the TPUClusterResolver and the distribution strategy objects.
"""
import tensorflow as tf


def detect_hardware(tpu_name):
    """
    Detects the hardware environment and sets up the appropriate TensorFlow distribution strategy.

    Parameters
    ----------
    tpu_name : str
        The name of the TPU to use (if available). If not provided, the function will run on CPU or GPU.

    Returns
    -------
    tpu : TPUClusterResolver
        The TPUClusterResolver object if the hardware environment is TPU, else None.
    strategy : tf.distribute.Strategy
        The appropriate TensorFlow distribution strategy based on the hardware environment.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)  # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on CPU')
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return tpu, strategy