import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import t5
import t5_tasks

_t = tf.distribute.cluster_resolver.TPUClusterResolver()
TPU_ADDRESS = _t.get_master()
TPU_TOPOLOGY = "2x2"

MODEL_TYPE = os.environ["HPT_MODEL_TYPE"]
MODEL_DIR = "gs://heptabot/models/{}/tpu".format(model_type)
model_parallelism, train_batch_size, _ = {
    "medium": (2, 64, 8),
    "xxl": (8, 8, 1)}[model_type]

model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={"inputs": 512, "targets": 512},
        learning_rate_schedule=0.0025,
        iterations_per_loop=100,
)

_tname = {
    "correction": "correct",
    "bea": "bea",
    "conll": "conll",
    "jfleg": "jfleg"
}

model.predict(
    input_file="./raw/raw_input.txt",
    output_file="./raw/raw_output.txt",
    checkpoint_steps=1014000)