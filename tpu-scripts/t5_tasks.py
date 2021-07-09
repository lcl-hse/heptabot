import t5
import os
import functools
import tensorflow as tf
import seqio

from t5.evaluation import metrics

DATA_DIR = os.path.realpath("./data/")

DEFAULT_SPM_PATH = os.path.realpath("./models/t5-tokenizer/spiece.model")

DEFAULT_VOCAB = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, 100)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}


def correction_preprocessor(ds):
  def normalize_text(text):
    """Remove quotes from a TensorFlow string."""
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"orig_text": ..., "corr_text": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["correction: ", normalize_text(ex["orig_text"])]),
        "targets": normalize_text(ex["corr_text"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def conll_preprocessor(ds):
  def normalize_text(text):
    """Remove quotes from a TensorFlow string."""
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"orig_text": ..., "corr_text": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["conll: ", normalize_text(ex["orig_text"])]),
        "targets": normalize_text(ex["corr_text"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def jfleg_preprocessor(ds):
  def normalize_text(text):
    """Remove quotes from a TensorFlow string."""
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"orig_text": ..., "corr_text": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["jfleg: ", normalize_text(ex["orig_text"])]),
        "targets": normalize_text(ex["corr_text"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def bea_preprocessor(ds):
  def normalize_text(text):
    """Remove quotes from a TensorFlow string."""
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"orig_text": ..., "corr_text": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["bea: ", normalize_text(ex["orig_text"])]),
        "targets": normalize_text(ex["corr_text"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


corr_tsv_path = {
    "train": os.path.join(DATA_DIR, "correct-train.tsv"),
    "validation": os.path.join(DATA_DIR, "correct-target.tsv")
}

def corr_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(corr_tsv_path[split])
  # Split each "<orig_text>\t<corr_text>" example into (orig_text, corr_text) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"orig_text": ... "corr_text": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["orig_text", "corr_text"], ex)))
  return ds

conll_tsv_path = {
    "train": os.path.join(DATA_DIR, "conll-train.tsv"),
    "validation": os.path.join(DATA_DIR, "conll-eval.tsv")
}

def conll_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(conll_tsv_path[split])
  # Split each "<orig_text>\t<corr_text>" example into (orig_text, corr_text) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"orig_text": ... "corr_text": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["orig_text", "corr_text"], ex)))
  return ds

jfleg_tsv_path = {
    "train": os.path.join(DATA_DIR, "jfleg-train.tsv"),
    "validation": os.path.join(DATA_DIR, "jfleg-eval.tsv")
}

def jfleg_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(jfleg_tsv_path[split])
  # Split each "<orig_text>\t<corr_text>" example into (orig_text, corr_text) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"orig_text": ... "corr_text": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["orig_text", "corr_text"], ex)))
  return ds

bea_tsv_path = {
    "train": os.path.join(DATA_DIR, "bea-train-strict.tsv"),
    "validation": os.path.join(DATA_DIR, "bea-eval.tsv")
}

def bea_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(bea_tsv_path[split])
  # Split each "<orig_text>\t<corr_text>" example into (orig_text, corr_text) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"orig_text": ... "corr_text": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["orig_text", "corr_text"], ex)))
  return ds


t5.data.TaskRegistry.add(
    "correct",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=corr_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=correction_preprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]
)

t5.data.TaskRegistry.add(
    "conll",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=conll_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=conll_preprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]
)

t5.data.TaskRegistry.add(
    "jfleg",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=jfleg_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=jfleg_preprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]
)

t5.data.TaskRegistry.add(
    "bea",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=bea_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=bea_preprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]
)

tasks_and_weights = [
  ('correct', 32668.0),
  ('conll', 28346.0),
  ('jfleg', 3016.0),
  ('bea', 34304.0)
]

t5.data.MixtureRegistry.add("correctit_all", tasks_and_weights)