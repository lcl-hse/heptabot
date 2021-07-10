TPU_IP=$(python -c "from tensorflow.distribute.cluster_resolver import TPUClusterResolver as tpu; t = tpu(); print(t.cluster_spec().as_dict()['worker'][0])")
TPU_NAME="grpc://"$TPU_IP

DIR=$(python -c "import os; print(os.path.realpath('.'))")
DATA_DIR=$DIR
MODEL_DIR="gs://heptabot/models/"$HPT_MODEL_TYPE"/tpu"

#CHECKPOINT_STEP
#TPU_TOPOLOGY

# Run inference
python -m t5.models.mesh_transformer_main \
  --module_import="t5_tasks" \
  --tpu="${TPU_NAME}" \
  --model_dir="${MODEL_DIR}" \
  --gin_param="MIXTURE_NAME = 'correctit_all'" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_TOPOLOGY}'" \
  --gin_param="utils.run.mode = 'infer'" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 8192)" \
  --gin_param="utils.run.sequence_length={'inputs': 512, 'targets': 512}" \
  --gin_param="decode_from_file.input_filename='${DATA_DIR}/raw/raw_input.txt'" \
  --gin_param="decode_from_file.output_filename='${DATA_DIR}/raw/raw_output.txt'" \
  --gin_param="utils.run.eval_checkpoint_step=${CHECKPOINT_STEP}"