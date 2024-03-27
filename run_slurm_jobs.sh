chmod a+x jobs/*.sh


srun \
  --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
  --job-name=master-thesis \
  --gpus=1 \
  --mem=64G \
  --cpus-per-task=8 \
  --partition=V100-32GB \
  --task-prolog="`pwd`/jobs/install.sh" \
  sh jobs/llm_pruner.sh
#  sh jobs/evaluate_test.sh
#  --partition=A100-40GB \
