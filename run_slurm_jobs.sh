# Use
# sh run_slurm_jobs.sh PARTITION jobs/SCRIPT [ARGS for SCRIPT]
# e.g. sh run_slurm_jobs.sh A100-80GB jobs/our_pruner.sh [ARGS for our_pruner.sh]
# Partitions: https://pegasus.dfki.de/docs/slurm-cluster/partitions/
#   A100-40GB
#   A100-80GB
#   A100-PCI   (40GB)
#   H100       (80GB)
#   RTX3090    (24GB)
#   RTXA6000   (48GB)
#   V100-16GB
#   V100-32GB
#   batch      (RTX 6000, 24GB)

chmod a+x jobs/*.sh

srun \
  --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.06-py3.sqsh\
  --job-name=master-thesis \
  --gpus=1 \
  --mem=64G \
  --cpus-per-task=8 \
  --partition=$1 \
  --task-prolog="`pwd`/jobs/install.sh" \
  sh $2
  #"${@:3}"
