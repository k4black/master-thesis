#!/bin/bash


# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  # install packages if not already installed
  apt update; apt install -y python3-venv ; apt clean
  python -m venv .venv || true
  . .venv/bin/activate
  python -m pip install wheel
  python -m pip install -r requirements.txt
#  python -m pip install -e ./external/llm_pruner

  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

