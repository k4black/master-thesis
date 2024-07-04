# master-thesis



## Installation

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

and install the requirements:
```bash
python -m pip install -r requirements.txt
```


Check partitions on https://pegasus.dfki.de/docs/slurm-cluster/partitions/
```bash
# sh run_slurm_jobs.sh [PARTITION] [SCRIPT]
# E.g.
sh run_slurm_jobs.sh "A100-40GB" ./jobs/run_original.sh
```