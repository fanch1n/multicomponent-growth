#!/bin/bash
#SBATCH --job-name=t1res       # create a short name for your job
#SBATCH --output=slurm-%A.out   # stdout file
#SBATCH --error=slurm-%A.err    # stderr file
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G        # memory per cpu-core (4G is default)
#SBATCH --time=23:59:00         # total run time limit (HH:MM:SS)

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "Executing on the machine:" $(hostname)

trajid=$1
snap="${trajid}_log.dat" # this is running configuration output

if [ ! -f "$snap" ]; then
    randseed=-1
    windowpos=-1
    loadconfig='0seed'
    mcstep=0
else
    header="$(head -n 1 $snap)"
    echo $header
    mcstep="$(echo $header | cut -d ' ' -f 2)"
    windowpos="$(echo $header | cut -d ' ' -f 4)"
    randseed="$(echo $header | cut -d ' ' -f 5)"
    echo $mcstep $randseed
    echo $windowpos
    loadconfig=$snap
    echo $loadconfig
fi

time \
     ./run \
    `#nw:`              $1       \
    `#sim_seed:`        $randseed\
    `#Li:`              1000     \
    `#Lj:`              56       \
    `#bond:`            4        \
    `#mu:`              -7.95    \
    `#mc_step:`         $mcstep  \
    `#N_sp:`            16       \
    `#N_cell:`          6	 \
    `#seed_label:`      0        \
    `#window_pos:`      $windowpos\
    `#load_path:`       $loadconfig

