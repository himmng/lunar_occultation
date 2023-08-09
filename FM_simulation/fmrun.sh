#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=NONE
#SBATCH --arr=201-300
#SBATCH -J FM_calc_Y
#SBATCH --out=outfm
#SBATCH --err=errfm

#module use /pawsey/mwa/software/python3/modulefiles
module load singularity

OBSID=${SLURM_ARRAY_TASK_ID}
ch_lb=0

new_ob=$((ch_lb+OBSID))

#singularity exec /astro/mwaeor/thimu/docker/pawsey_v3.sif python3 FM.py ${OBSID}

singularity exec /astro/mwaeor/thimu/docker/pawsey_v3.sif python3 FM_year_SIM.py ${new_ob}

#singularity exec /astro/mwaeor/thimu/docker/pawsey_v3.sif python3 fm_flux.py ${OBSID}