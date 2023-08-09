#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:20:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=NONE
# SBATCH --arr=0
#SBATCH -J renm_chl
#SBATCH --out=outrch
#SBATCH --err=errrch

module use /pawsey/mwa/software/python3/modulefiles
module load python
module load astropy
module load numpy

# mkdir checks

#python channels_check.py
python rename_remain_channl.py
# obsarr=()
# while IFS= read -r line || [[ "$line" ]]; do
#        obsarr+=("$line")
# done < obsIDs
# OBSID=${obsarr[${SLURM_ARRAY_TASK_ID}]}


# mv images/${OBSID}-*-dirty.fits //astro/mwaeor/thimu/phase1/images/.
# mv images/${OBSID}-*-psf.fits //astro/mwaeor/thimu/phase1/images/.
# mv images/${OBSID}-*-image-pb.fits //astro/mwaeor/thimu/phase1/images/.
# mv images/${OBSID}-*-psf-pb.fits //astro/mwaeor/thimu/phase1/images/.