#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:50:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=NONE
# SBATCH --arr=12-22
#SBATCH -J EoR_moonP1_rem
#SBATCH --out=out_rem
#SBATCH --err=err_rem

module use /pawsey/mwa/software/python3/modulefiles
module load hyperdrive/chj
module load cotter
module load mwa-reduce
module load wsclean
module load chgcentre

sources=800


#obsarr=()
#while IFS= read -r line || [[ "$line" ]]; do
#        obsarr+=("$line")
#done < checks/Dec2015/rem_obsID
OBSID=1130163280
#{obsarr[${SLURM_ARRAY_TASK_ID}]}

#ch_ub=768

#chlb=()
#while IFS= read -r line
#do
#  chlb+=("$line")
#done < checks/Dec2015/chlb

#ch_lb=${chlb[${SLURM_ARRAY_TASK_ID}]}

#remain_ch=$((ch_ub-ch_lb))
# echo ${ch_ub} ${ch_lb} ${remain_ch}

wsclean -mwa-path /pawsey/mwa/ -apply-primary-beam -channels-out 768 \
    -channel-range 469 470 \
	-name ${OBSID} -size 1024 1024 -scale 0.51arcmin \
	-save-psf-pb -multiscale -auto-threshold 5 -niter 1 \
	-mgain 0.85 -weight uniform -pol I,XX,YY /astro/mwaeor/thimu/phase1/data/Sept2015/cal_ms/${OBSID}.ms
