#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=NONE
#SBATCH --arr=0-60
#SBATCH -J EoR_moonP1
#SBATCH --out=out_on
#SBATCH --err=err_on

module use /pawsey/mwa/software/python3/modulefiles
# module load python-singularity
module load hyperdrive/chj
#module load casa
#module load cotter
module load mwa-reduce
module load wsclean
module load chgcentre
#module load pyyaml
#module load python
#module load numpy

# OBSID=1208517584
sources=800
# add=1

obsarr=()
while IFS= read -r line || [[ "$line" ]]; do
        obsarr+=("$line")
done < obsIDs
OBSID=${obsarr[${SLURM_ARRAY_TASK_ID}]}

radec=()
while IFS= read -r line
do
  radec+=("$line")
done < lst
RADEC=${radec[${SLURM_ARRAY_TASK_ID}]}


### Puma sky-model
hyperdrive srclist-by-beam \
	/pawsey/mwa/software/python3/srclists/master/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_phase1+2.txt \
	-n ${sources} -i rts -o ao -b /pawsey/mwa/mwa_full_embedded_element_pattern.h5 \
	-m ../data/Sept2015/metafits/${OBSID}.metafits ../data/Sept2015/sourcelist/src-${sources}-${OBSID}.txt



#casa -c uvfits_to_ms.py ${OBSID}

#python 3c270_rem.py ${OBSID} $(($sources + $add))

#fixmwams ${OBSID}_sub.ms ${OBSID}.metafits

chgcentre ../data/Sept2015/ms/${OBSID}.ms ${RADEC} 

time calibrate -mwa-path /astro/mwaeor/jline/software -minuv 20 \
	-m ../data/Sept2015/sourcelist/src-${sources}-${OBSID}.txt \
	-applybeam -j 32 -i 300 ../data/Sept2015/ms/${OBSID}.ms sol-$sources-${OBSID}.bin

time applysolutions ../data/Sept2015/ms/${OBSID}.ms sol-$sources-${OBSID}.bin

wsclean -mwa-path /pawsey/mwa/ -apply-primary-beam -channels-out 768\
	-name ${OBSID} -size 3072 3072 -scale 0.51arcmin \
	-save-psf-pb -multiscale -auto-threshold 5 -niter 1\
	-mgain 0.85 -weight natural -pol I ../data/Sept2015/ms/${OBSID}.ms


mv *.fits ../images/.

# wsclean -mwa-path /pawsey/mwa/ -apply-primary-beam -channels-out 24\
#	-name ${OBSID}_1 -size 4096 4096 -scale 0.25arcmin \
#	-save-psf-pb -multiscale -auto-threshold 5 -niter 10000\
#	-mgain 0.85 -weight natural -pol I ${OBSID}.ms

# mv *-*-image-pb.fits /astro/mwaeor/thimu/results/moon/April2018/.
# mv *-*-psf.fits /astro/mwaeor/thimu/results/moon/April2018/.
# mv *-*-psf-pb.fits /astro/mwaeor/thimu/results/moon/April2018/.
# mv *-*-dirty.fits /astro/mwaeor/thimu/results/moon/April2018/.
# mv *.fits /astro/mwaeor/thimu/phase2/April2018/rest_images/.