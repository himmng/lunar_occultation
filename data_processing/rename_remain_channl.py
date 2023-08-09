import numpy as np
import os 

## log
## Checked Dec2015 obsids
## Checked Dec2015 obsids
## 


rem_obs = np.loadtxt('checks/Dec2015/rem_obsID', dtype=np.int32)
rem_chnl = np.loadtxt('checks/Dec2015/chlb', dtype=np.int32)
chub=768

# Dec2015 obsids 0 to 40 done
# 
for i in range(len(rem_chnl)):
    j = int(chub-rem_chnl[i])
    try:
        for k in range(0, j):
            if k<10:
                os.system('cp /astro/mwaeor/thimu/phase1/rem_images/Dec2015/%d-000%d-dirty.fits /astro/mwaeor/thimu/phase1/images/Dec2015/%d-0%d-dirty.fits'\
                %(rem_obs[i], k, rem_obs[i], int(rem_chnl[i]+k)))
                os.system('cp /astro/mwaeor/thimu/phase1/rem_images/Dec2015/%d-000%d-psf.fits /astro/mwaeor/thimu/phase1/images/Dec2015/%d-0%d-psf.fits'\
                %(rem_obs[i], k, rem_obs[i], int(rem_chnl[i]+k)))
            elif 10<=k<100:
                os.system('cp /astro/mwaeor/thimu/phase1/rem_images/Dec2015/%d-00%d-dirty.fits /astro/mwaeor/thimu/phase1/images/Dec2015/%d-0%d-dirty.fits'\
                %(rem_obs[i], k, rem_obs[i], int(rem_chnl[i]+k)))
                os.system('cp /astro/mwaeor/thimu/phase1/rem_images/Dec2015/%d-00%d-psf.fits /astro/mwaeor/thimu/phase1/images/Dec2015/%d-0%d-psf.fits'\
                %(rem_obs[i], k, rem_obs[i], int(rem_chnl[i]+k)))
            elif k>=100:
                os.system('cp /astro/mwaeor/thimu/phase1/rem_images/Dec2015/%d-0%d-dirty.fits /astro/mwaeor/thimu/phase1/images/Dec2015/%d-0%d-dirty.fits'\
                %(rem_obs[i], k, rem_obs[i], int(rem_chnl[i]+k)))
                os.system('cp /astro/mwaeor/thimu/phase1/rem_images/Dec2015/%d-0%d-psf.fits /astro/mwaeor/thimu/phase1/images/Dec2015/%d-0%d-psf.fits'\
                %(rem_obs[i], k, rem_obs[i], int(rem_chnl[i]+k)))
        print('all done')

    except:
        print('error', rem_chnl[i])

        

'''
obsids = np.loadtxt('../obsIDs_Dec2015', dtype=np.int32)

for i in range(len(obsids)):
    try:
        os.system('mv /astro/mwaeor/thimu/phase1/images/Dec2015/images/%d-*.fits /astro/mwaeor/thimu/phase1/images/Dec2015/'\
        %(obsids[i]))
        print('done', obsids[i], i)
    except:
        print('error', obsids[i], i)

'''