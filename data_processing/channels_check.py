import numpy as np
import os 

hyp_path = '/astro/mwaeor/thimu/phase1/images/Dec2015/'
checks = 'checks/'
obsids = np.loadtxt('../obsIDs_Dec2015', dtype=np.int32)

f2 = open('checks/Dec2015/chlb', 'w+')
f3 = open('checks/Dec2015/rem_obsID', 'w+')

for i in range(0,len(obsids)):
    try:
        os.system('ls %s%d-*-dirty.fits > %s%d_dirty'%(hyp_path, obsids[i], checks, obsids[i]))
        os.system('ls %s%d-*-psf-pb.fits > %s%d_psf'%(hyp_path, obsids[i], checks, obsids[i]))

        f = open('%s%d_dirty'%(checks, obsids[i]), 'r')
        f = f.readlines()
        f1 = open('%s%d_psf'%(checks, obsids[i]), 'r')
        f1 = f1.readlines()

        if len(f) == 769:
            print('full set done', i)
        elif len(f) == 768:
            print('full set done', i)
        elif len(f) > 768:
            print('extra set', i)
        else:
            print('set %s, %d'%(obsids[i],i))
            f3.writelines('%s\n'%obsids[i])
            f2.writelines('%s\n'%len(f))
    except:
        print('error at', i)


