try:
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    #import pyfits
except ImportError:
    raise ImportError
    print('repo. missing!')

def fits_crop(path, on, off, psf, ob):

    ######## Data and WCS crop #############
    
    hdu_on = fits.open(path + on)
    head = hdu_on[0].header
    hdu_off = fits.open(path + off)
    hdu_psf = fits.open(path + psf)
    
    
    x_s, x_e, y_s, y_e = 1536-250,1536+250,1536-250, 1536+250# 2046-512,2046+512,2050-512,2050+512
    rms_s, rms_e = 0,450
    x_sp, x_ep, y_sp, y_ep = 1536-250,1536+250,1536-250, 1536+250 #2048-512,2048+512,2048-512,2048+512
    
    pix_size_deg = np.abs(float(head['cdelt1']))
    pix_area_deg_sq = pix_size_deg * pix_size_deg
    bmaj_deg=np.abs(float(head['bmaj'])) 
    bmin_deg=np.abs(float(head['bmin']))
    beam_area_deg_sq= 1.133 * bmaj_deg * bmin_deg
    n_pixels_per_beam=beam_area_deg_sq/pix_area_deg_sq
    
    

    psf_data=hdu_psf[0].data[0,0,:,:]
    psf_data = psf_data
    psf_data=np.nan_to_num(psf_data)
    psf_zoom=psf_data[y_sp:y_ep, x_sp:x_ep]

    psf_zoom=np.require(psf_zoom, dtype=np.float32)
    #psf_zoom_jyppix=psf_zoom/n_pixels_per_beam

    #fits.writeto('%s-%s-psf_jyppix.fits'%(on.split('-')[0], on.split('-')[1]), psf_zoom_jyppix,)
    #fits.update('%s-%s-psf_jyppix.fits'%(on.split('-')[0], on.split('-')[1]), psf_zoom_jyppix, header=hdu_psf[0].header)
    
    fits.writeto('%s-%s-psf.fits'%(on.split('-')[0], on.split('-')[1]), psf_zoom,)
    fits.update('%s-%s-psf.fits'%(on.split('-')[0], on.split('-')[1]), psf_zoom, header=hdu_psf[0].header)
    
    #except:
    #    print('psf fits missing')

    ####### Header related tasks ############
    
    del head[8]
    del head[8]
    del head['history']
    

    
    wcs = WCS(head)
    wcs=wcs.dropaxis(2)
  
    wcs_crop = wcs[y_s:y_e, x_s:x_e]
    head.update(wcs_crop.to_header())
    
    arr_on = hdu_on[0].data[0,0,:,:]
    crop_on = arr_on[y_s:y_e, x_s:x_e]
    
    arr_off = hdu_off[0].data[0,0,:,:]
    crop_off = arr_off[y_s:y_e, x_s:x_e]
    
    moon_minus_sky=crop_on-crop_off
    moon_minus_sky_jyppix=moon_minus_sky/n_pixels_per_beam
    difference_rms=np.sqrt(np.mean(np.square(moon_minus_sky[rms_s:rms_e, rms_s:rms_e])))
    difference_rms_jy_per_pixel=np.sqrt(np.mean(np.square(moon_minus_sky_jyppix[rms_s:rms_e, rms_s:rms_e])))
    if np.max(crop_on) == np.min(crop_on) or np.max(crop_off) == np.min(crop_on):
        print('fits problem')
        
    elif difference_rms_jy_per_pixel >= 10:
        print('high noise at %s on-off pair'%on)
        fits.writeto('%s-%s-diff.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky,)
        fits.update('%s-%s-diff.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky, header=head)

        #fits.writeto('%s-%s-diff_jyppix.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky_jyppix,)
        #fits.update('%s-%s-diff_jyppix.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky_jyppix, header=head)

    else:
        fits.writeto('%s-%s-diff.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky,)
        fits.update('%s-%s-diff.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky, header=head)

        #fits.writeto('%s-%s-diff_jyppix.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky_jyppix,)
        #fits.update('%s-%s-diff_jyppix.fits'%(on.split('-')[0], on.split('-')[1]), moon_minus_sky_jyppix, header=head)
    
            
#on = fits.open('set1/1213971808_1-0008-image-pb.fits')[0]
ob = np.loadtxt('list', dtype=np.int64)

for i in range(0, len(ob), 2):
   path='/astro/mwaeor/thimu/results/phase1fits/'
   for j in range(24):
   
       if j<10:	
           on='%d-000%d-image-pb.fits'%(ob[i], j)
           off='%d-000%d-image-pb.fits'%(ob[i+1], j)
           psf='%d-000%d-psf-pb.fits'%(ob[i], j)
           fits_crop(path=path, on = on, off=off, psf=psf, ob=ob[i])
           #os.system('*.fits /astro/mwaeor/thimu/set/set_%d'%)
       else:
           on='%d-00%d-image-pb.fits'%(ob[i], j)
           off='%d-00%d-image-pb.fits'%(ob[i+1], j)
           psf='%d-00%d-psf-pb.fits'%(ob[i], j)
           fits_crop(path=path, on = on, off=off, psf=psf, ob=ob[i])
'''

for i in range(0, len(ob), 2):
   path='/astro/mwaeor/thimu/set/set_0/'
   on='%d-MFS-image-pb.fits'%(ob[i],)
   off='%d-MFS-image-pb.fits'%(ob[i+1],)
   psf='%d-MFS-psf-pb.fits'%(ob[i],)
   fits_crop(path=path, on = on, off=off, psf=psf, ob=ob[i])
'''
