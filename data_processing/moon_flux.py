import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import signal

np.seterr(divide='ignore', invalid='ignore')

def dirty_diff(ON, OFF, PSF, im_pixels, crop_pixels, norm_psf=True):
    '''
    This function uses difference images of ON and OFF Moon 
    and produce difference images as numpy arrays
    '''

    imo = fits.open(ON)[0] ## ON Moon
    imf= fits.open(OFF)[0] ## OFF Moon
    psfo = fits.open(PSF)[0] ## PSF ON Moon
        
    heado = imo.header ## Header ON Moon
    pix_size_dego = np.abs(float(heado['cdelt1'])) ## pixel size in degrees
    pix_area_deg_sqo = pix_size_dego * pix_size_dego ## angular area of pixels in degrees^2

    bmaj_dego=np.abs(float(heado['bmaj'])) ## synthesized beam size
    bmin_dego=np.abs(float(heado['bmin']))
    beam_area_deg_sqo= 1.133 * bmaj_dego * bmin_dego
    n_pixels_per_beamo=beam_area_deg_sqo/pix_area_deg_sqo
        
        
        
    # make data

    diff = (imo.data.reshape(im_pixels, im_pixels)/n_pixels_per_beamo) - \
                (imf.data.reshape(im_pixels, im_pixels)/n_pixels_per_beamo)
            
    diff_crop = diff[int(im_pixels/2)-int(crop_pixels) : \
                            int(im_pixels/2)+int(crop_pixels), \
                            int(im_pixels/2)-int(crop_pixels) : \
                            int(im_pixels/2)+int(crop_pixels)]

    psf = psfo.data.reshape(im_pixels, im_pixels)
    if norm_psf == True:
        psf = psf/np.max(psf)
    psf_crop = psf[int(im_pixels/2)-int(crop_pixels) : \
                         int(im_pixels/2)+int(crop_pixels), \
                        int(im_pixels/2)-int(crop_pixels) : \
                         int(im_pixels/2)+int(crop_pixels)]
 
    return diff_crop, psf_crop, n_pixels_per_beamo, pix_size_dego, beam_area_deg_sqo

def dirty(ON, PSF, im_pixels, crop_pixels, norm_psf=True):
    '''
    This function uses ON Moon images
    and produce crop images as numpy arrays
    '''
    imo = fits.open(ON)[0] ## ON Moon
    
    psfo = fits.open(PSF)[0] ## PSF 
        
    heado = imo.header ## Header ON Moon
    pix_size_dego = np.abs(float(heado['cdelt1'])) ## pixel size
    pix_area_deg_sqo = pix_size_dego * pix_size_dego ## area pixel^2

    bmaj_dego=np.abs(float(heado['bmaj'])) ## synthesized beam size
    bmin_dego=np.abs(float(heado['bmin']))
    beam_area_deg_sqo= 1.133 * bmaj_dego * bmin_dego
    n_pixels_per_beamo=beam_area_deg_sqo/pix_area_deg_sqo
        
        
    # make data

    diff = (imo.data.reshape(im_pixels, im_pixels)/n_pixels_per_beamo)
            
    diff_crop = diff[int(im_pixels/2)-int(crop_pixels) : \
                            int(im_pixels/2)+int(crop_pixels), \
                            int(im_pixels/2)-int(crop_pixels) : \
                            int(im_pixels/2)+int(crop_pixels)]

    psf = psfo.data.reshape(im_pixels, im_pixels)
    if norm_psf == True:
        psf = psf/np.max(psf)
    psf_crop = psf[int(im_pixels/2)-int(crop_pixels) : \
                         int(im_pixels/2)+int(crop_pixels), \
                        int(im_pixels/2)-int(crop_pixels) : \
                         int(im_pixels/2)+int(crop_pixels)]
 
    return diff_crop, psf_crop, n_pixels_per_beamo, pix_size_dego, beam_area_deg_sqo
    
def mask(diff, psf, n_pixels_per_beam, pix_size_deg, beam_area_deg_sq, \
         RFI_broadn = True, RFI_pix=3, hl=3, hr=5, vl=2, vr=3):
    '''
    This function creates Moon mask and perform maximum likelihood estimation to get
    flux density of Moon and RFI components'''
    
    moon_diameter_arcmin = 33.29   ## Moon's diameter in arcmins
    moon_diameter_deg = moon_diameter_arcmin/60
    moon_radius_deg = moon_diameter_deg/2.  ## Moon's radius [deg.]
    #area of moon in deg_sq
    moon_area_deg_sq = np.pi*(moon_radius_deg)**2

    steradian_in_sq_deg = (180/np.pi)**2
    
    Omega = moon_area_deg_sq/steradian_in_sq_deg #solid angle subtended by Moon in Steradians
    
    psf_jyppix = psf/n_pixels_per_beam
    
    #diff = np.load(moon_im)
    image_length = diff.shape[0]
    image_height = diff.shape[1]

    # moon mask
    moon_radius_pix = np.round(moon_radius_deg/pix_size_deg)
    moon_mask = np.zeros((image_length,image_height))
    a,b = (image_length/2)-1, (image_height/2)-1
    y,x = np.ogrid[-a:image_length-a, -b:image_height-b]
    mask = x*x + y*y <= moon_radius_pix*moon_radius_pix
    moon_mask[mask]=1
    
    ### RFI model
    if RFI_broadn == True:
        rfi_model_mask = np.zeros((image_length,image_height))
        rfi_model_mask[int(b)-vl:int(b)+vr,int(a)-hl:int(a)+hr] =\
        np.ones([int(vl+vr),int(hl+hr)])
        RFI_broadening = rfi_model_mask
    else: 
        rfi_model_mask = np.zeros((image_length,image_height))
        rfi_model_mask[int(b)- RFI_pix:int(b)+RFI_pix,int(a)-RFI_pix:int(a)+RFI_pix] =\
        np.ones([int(2*RFI_pix),int(2*RFI_pix)])
        RFI_broadening = rfi_model_mask
    
    ### convolving moon mask with psf
    G_image_shift = signal.convolve2d(moon_mask, psf_jyppix, mode='same')

    ### convolving psf with rfi mask
    RFI_convolved_image = signal.convolve2d(psf, RFI_broadening, mode='same')
    RFI_convolved_shift = RFI_convolved_image
    RFI_convolved_shift_jyppix = RFI_convolved_shift/n_pixels_per_beam  
    
    ## doing linear algebra (likelihood)
    vec_RFI = RFI_convolved_shift_jyppix.flatten('F') 
    vec_D = diff.flatten('F')
    vec_G = G_image_shift.flatten('F')
    vec_PSF = psf_jyppix.flatten('F')
    

    H2_matrix = np.column_stack((vec_G,vec_RFI))
    theta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(H2_matrix.T,H2_matrix)),\
                                    H2_matrix.T),vec_D)

    ## moon's flux, RFI, earthshine components
    S_moon = theta_hat[0]
    S_RFI = theta_hat[1] 

    ## total flux
    S_moon_tot_Jy = np.sum(S_moon*moon_mask)
    S_RFI_total_Jy = np.sum(S_RFI*RFI_broadening)

    ## reconstructin moon, RFI with moon's flux
    reconstructed_moon = S_moon*G_image_shift
    reconstructed_RFI = S_RFI*RFI_convolved_shift_jyppix
    residual = diff-reconstructed_moon-reconstructed_RFI
    
    ### RMS errors
    rms_start_x = rms_start_y = 0
    rms_end_x = 20
    rms_end_y =  120
    
    difference_rms=np.sqrt(np.mean(np.square(diff[rms_start_x:rms_end_x,rms_start_y:rms_end_y]*n_pixels_per_beam)))
    covariance_matrix_theta=(difference_rms**2)*(np.linalg.inv(np.matmul(H2_matrix.T,H2_matrix)))

    S_moon_error=np.sqrt(covariance_matrix_theta[0,0])
    S_RFI_error=np.sqrt(covariance_matrix_theta[1,1])


    S_moon_error_Jy_beam=S_moon_error*n_pixels_per_beam
    S_RFI_error_Jy_beam=S_RFI_error*n_pixels_per_beam

    S_moon_tot_Jy_error=S_moon_error_Jy_beam*np.sqrt(moon_area_deg_sq/beam_area_deg_sq)
    S_RFI_tot_Jy_error=S_RFI_error_Jy_beam*np.sqrt(moon_area_deg_sq/beam_area_deg_sq)
    
    return G_image_shift, RFI_convolved_shift_jyppix, reconstructed_moon, reconstructed_RFI, residual, \
            S_moon, S_moon_error, S_moon_error_Jy_beam, S_moon_tot_Jy, S_moon_tot_Jy_error, \
            S_RFI, S_RFI_error, S_RFI_error_Jy_beam, S_RFI_total_Jy, S_RFI_tot_Jy_error, \
            difference_rms

def mask_onlyMoon(diff, psf, n_pixels_per_beam, pix_size_deg, beam_area_deg_sq, \
         RFI_broadn = True, RFI_pix=3, hl=3, hr=5, vl=2, vr=3):
    '''
    This funcation only assumes a single RFI model of Moon'''
    moon_diameter_arcmin = 33.29
    moon_diameter_deg = moon_diameter_arcmin/60
    moon_radius_deg = moon_diameter_deg/2. 
    #area of moon in deg_sq
    moon_area_deg_sq = np.pi*(moon_radius_deg)**2

    steradian_in_sq_deg = (180/np.pi)**2
    #solid angle subtended by Moon in Steradians
    Omega = moon_area_deg_sq/steradian_in_sq_deg
    
    psf_jyppix = psf/n_pixels_per_beam
    
    #diff = np.load(moon_im)
    image_length = diff.shape[0]
    image_height = diff.shape[1]

    # moon mask
    moon_radius_pix = np.round(moon_radius_deg/pix_size_deg)
    moon_mask = np.zeros((image_length,image_height))
    a,b = (image_length/2)-1, (image_height/2)-1
    y,x = np.ogrid[-a:image_length-a, -b:image_height-b]
    mask = x*x + y*y <= moon_radius_pix*moon_radius_pix
    moon_mask[mask]=1
    
    ### RFI model
    if RFI_broadn == True:
        rfi_model_mask = np.zeros((image_length,image_height))
        rfi_model_mask[int(b)-vl:int(b)+vr,int(a)-hl:int(a)+hr] = np.ones([int(vl+vr),int(hl+hr)])
        RFI_broadening = rfi_model_mask
    else: 
        rfi_model_mask = np.zeros((image_length,image_height))
        rfi_model_mask[int(b)- RFI_pix:int(b)+RFI_pix,int(a)-RFI_pix:int(a)+RFI_pix] = np.ones([int(2*RFI_pix),int(2*RFI_pix)])
        RFI_broadening = rfi_model_mask
    
    ### convolving moon mask with psf
    G_image_shift = signal.convolve2d(moon_mask, psf_jyppix, mode='same')
    
    ## doing linear algebra (likelihood)
    vec_D = diff.flatten('F')
    vec_D = vec_D.reshape((len(vec_D),1))
    vec_G = G_image_shift.flatten('F')
    vec_G = vec_G.reshape((len(vec_G),1))
    vec_PSF = psf_jyppix.flatten('F')
    vec_PSF = vec_PSF.reshape((len(vec_PSF),1))
    H2_matrix = vec_G 
    
    theta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(H2_matrix.T,H2_matrix)),H2_matrix.T),vec_D)

    ## moon's flux, RFI, earthshine components
    S_moon = theta_hat

    ## total flux
    S_moon_tot_Jy = np.sum(S_moon*moon_mask)

    ## reconstructin moon, RFI with moon's flux
    reconstructed_moon = S_moon*G_image_shift
    residual = diff-reconstructed_moon
    
    ### RMS errors
    rms_start_x = rms_start_y = 0
    rms_end_x = rms_end_y =  120
    
    difference_rms=np.sqrt(np.mean(np.square(diff[rms_start_x:rms_end_x,rms_start_y:rms_end_y]*n_pixels_per_beam)))
    covariance_matrix_theta=(difference_rms**2)*(np.linalg.inv(np.matmul(H2_matrix.T,H2_matrix)))

    S_moon_error=np.sqrt(covariance_matrix_theta[0,0])


    S_moon_error_Jy_beam=S_moon_error*n_pixels_per_beam

    S_moon_tot_Jy_error=S_moon_error_Jy_beam*np.sqrt(moon_area_deg_sq/beam_area_deg_sq)
    
    return G_image_shift, reconstructed_moon, residual, \
            S_moon, S_moon_error, S_moon_error_Jy_beam, S_moon_tot_Jy, S_moon_tot_Jy_error, \
            difference_rms
