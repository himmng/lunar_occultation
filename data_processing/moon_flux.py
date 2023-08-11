try:

    import numpy as np
    import astropy.units as u
    from astropy.io import fits
    from scipy import signal

    np.seterr(divide='ignore', invalid='ignore')
    
except ImportError:
    
    raise ImportError('repo lost')

class Process_Data(object):
    
    """
    
    The class specific to Lunar occultation, uses the radio images to perform image-differencing
    and modelling
    
    Available Functions:  get_difference_cropped_image, get_flux_n_difference_images, get_flux_n_images

    Args:
        object (object): None
        
    """
    def __init__(self) -> None:
        
        self.moon_diameter_arcmin = 33.29*u.arcmin  ## Moon's diameter in arcmins
        self.moon_diameter_deg = self.moon_diameter_arcmin.to(u.deg)
        self.moon_radius_deg = self.moon_diameter_deg/2.  ## Moon's radius [deg.]
        self.moon_area_deg_sq = np.pi*(self.moon_radius_deg)**2
        
    
    def get_difference_cropped_image(self, ON, OFF, PSF, crop_pixels, image_pixels=None, norm_psf=True, avoid_difference=False):
        
        """
        
        Get the cropped difference of ON-Moon, OFF-Moon images.

        Args:
        
            ON (str): path to the ON-Moon image; e.g. fits file
            OFF (str): path to the OFF-Moon image; e.g. fits file
            PSF (str): path to the PSF of either ON/OFF-Moon image; e.g. fits file
            crop_pixels (int): Number of pixels required in the cropped images
            image_pixels (int, optional): Number of pixels in the images
            norm_psf (bool, optional): Normalise the PSF image. Defaults to True.
            avoid_difference (bool, optional): do not take difference image. Defauls to False
            
        Returns:
        
            if avoid_difference: False
            diff_crop (ndarray): 2D cropped image ON-OFF Moon (units, Jy/pixels)
            
            if avoid_difference: True
            else: ON-Moon_crop, OFF-Moon_crop (ndarray): 2D cropped image (units, Jy/pixels)
            
            psf_crop ( ndarray): 2D cropped PSF (units, Jy/beam)
            n_pixels_per_beamo (float): number of pixels in the synthesised beam
            pix_size_deg (float): size of the image pixel, units: degree
            beam_area_deg_sq (float): angular size of the beam, units: steradian

        """
        
        imON = fits.open(ON)[0]
        imOFF = fits.open(OFF)[0] 
        psf = fits.open(PSF)[0]
            
        header = imON.header
        pix_size_deg = np.abs(float(header['cdelt1']))*u.deg ## pixel size in degrees
        pix_area_deg_sq = pix_size_deg * pix_size_deg  ## angular area of pixels in deg^2

        bmaj_deg = np.abs(float(header['bmaj']))* u.deg ## synthesized beam size major axis, minor axis
        bmin_deg = np.abs(float(header['bmin']))* u.deg
        fwhm_t_sig = 2 * np.pi * (1. / (8 * np.log(2)) ** 0.5) ** 2 ## gives 1.133 factor
        beam_area = fwhm_t_sig * bmaj_deg * bmin_deg
        n_pixels_per_beam = beam_area/pix_area_deg_sq
        
        if image_pixels == None:
        
            image_pixels = len(imON.data[0,0,:,:])
            
        # make data
        
        imON = imON.data.reshape(image_pixels, image_pixels) / n_pixels_per_beam
        imOFF = imOFF.data.reshape(image_pixels, image_pixels) / n_pixels_per_beam
        psf = psf.data.reshape(image_pixels, image_pixels)
        
        if norm_psf == True:
            
            psf = psf/np.max(psf)

        psf_crop = psf[int(image_pixels/2) - int(crop_pixels) : \
                            int(image_pixels/2) + int(crop_pixels), \
                            int(image_pixels/2) - int(crop_pixels) : \
                            int(image_pixels/2) + int(crop_pixels)]

        if avoid_difference != True:
            
            diff = imON - imOFF
            diff_crop = diff[int(image_pixels/2) - int(crop_pixels) : \
                                    int(image_pixels/2) + int(crop_pixels), \
                                    int(image_pixels/2) - int(crop_pixels) : \
                                    int(image_pixels/2) + int(crop_pixels)]
    
            return diff_crop, psf_crop, n_pixels_per_beam, pix_size_deg, beam_area

        else:
            
            imON = imON[int(image_pixels/2) - int(crop_pixels) : \
                                    int(image_pixels/2) + int(crop_pixels), \
                                    int(image_pixels/2) - int(crop_pixels) : \
                                    int(image_pixels/2) + int(crop_pixels)]
            
            imOFF = imOFF[int(image_pixels/2) - int(crop_pixels) : \
                                    int(image_pixels/2) + int(crop_pixels), \
                                    int(image_pixels/2) - int(crop_pixels) : \
                                    int(image_pixels/2) + int(crop_pixels)]
            
            return imON, imOFF, psf_crop, n_pixels_per_beam, pix_size_deg, beam_area

        
    def get_flux_n_difference_images(self, diff, PSF, n_pixels_per_beam, pix_size_deg,\
                                    beam_area, RFI_pixels=None, RFI_broadenning=False, extent=None,\
                                    rms_extent=None, single_RFImodel=False):
        
        
        """
        
        Get the flux-density from the differnce images
        
        Args:
        
            diff (ndarray): 2D difference (ON-OFF Moon) image, or ON/OFF Moon image. (units: Jy/pixels)
            psf (ndarray): normalised PSF should have same shape as diff image. (units: Jy/beam)
            n_pixels_per_beam (int, float): number of pixels in the synthesised beam.
            pix_size_deg (float): angular size of the pixel, unit: degrees
            beam_area (float): synthesised beam area in degree^2
            
            processing Args:

            RFI_pixels (int, optional): number of pixels for the quasi-specular Earthshine mask. Defaults to 12.
            RFI_broadenning (bool): if the quasi-specular gets smeared due to the motion of the Moon. Defaults to False.
            extent (list[int,]): [RA_min(left pixel), RA_max(right pixel), DEC_min(bottom_pixel), DEC_max(top_pixel)]
            RA_min (int, optional) Defaults to 12.
            RA_max (int, optional) Defaults to 20.
            DEC_min (int, optional) Defaults to 8.
            DEC_max (int, optional): Defaults to 14.
            
            RMS error evaluation Args:
            rms_extent (list[int,]): [RA_min(left pixel), RA_max(right pixel), DEC_min(bottom_pixel), DEC_max(top_pixel)]
            
            Single RFI model Args: 
            (merges diffuse + quasi specular model)
            
            single_RFImodel (bool, optional): if using only single RFI model in the analysis. Defaults to False
            
        Returns:
        
            tuple of ndarrys: 
            G_image_shift (ndarray): Moon mask convolved with PSF, units: Jy/pixels
            RFI_convolved_jyppix (ndarray): Quasi Specular Mask covolved with PSF, units: Jy/pixels
            Reconstructed_moon (ndarray): Reconstructed model of Moon, Disk flux-density multiplied with Moon mask, units: Jy/pixels
            Reconstructed_RFI (ndarray): Reconstructed RFI model, RFI flux-density multiplied with RFI mask, units: Jy/pixels
            Residual (ndarray): Difference Image - Reconstructed Moon - Reconstucted RFI, units: Jy/pixels
            
            S_moon (float): flux-density of the disk component, units: Jy
            S_moon_tot_Jy (float): Summed over Moon's disk flux-density, units: Jy
            S_moon_error (float): RMS error in disk flux-density, units: Jy
            S_moon_tot_jy_error (float): weighted over angular size RMS error, units: Jy,
                    
            S_RFI (float): flux-density of the disk component, units: Jy
            S_RFI_total_Jy (float): Summed over Moon's disk flux-density, units: Jy
            S_RFI_error (float): RMS error in disk flux-density, units: Jy
            S_RFI_tot_jy_error (float): weighted over angular size RMS error, units: Jy,
            
            Difference_rms (float): RMS noise in the difference image, units: Jy
            
        """
        
        if diff.shape != PSF.shape:
            
            raise TypeError('diff and psf shape mismatch!')
        
        PSF_jyppix = PSF/n_pixels_per_beam
        image_length, image_height = diff.shape

        # moon mask, creating unity mask of the angular size of the moon

        moon_radius_pix = np.round(self.moon_radius_deg/pix_size_deg)
        Moon_mask = np.zeros(diff.shape)
        x_cord, y_cord = np.ogrid[-int(image_length/2) : int(image_length/2),\
                        -int(image_height/2) : int(image_height/2)]
        mask = x_cord*x_cord + y_cord*y_cord <= moon_radius_pix.value * moon_radius_pix.value
        Moon_mask[mask]=1
        
        ### RFI model mask
        
        RFI_mask = np.zeros(Moon_mask.shape)
        
        if RFI_pixels == None:
            
            ## based on calculations 22 arcsec size used 
            RFI_size_deg = (22*u.arcsec).to(u.deg)
            RFI_pixels = np.round(RFI_size_deg/pix_size_deg)

        if RFI_broadenning == True:

           RFI_mask[int(image_length/2) - extent[0] : int(image_length/2) + extent[1],\
                        int(image_height/2) - extent[2] : int(image_height/2) + extent[3]] = 1
           
        else: 
            
            x_cord, y_cord = np.ogrid[int(image_length/2):int(image_length/2),\
                                    int(image_length/2):int(image_length/2)]
            mask = x_cord*x_cord + y_cord*y_cord <= RFI_pixels.value * RFI_pixels.value
            RFI_mask[mask] = 1
            
        ### convolving moon mask with psf
        G_image_shift = signal.convolve2d(Moon_mask, PSF_jyppix, mode='same')

        ### convolving psf with rfi mask
        RFI_convolved_image = signal.convolve2d(PSF, RFI_mask, mode='same')
        RFI_convolved_jyppix = RFI_convolved_image/n_pixels_per_beam  
        
        ## doing linear algebra (likelihood)
        if single_RFImodel != True:
            
            vec_RFI = RFI_convolved_jyppix.flatten('C') 
            vec_D = diff.flatten('C')
            vec_G = G_image_shift.flatten('C')
            H2_matrix = np.column_stack((vec_G,vec_RFI))
            theta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(H2_matrix.T, H2_matrix)),\
                                            H2_matrix.T),vec_D)
            print(np.linalg.det(np.matmul(H2_matrix.T,H2_matrix)))
            ## moon's flux, RFI, earthshine components
            S_moon = theta_hat[0]
            S_RFI = theta_hat[1] 

            ## total flux
            S_moon_tot_Jy = np.sum(S_moon * Moon_mask)
            S_RFI_total_Jy = np.sum(S_RFI * RFI_mask)

            ## reconstructin moon, RFI with moon's flux
            Reconstructed_moon = S_moon * G_image_shift
            Reconstructed_RFI = S_RFI * RFI_convolved_jyppix
            Residual = diff - Reconstructed_moon - Reconstructed_RFI
        
        elif single_RFImodel == True:
            
            vec_D = diff.flatten('C')
            vec_D = vec_D.reshape((len(vec_D),1))
            vec_G = G_image_shift.flatten('C')
            vec_G = vec_G.reshape((len(vec_G),1))
            vec_PSF = PSF_jyppix.flatten('C')
            vec_PSF = vec_PSF.reshape((len(vec_PSF),1))
            H2_matrix = vec_G 
            print(np.linalg.det(np.matmul(H2_matrix.T,H2_matrix)))
            theta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(H2_matrix.T,H2_matrix)),H2_matrix.T),vec_D)
            S_moon = theta_hat
            S_moon_tot_Jy = np.sum(S_moon*Moon_mask)
            Reconstructed_moon = S_moon*G_image_shift
            Residual = diff - Reconstructed_moon

        
        ### RMS error evalutate
        if rms_extent == None:
            rms_extent = [50,150,50,150]
        
        Difference_rms = np.sqrt(np.mean(np.square(diff[rms_extent[0]:rms_extent[1],rms_extent[2]:rms_extent[3]])))
        covariance_matrix_theta=(Difference_rms**2) * (np.linalg.inv(np.matmul(H2_matrix.T,H2_matrix)))

        S_moon_error=np.sqrt(covariance_matrix_theta[0,0])
        S_RFI_error=np.sqrt(covariance_matrix_theta[1,1])
        
        S_moon_tot_jy_error = S_moon_error * np.sqrt(self.moon_area_deg_sq/beam_area)
        S_RFI_tot_jy_error = S_RFI_error * np.sqrt(self.moon_area_deg_sq/beam_area)
        
        return G_image_shift, RFI_convolved_jyppix, Reconstructed_moon, Reconstructed_RFI, Residual, \
                S_moon, S_moon_tot_Jy, S_moon_error, S_moon_tot_jy_error, \
                    S_RFI, S_RFI_total_Jy, S_RFI_error, S_RFI_tot_jy_error, Difference_rms
                    