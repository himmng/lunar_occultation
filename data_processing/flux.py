try:
    
    import sys
    import argparse
    import numpy as np
    from moon_flux import Process_Data

except ImportError:
    
    raise ImportError('repo. absent!')


parser = argparse.ArgumentParser(description='Running the Lunar image processor')
parser.add_argument('-p', '--datapath', type=str, metavar='', required=True, help='path of the dataset, e.g fits images')
parser.add_argument('-s', '--savepath', type=str, metavar='', required=True, help='path to save the datasets, e.g numpy arrays, cropped images')
parser.add_argument('-o', '--obsIDpath', type=str, metavar='', required=True, help='path of the observational IDs, e.g GPSTIME array')
parser.add_argument('-i', '--obsID_index', type=int, metavar='', required=True, help='index of the obsID to process')
args = parser.parse_args()

#datapath = 'testimages/'
#savepath = 'flux_calc/'

ind = int(args.obsID_index)# int(sys.argv[1])
obsID = np.loadtxt(args.obsIDpath, dtype=np.int32)

ON_ind = int(2*ind)
OFF_ind = int(ON_ind + 1)
ON_obsID = obsID[ON_ind]
OFF_obsID = obsID[OFF_ind]

crop_pixels = 256
image_pixels = 4096
channels = 24

def process(datapath=None, savepath=None, ON_obsID=None, OFF_obsID=None,\
            crop_pixels=None, image_pixels=None, channels=None):
    """
    
    Processing the images
    
    Args:
        datapath (str, optional): path to the dataset, e.g. images.
        savepath (str, optional): save path location. 
        ON_obsID (int, optional): GPSTIME ON-Moon.
        OFF_obsID (int, optional): GPSTIME OFF-Moon.
        image_pixels (int, optional): number of pixels in the images.
        channels (int, optional): number of frequency channels.

    Raises:
    
        IOError: file absent error
        
    """
    
    Diff = np.zeros(shape = (channels, int(2*crop_pixels), int(2*crop_pixels)))
    PSF = np.zeros(shape = Diff.shape)
    Recon_Moon = np.zeros(shape = Diff.shape)
    Recon_RFI = np.zeros(shape = Diff.shape)
    Moon_conv_PSF = np.zeros(shape = Diff.shape)
    RFI_conv_PSF = np.zeros(shape= Diff.shape)

    Diff_rms = np.zeros(channels)
    S_disk = np.zeros(channels)
    S_disk_err = np.zeros(channels)
    S_spec = np.zeros(channels)
    S_spec_err = np.zeros(channels)
    npix_per_beam = np.zeros(channels)
    pix_size_deg = np.zeros(channels)
    
    for channel in range(channels):

        try:
            if channel < 10:
                imON = datapath+'%d-000%d-image-pb.fits'%(ON_obsID, channel)
                imOFF= datapath+'%d-000%d-image-pb.fits'%(OFF_obsID, channel)
                PSF= datapath+'%d-000%d-psf-pb.fits'%(ON_obsID, channel)   
                
            elif 10 <= channel < 100:
                imON = datapath+'%d-00%d-image-pb.fits'%(ON_obsID, channel)
                imOFF= datapath+'%d-00%d-image-pb.fits'%(OFF_obsID, channel)
                PSF= datapath+'%d-00%d-psf-pb.fits'%(ON_obsID, channel)
                
            else:
                imON = datapath+'%d-0%d-image-pb.fits'%(ON_obsID, channel)
                imOFF= datapath+'%d-0%d-image-pb.fits'%(OFF_obsID, channel)
                PSF= datapath+'%d-0%d-psf-pb.fits'%(ON_obsID, channel)
                
        except IOError:
            raise IOError('error at', ON_obsID, OFF_obsID)
  
        A = Process_Data().get_difference_cropped_image(ON=imON,
                                                        OFF=imOFF,
                                                        PSF=PSF,
                                                        crop_pixels=crop_pixels,
                                                        image_pixels=image_pixels,
                                                        norm_psf=True,
                                                        avoid_difference=False)
        if np.isnan(A[0][0][0]) == np.nan:
            pass
        
        else:
            
            B = Process_Data().get_flux_n_difference_images(diff=A[0],
                                                            PSF=A[1],
                                                            n_pixels_per_beam=A[2],
                                                            pix_size_deg=A[3],
                                                            beam_area=A[4],
                                                            RFI_pixels=None,
                                                            RFI_broadenning=False,
                                                            extent=None,
                                                            rms_extent=None,
                                                            single_RFImodel=False)
            
            Diff[channel] = A[0]
            PSF[channel] = A[1]
            npix_per_beam[channel] = A[2]
            pix_size_deg[channel] = A[3]
            
            Moon_conv_PSF[channel] = B[0]
            RFI_conv_PSF[channel] = B[1]
            Recon_Moon[channel] = B[2]
            Recon_RFI[channel] = B[3]
            
            S_disk[channel] = B[6]
            S_disk_err[channel] = B[7]
            
            S_spec[channel] = B[10]
            S_spec_err[channel] = B[11]

            Diff_rms[channel] = B[13]

        print('done %d channels'%channel)

    np.save(savepath+'%d_diff_crop'%ON_obsID, Diff)    
    np.save(savepath+'%d_PSF_crop'%ON_obsID, PSF)
    np.save(savepath+'%d_Recon_Moon_crop'%ON_obsID, Recon_Moon)
    np.save(savepath+'%d_Recon_RFI_crop'%ON_obsID, Recon_RFI)
    
    np.save(savepath+'%d_Moon_conv_PSF'%ON_obsID, Moon_conv_PSF)    
    np.save(savepath+'%d_RFI_conv_PSF'%ON_obsID, RFI_conv_PSF)
    np.savetxt(savepath+'%d_pix_deg'%ON_obsID, pix_size_deg)
    np.savetxt(savepath+'%d_n_pix_beam'%ON_obsID, npix_per_beam)

    np.savetxt(savepath+'%d_S_disk'%ON_obsID, S_disk)
    np.savetxt(savepath+'%d_S_disk_err'%ON_obsID, S_disk_err)
    
    np.savetxt(savepath+'%d_S_spec'%ON_obsID, S_spec)
    np.savetxt(savepath+'%d_S_spec_err'%ON_obsID, S_spec_err)
    
    np.save(savepath+'%d_diff_RMS_'%ON_obsID, Diff_rms)
    

if __name__ == '__main__':
    
    process(datapath=args.datapath, savepath=args.savepath, ON_obsID=ON_obsID, OFF_obsID=OFF_obsID,\
            crop_pixels=crop_pixels, image_pixels=image_pixels, channels=channels)

    if args.quiet:
        print('done')
    elif args.verbose:
        print('ON-OFF pair %d, %d obsIDs done'%(ON_obsID, OFF_obsID))
    else:
        print('obsIDs done!')