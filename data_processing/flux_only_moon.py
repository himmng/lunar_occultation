from moon_flux import *
import sys

ind = int(sys.argv[1])

obsID = np.loadtxt('/astro/mwaeor/thimu/phase1/scripts/hyperdrive/obsIDs_Aug2015', dtype=np.int32)

on_ind = int(2*ind)
off_ind = int(on_ind + 1)

on_obsid = obsID[on_ind]
off_obsid = obsID[off_ind]

path = '/astro/mwaeor/thimu/phase1/scripts/hyperdrive/images_rerun2/'
path2 = 'flux_calc/'

crop_pixels = 126
im_pixels = 2048
channels = 768

Diff = np.zeros(shape = (channels,int(2*crop_pixels),int(2*crop_pixels)))
Psf = np.zeros(shape = Diff.shape)
Recon_Moon = np.zeros(shape = Diff.shape)
Recon_Rfi = np.zeros(shape = Diff.shape)
M_conv = np.zeros(shape = Diff.shape)
R_conv = np.zeros(shape= Diff.shape)
diff_rms = np.zeros(channels)

#S_M = np.zeros(channels)
#S_M_err = np.zeros(channels)
#S_M_err_Jpb = np.zeros(channels)

S_M_T = np.zeros(channels)
S_M_T_err = np.zeros(channels)
S_M_T_err_Jpb = np.zeros(channels)

#S_RFI = np.zeros(channels)
#S_RFI_err = np.zeros(channels)
#S_RFI_err_Jpb = np.zeros(channels)

S_RFI_T = np.zeros(channels)
S_RFI_T_err = np.zeros(channels)
S_RFI_T_err_Jpb = np.zeros(channels)

n_px_beam = np.zeros(channels)
pix_size_deg = np.zeros(channels)


try:

    for i in range(channels):

        if i < 10:
            imo = path+'%d-000%d-image-pb.fits'%(on_obsid, i)
            psf= path+'%d-000%d-psf-pb.fits'%(on_obsid, i)
            
        elif 10 <= i < 100:
            imo = path+'%d-00%d-image-pb.fits'%(on_obsid, i)
            psf= path+'%d-00%d-psf-pb.fits'%(on_obsid, i)
            
        else:
            imo = path+'%d-0%d-image-pb.fits'%(on_obsid, i)
            psf= path+'%d-0%d-psf-pb.fits'%(on_obsid, i)

 
        a = dirty(ON=imo, PSF=psf, im_pixels=im_pixels, crop_pixels=crop_pixels, norm_psf=True)

        if np.isnan(a[0][0][0]) == True:
            pass

        else:
            b = mask(diff=a[0], psf=a[1], n_pixels_per_beam=a[2], pix_size_deg=a[3], beam_area_deg_sq=a[4], RFI_broadn=True,\
                hl=5, vl=2, hr=3, vr=3)

            Diff[i] = a[0] ## difference image
            Psf[i] = a[1]  ## psf
            pix_size_deg[i] = a[3] ## pixel size
            n_px_beam[i] = a[2] ## npixel /beam

            M_conv[i] = b[0] ## G_image_shift
            R_conv[i] = b[1] # RFI convolved jy/pixel
            Recon_Moon[i] = b[2] ## reconstructed Moon
            Recon_Rfi[i] = b[3]  ## Reconstructed RFI

            #S_M[i] = b[5]   ## Flux density of Moon parameter
            #S_M_err[i] = b[6] ## Error in Flux density
            #S_M_err_Jpb[i] = b[7] ## Error in J/beam

            S_M_T[i] = b[8]  ## Flux density summed over Moon area
            S_M_T_err[i] = b[9] ## Error Flux density summed over Moon area

            #S_RFI[i] = b[10] ## Flux density of RFI mask
            #S_RFI_err[i] = b[11] ## Flux density Error of RFI mask
            #S_RFI_err_Jpb[i] = b[12] ## RFI mask error in Jy/beam

            S_RFI_T[i] = b[13] ## Total RFI flux density in Jy
            S_RFI_T_err[i] = b[14] ## Error in RFI flux density

            diff_rms[i] = b[15] ## RMS error, this might get high as the pixels used in the calculation include MWA

        print('done %d channels'%i)


    np.save(path2+'%d_diff_crop_image_o'%on_obsid, Diff)    
    np.save(path2+'%d_psf_crop_image_o'%on_obsid, Psf)
    np.save(path2+'%d_M_conv_crop_image_o'%on_obsid, M_conv)    
    np.save(path2+'%d_R_conv_crop_image_o'%on_obsid, R_conv)
    np.savetxt(path2+'%d_pix_deg_o'%on_obsid, pix_size_deg)
    np.savetxt(path2+'%d_n_pix_beam_o'%on_obsid, n_px_beam)

    #np.savetxt(path2+'%d_S_M_'%on_obsid, S_M)
    #np.savetxt(path2+'%d_S_M_err_'%on_obsid, S_M_err)
    #np.savetxt(path2+'%d_S_M_err_Jpb_'%on_obsid, S_M_err_Jpb)

    np.savetxt(path2+'%d_S_M_T_o'%on_obsid, S_M_T)
    np.savetxt(path2+'%d_S_M_T_err_o'%on_obsid, S_M_T_err)


    #np.savetxt(path2+'%d_S_RFI_'%on_obsid, S_RFI)
    #np.savetxt(path2+'%d_S_RFI_err_'%on_obsid, S_RFI_err)
    #np.savetxt(path2+'%d_S_RFI_err_Jpb_'%on_obsid, S_RFI_err_Jpb)

    np.savetxt(path2+'%d_S_RFI_T_o'%on_obsid, S_RFI_T)
    np.savetxt(path2+'%d_S_RFI_T_err_o'%on_obsid, S_RFI_T_err)

    np.save(path2+'%d_Recon_Moon_crop_image_o'%on_obsid, Recon_Moon)
    np.save(path2+'%d_Recon_Rfi_crop_image_o'%on_obsid, Recon_Rfi)

    np.save(path2+'%d_diff_RMS_o'%on_obsid, diff_rms)


except:
    print('error at', on_obsid, off_obsid)

