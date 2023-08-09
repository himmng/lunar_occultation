try:
    import os
    import lal
    import json
    import mpdaf
    import ephem
    import numpy as np
    import mwa_hyperbeam
    import matplotlib.pyplot as plt
    import astropy.coordinates as ac
    
    from scipy import stats
    from lal import gpstime
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.time import Time
    from erfa import hd2ae, ae2hd
    from astropy import units as u
    from astropy import time as ttm
    from scipy.stats import ks_2samp
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    from datetime import datetime, timedelta
    from skyfield.api import Topos, load, utc
    from astropy.coordinates import SkyCoord,\
          EarthLocation, AltAz, get_moon, solar_system_ephemeris
  

except ImportError:
    raise ImportError

lal.gpstime.GPS_EPOCH = lal.EPOCH_J2000_0_JD
np.seterr(divide='ignore', invalid='ignore')


def flux_density_make(nparray_path, obsIDs, only_Moon, n_channels = 768, n_coarse_channels = 5):
    '''
    This function assumes that the data is store in the following manner:
    obsIDs : observations are arranged in increasing LST order are arranged as [ON-Moon, OFF-Moon, etc.]
    S_M_T_ : flux density of the Moon [difference image] summed over Moon disk
    S_M_T_o : flux density of the Moon [only ON Moon image] summed over Moon disk
    S_RFI_T_ : flux density of the specular component
    S_RFI_T_o : flux density of the specular component [only ON Moon image]
    
    This function will return data in a concise format:
    A full-band observation of MWA comprise of 5 contigeous 30.72 MHz channels. Therefore a full-set
    will contain (5 x 30.72)MHz channels. 
    This function will make full-band observation sets by stacking 5 LST corresponding 30.72MHz channels
    together and return the dataset as follows:
    e.g. If there are N_obs no. of ON Moon observations in one day (Assuming we make equal
    observations for each 30.72MHz channel). The quantity (N_obs/5) will be the no. of full-band observations
    and if N_ch is the frequency resolution, 
    then the final dataset would be result a numpy array: (N_obs/5, N_ch x 5)
    
    params:
    path: path to the datasets, S_M_T_, S_M_T_o, S_R_T_, S_R_T_o
    obsIDs: observations
    only_Moon: bool,[True, False] will decide to which files to process
    n_channels: frequency resolution, default: 768
    n_coarse_channels: contigeous bands, default: 5
    '''
    path = nparray_path
    step = 10 ## 5 ON-Moon 5 OFF-Moon
    S_disk = np.ma.zeros(shape=(int(len(obsIDs)/step), int(n_coarse_channels * n_channels)))
    S_spec = np.ma.zeros(shape=S_disk.shape)
    S_disk_err = np.ma.zeros(shape=S_disk.shape)
    S_spec_err = np.ma.zeros(shape=S_disk.shape)
    if only_Moon == False:
        for i in range(0, len(obsIDs), step):
            
            try:
                S_M_T_ch1  = np.loadtxt(path+'%d_S_M_T_'%obsIDs[i]) ## stepping only on the ON-Moon observation
                S_R_T_ch1  = np.loadtxt(path+'%d_S_RFI_T_'%obsIDs[i])
                S_M_T_ch1_err = np.loadtxt(path+'%d_S_M_T_err_'%obsIDs[i])
                S_R_T_ch1_err = np.loadtxt(path+'%d_S_RFI_T_err_'%obsIDs[i])
                
            except:
                S_M_T_ch1 = np.ma.zeros(n_channels)
                S_R_T_ch1 = np.ma.zeros(n_channels)
                S_M_T_ch1_err = np.ma.zeros(n_channels)
                S_R_T_ch1_err = np.ma.zeros(n_channels)
            
            try:
                S_M_T_ch2  = np.loadtxt(path+'%d_S_M_T_'%obsIDs[i+2]) 
                S_R_T_ch2  = np.loadtxt(path+'%d_S_RFI_T_'%obsIDs[i+2])
                S_M_T_ch2_err = np.loadtxt(path+'%d_S_M_T_err_'%obsIDs[i+2])
                S_R_T_ch2_err = np.loadtxt(path+'%d_S_RFI_T_err_'%obsIDs[i+2])
                
            except:
                S_M_T_ch2 = np.ma.zeros(n_channels)
                S_R_T_ch2 = np.ma.zeros(n_channels)
                S_M_T_ch2_err = np.ma.zeros(n_channels)
                S_R_T_ch2_err = np.ma.zeros(n_channels)
                
            try:
                S_M_T_ch3  = np.loadtxt(path+'%d_S_M_T_'%obsIDs[i+4])  
                S_R_T_ch3  = np.loadtxt(path+'%d_S_RFI_T_'%obsIDs[i+4])
                S_M_T_ch3_err = np.loadtxt(path+'%d_S_M_T_err_'%obsIDs[i+4])
                S_R_T_ch3_err = np.loadtxt(path+'%d_S_RFI_T_err_'%obsIDs[i+4])
            
            except:
                S_M_T_ch3 = np.ma.zeros(n_channels)
                S_R_T_ch3 = np.ma.zeros(n_channels)
                S_M_T_ch3_err = np.ma.zeros(n_channels)
                S_R_T_ch3_err = np.ma.zeros(n_channels)
                
            try:
                S_M_T_ch4  = np.loadtxt(path+'%d_S_M_T_'%obsIDs[i+6])
                S_R_T_ch4  = np.loadtxt(path+'%d_S_RFI_T_'%obsIDs[i+6])
                S_M_T_ch4_err = np.loadtxt(path+'%d_S_M_T_err_'%obsIDs[i+6])
                S_R_T_ch4_err = np.loadtxt(path+'%d_S_RFI_T_err_'%obsIDs[i+6])
    
            except:
                S_M_T_ch4 = np.ma.zeros(n_channels)
                S_R_T_ch4 = np.ma.zeros(n_channels)
                S_M_T_ch4_err = np.ma.zeros(n_channels)
                S_R_T_ch4_err = np.ma.zeros(n_channels)
            
            try:
                S_M_T_ch5  = np.loadtxt(path+'%d_S_M_T_'%obsIDs[i+8])
                S_R_T_ch5  = np.loadtxt(path+'%d_S_RFI_T_'%obsIDs[i+8])
                S_M_T_ch5_err = np.loadtxt(path+'%d_S_M_T_err_'%obsIDs[i+8])
                S_R_T_ch5_err = np.loadtxt(path+'%d_S_RFI_T_err_'%obsIDs[i+8])
                
            except:
                S_M_T_ch5 = np.ma.zeros(n_channels)
                S_R_T_ch5 = np.ma.zeros(n_channels)
                S_M_T_ch5_err = np.ma.zeros(n_channels)
                S_R_T_ch5_err = np.ma.zeros(n_channels)

            S_M_T = np.concatenate([S_M_T_ch1, S_M_T_ch2, S_M_T_ch3, S_M_T_ch4, S_M_T_ch5])
            S_R_T = np.concatenate([S_R_T_ch1, S_R_T_ch2, S_R_T_ch3, S_R_T_ch4, S_R_T_ch5])
            S_M_T_err = np.concatenate([S_M_T_ch1_err, S_M_T_ch2_err, S_M_T_ch3_err,\
                                        S_M_T_ch4_err, S_M_T_ch5_err])
            S_R_T_err = np.concatenate([S_R_T_ch1_err, S_R_T_ch2_err, S_R_T_ch3_err,\
                                        S_R_T_ch4_err, S_R_T_ch5_err])


            
            S_disk[int(i/step)] = S_M_T
            S_spec[int(i/step)] = S_R_T
            S_disk_err[int(i/step)] = S_M_T_err
            S_spec_err[int(i/step)] = S_R_T_err

    elif only_Moon == True:
        
        for i in range(0, len(obsIDs), step):
            try:
                S_M_T_ch1  = np.loadtxt(path+'%d_S_M_T_o'%obsIDs[i]) ## stepping only on the ON-Moon observation
                S_R_T_ch1  = np.loadtxt(path+'%d_S_RFI_T_o'%obsIDs[i])
                S_M_T_ch1_err = np.loadtxt(path+'%d_S_M_T_err_o'%obsIDs[i])
                S_R_T_ch1_err = np.loadtxt(path+'%d_S_RFI_T_err_o'%obsIDs[i])
                
            except:
                S_M_T_ch1 = np.ma.zeros(n_channels)
                S_R_T_ch1 = np.ma.zeros(n_channels)
                S_M_T_ch1_err = np.ma.zeros(n_channels)
                S_R_T_ch1_err = np.ma.zeros(n_channels)
            
            try:
                S_M_T_ch2  = np.loadtxt(path+'%d_S_M_T_o'%obsIDs[i+2]) 
                S_R_T_ch2  = np.loadtxt(path+'%d_S_RFI_T_o'%obsIDs[i+2]) 
                S_M_T_ch2_err = np.loadtxt(path+'%d_S_M_T_err_o'%obsIDs[i+2])
                S_R_T_ch2_err = np.loadtxt(path+'%d_S_RFI_T_err_o'%obsIDs[i+2])
                
            except:
                S_M_T_ch2 = np.ma.zeros(n_channels)
                S_R_T_ch2 = np.ma.zeros(n_channels)
                S_M_T_ch2_err = np.ma.zeros(n_channels)
                S_R_T_ch2_err = np.ma.zeros(n_channels)
                
            try:
                S_M_T_ch3  = np.loadtxt(path+'%d_S_M_T_o'%obsIDs[i+4])  
                S_R_T_ch3  = np.loadtxt(path+'%d_S_RFI_T_o'%obsIDs[i+4])
                S_M_T_ch3_err = np.loadtxt(path+'%d_S_M_T_err_o'%obsIDs[i+4])
                S_R_T_ch3_err = np.loadtxt(path+'%d_S_RFI_T_err_o'%obsIDs[i+4])
            
            except:
                S_M_T_ch3 = np.ma.zeros(n_channels)
                S_R_T_ch3 = np.ma.zeros(n_channels)
                S_M_T_ch3_err = np.ma.zeros(n_channels)
                S_R_T_ch3_err = np.ma.zeros(n_channels)
                
            try:
                S_M_T_ch4  = np.loadtxt(path+'%d_S_M_T_o'%obsIDs[i+6])
                S_R_T_ch4  = np.loadtxt(path+'%d_S_RFI_T_o'%obsIDs[i+6])
                S_M_T_ch4_err = np.loadtxt(path+'%d_S_M_T_err_o'%obsIDs[i+6])
                S_R_T_ch4_err = np.loadtxt(path+'%d_S_RFI_T_err_o'%obsIDs[i+6])
    
            except:
                S_M_T_ch4 = np.ma.zeros(n_channels)
                S_R_T_ch4 = np.ma.zeros(n_channels)
                S_M_T_ch4_err = np.ma.zeros(n_channels)
                S_R_T_ch4_err = np.ma.zeros(n_channels)
            
            try:
                S_M_T_ch5  = np.loadtxt(path+'%d_S_M_T_o'%obsIDs[i+8])
                S_R_T_ch5  = np.loadtxt(path+'%d_S_RFI_T_o'%obsIDs[i+8])
                S_M_T_ch5_err = np.loadtxt(path+'%d_S_M_T_err_o'%obsIDs[i+8])
                S_R_T_ch5_err = np.loadtxt(path+'%d_S_RFI_T_err_o'%obsIDs[i+8])
                
            except:
                S_M_T_ch5 = np.ma.zeros(n_channels)
                S_R_T_ch5 = np.ma.zeros(n_channels)
                S_M_T_ch5_err = np.ma.zeros(n_channels)
                S_R_T_ch5_err = np.ma.zeros(n_channels)
                
            ## concatenating 5 observation to full band observation
            S_M_T = np.concatenate([S_M_T_ch1, S_M_T_ch2, S_M_T_ch3, S_M_T_ch4, S_M_T_ch5])
            S_R_T = np.concatenate([S_R_T_ch1, S_R_T_ch2, S_R_T_ch3, S_R_T_ch4, S_R_T_ch5])
            S_M_T_err = np.concatenate([S_M_T_ch1_err, S_M_T_ch2_err, S_M_T_ch3_err,\
                                        S_M_T_ch4_err, S_M_T_ch5_err])
            S_R_T_err = np.concatenate([S_R_T_ch1_err, S_R_T_ch2_err, S_R_T_ch3_err,\
                                        S_R_T_ch4_err, S_R_T_ch5_err])


            S_disk[int(i/step)] = S_M_T
            S_spec[int(i/step)] = S_R_T
            S_disk_err[int(i/step)] = S_M_T_err
            S_spec_err[int(i/step)] = S_R_T_err
            
    ## deleting/masking zero values:
    zero_index = np.argwhere(S_disk==0.0)
    for i in range(len(zero_index)):
        k = zero_index[i]
        S_disk[k[0]][k[1]] = np.nan
        S_spec[k[0]][k[1]] = np.nan
        S_disk_err[k[0]][k[1]] = np.nan
        S_spec_err[k[0]][k[1]] = np.nan

    
    S_disk = np.ma.masked_invalid(S_disk)
    S_spec = np.ma.masked_invalid(S_spec)
    S_disk_err = np.ma.masked_invalid(S_disk_err)
    S_spec_err = np.ma.masked_invalid(S_spec_err)

                      
    return S_disk, S_spec, S_disk_err, S_spec_err


def get_flux_for_analysis(obsIDs_Sept, obsIDs_Aug, obsIDs_Nov, obsIDs_Dec, nparray_path, month, only_Moon,\
                          lim_beam=0.5, fx_lim=1e3, local_load=True): 

    '''
    Get the flux densities specifically designed for the EoR Moon paper 2
    params:
    month: month of the observation
    only_Moon: bool
    lim_beam: beam response limit, type: float(0:Min-1:Max)
    fx_lim: flux density limit, unit: Jy
    local_load: load data from local

    return:
    S_disk: flux-density of the Moon's disk
    S_spec: flux-density of the Moon's specular 
    inplt_full_beam: interpolated beam response
    S_disk_err: uncertainity in Moon's disk flux-density (from individual observation)
    S_spec_err: uncertainity in the Specular flux-density (from individual observation)
    '''  
    if local_load == False:

        S_disk_Sept, S_spec_Sept, S_disk_err_Sept, S_spec_err_Sept = flux_density_make(nparray_path=nparray_path, \
                                                     obsIDs=obsIDs_Sept, only_Moon=only_Moon,\
                                       n_channels = nfine, n_coarse_channels = ncoarse)

        S_disk_Aug, S_spec_Aug, S_disk_err_Aug, S_spec_err_Aug = flux_density_make(nparray_path=nparray_path, \
                                                       obsIDs=obsIDs_Aug, only_Moon=only_Moon,\
                                           n_channels = nfine, n_coarse_channels = ncoarse)

        S_disk_Nov, S_spec_Nov, S_disk_err_Nov, S_spec_err_Nov = flux_density_make(nparray_path=nparray_path, \
                                                       obsIDs=obsIDs_Nov, only_Moon=only_Moon,\
                                           n_channels = nfine, n_coarse_channels = ncoarse)

        S_disk_Dec, S_spec_Dec, S_disk_err_Dec, S_spec_err_Dec = flux_density_make(nparray_path=nparray_path, \
                                                       obsIDs=obsIDs_Dec, only_Moon=only_Moon,\
                                           n_channels = nfine, n_coarse_channels = ncoarse)

    elif local_load==True:
        
        S_disk_Sept, S_spec_Sept, S_disk_err_Sept, S_spec_err_Sept = np.loadtxt('S_disk_Sept_diff'),\
                                                                    np.loadtxt('S_spec_Sept_diff'),\
        np.loadtxt('S_disk_Sept_err_diff'), np.loadtxt('S_spec_Sept_err_diff')
        
        S_disk_Aug, S_spec_Aug, S_disk_err_Aug, S_spec_err_Aug = np.loadtxt('S_disk_Aug_diff'),\
                                                                    np.loadtxt('S_spec_Aug_diff'),\
        np.loadtxt('S_disk_Aug_err_diff'), np.loadtxt('S_spec_Aug_err_diff')
        
        S_disk_Nov, S_spec_Nov, S_disk_err_Nov, S_spec_err_Nov = np.loadtxt('S_disk_Nov_diff'),\
                                                                    np.loadtxt('S_spec_Nov_diff'),\
        np.loadtxt('S_disk_Nov_err_diff'), np.loadtxt('S_spec_Nov_err_diff')
        
        S_disk_Dec, S_spec_Dec, S_disk_err_Dec, S_spec_err_Dec = np.loadtxt('S_disk_Dec_diff'),\
                                                                    np.loadtxt('S_spec_Dec_diff'),\
        np.loadtxt('S_disk_Dec_err_diff'), np.loadtxt('S_spec_Dec_err_diff')

        
    Aug_RMS = np.loadtxt('Aug_RMS_full')
    Sept_RMS = np.loadtxt('Sept_RMS_full')
    Nov_RMS = np.loadtxt('Nov_RMS_full')
    Dec_RMS = np.loadtxt('Dec_RMS_full')

    inplt_full_beam_Aug = np.loadtxt('inplt_full_beam_Aug')
    inplt_full_beam_Sept = np.loadtxt('inplt_full_beam_Sept')
    inplt_full_beam_Nov = np.loadtxt('inplt_full_beam_Nov')
    inplt_full_beam_Dec = np.loadtxt('inplt_full_beam_Dec')
    if local_load==False:
        mask_Aug = []
        for i in range(len(S_disk_Aug)):
            m1 = np.ma.where(S_disk_Aug[i]>fx_lim)[0]
            m2 = np.ma.where(S_disk_Aug[i]<-fx_lim)[0]
            
            m21 = np.ma.where(S_spec_Aug[i]>fx_lim)[0]
            m22 = np.ma.where(S_spec_Aug[i]<-fx_lim)[0]
            
            
            m3 = np.array(list(set(m1) | set(m2) | set(m21) | set(m22)))
            if len(m3) != 0:
                S_disk_Aug[i][m3] = np.nan
                S_disk_Aug[i] = np.ma.masked_invalid(S_disk_Aug[i])

                S_spec_Aug[i][m3] = np.nan
                S_spec_Aug[i] = np.ma.masked_invalid(S_spec_Aug[i])

                S_disk_err_Aug[i][m3] = np.nan
                S_disk_err_Aug[i] = np.ma.masked_invalid(S_disk_err_Aug[i])

                S_spec_err_Aug[i][m3] = np.nan
                S_spec_err_Aug[i] = np.ma.masked_invalid(S_spec_err_Aug[i])

            else:
                S_disk_Aug[i] = np.ma.masked_invalid(S_disk_Aug[i])
                S_spec_Aug[i] = np.ma.masked_invalid(S_spec_Aug[i])
                S_disk_err_Aug[i] = np.ma.masked_invalid(S_disk_err_Aug[i])
                S_spec_err_Aug[i] = np.ma.masked_invalid(S_spec_err_Aug[i])
            mask = S_disk_Aug[i].mask

            mask_Aug.append(mask)



        mask_Aug = np.array(mask_Aug)


        mask_Sept = []

        for i in range(len(S_disk_Sept)):
            m1 = np.ma.where(S_disk_Sept[i]>fx_lim)[0]
            m2 = np.ma.where(S_disk_Sept[i]<-fx_lim)[0]
            
            m21 = np.ma.where(S_spec_Sept[i]>fx_lim)[0]
            m22 = np.ma.where(S_spec_Sept[i]<-fx_lim)[0]
            
            m3 = np.array(list(set(m1) | set(m2) | set(m21) | set(m22)))

            if len(m3) != 0:
                S_disk_Sept[i][m3] = np.nan
                S_disk_Sept[i] = np.ma.masked_invalid(S_disk_Sept[i])

                S_spec_Sept[i][m3] = np.nan
                S_spec_Sept[i] = np.ma.masked_invalid(S_spec_Sept[i])

                S_disk_err_Sept[i][m3] = np.nan
                S_disk_err_Sept[i] = np.ma.masked_invalid(S_disk_err_Sept[i])

                S_spec_err_Sept[i][m3] = np.nan
                S_spec_err_Sept[i] = np.ma.masked_invalid(S_spec_err_Sept[i])
            else:
                S_disk_Sept[i] = np.ma.masked_invalid(S_disk_Sept[i])
                S_spec_Sept[i] = np.ma.masked_invalid(S_spec_Sept[i])
                S_disk_err_Sept[i] = np.ma.masked_invalid(S_disk_err_Sept[i])
                S_spec_err_Sept[i] = np.ma.masked_invalid(S_spec_err_Sept[i])

            mask = S_disk_Sept[i].mask

            mask_Sept.append(mask)

        mask_Sept = np.array(mask_Sept)


        mask_Nov = []

        for i in range(len(S_disk_Nov)):
            m1 = np.ma.where(S_disk_Nov[i]>fx_lim)[0]
            m2 = np.ma.where(S_disk_Nov[i]<-fx_lim)[0]
            
            m21 = np.ma.where(S_spec_Nov[i]>fx_lim)[0]
            m22 = np.ma.where(S_spec_Nov[i]<-fx_lim)[0]
            
            m3 = np.array(list(set(m1) | set(m2) | set(m21) | set(m22)))

            if len(m3) != 0:
                S_disk_Nov[i][m3] = np.nan
                S_disk_Nov[i] = np.ma.masked_invalid(S_disk_Nov[i])

                S_spec_Nov[i][m3] = np.nan
                S_spec_Nov[i] = np.ma.masked_invalid(S_spec_Nov[i])

                S_disk_err_Nov[i][m3] = np.nan
                S_disk_err_Nov[i] = np.ma.masked_invalid(S_disk_err_Nov[i])

                S_spec_err_Nov[i][m3] = np.nan
                S_spec_err_Nov[i] = np.ma.masked_invalid(S_spec_err_Nov[i])
            else:
                S_disk_Nov[i] = np.ma.masked_invalid(S_disk_Nov[i])
                S_spec_Nov[i] = np.ma.masked_invalid(S_spec_Nov[i])
                S_disk_err_Nov[i] = np.ma.masked_invalid(S_disk_err_Nov[i])
                S_spec_err_Nov[i] = np.ma.masked_invalid(S_spec_err_Nov[i])

            mask = S_disk_Nov[i].mask

            mask_Nov.append(mask)

        mask_Nov = np.array(mask_Nov)

        mask_Dec = []

        for i in range(len(S_disk_Dec)):
            m1 = np.ma.where(S_disk_Dec[i]>fx_lim)[0]
            m2 = np.ma.where(S_disk_Dec[i]<-fx_lim)[0]
            
            m21 = np.ma.where(S_spec_Dec[i]>fx_lim)[0]
            m22 = np.ma.where(S_spec_Dec[i]<-fx_lim)[0]
            
            m3 = np.array(list(set(m1) | set(m2) | set(m21) | set(m22)))

            if len(m3) != 0:
                S_disk_Dec[i][m3] = np.nan
                S_disk_Dec[i] = np.ma.masked_invalid(S_disk_Dec[i])

                S_spec_Dec[i][m3] = np.nan
                S_spec_Dec[i] = np.ma.masked_invalid(S_spec_Dec[i])

                S_disk_err_Dec[i][m3] = np.nan
                S_disk_err_Dec[i] = np.ma.masked_invalid(S_disk_err_Dec[i])

                S_spec_err_Dec[i][m3] = np.nan
                S_spec_err_Dec[i] = np.ma.masked_invalid(S_spec_err_Dec[i])
            else:
                S_disk_Dec[i] = np.ma.masked_invalid(S_disk_Dec[i])
                S_spec_Dec[i] = np.ma.masked_invalid(S_spec_Dec[i])
                S_disk_err_Dec[i] = np.ma.masked_invalid(S_disk_err_Dec[i])
                S_spec_err_Dec[i] = np.ma.masked_invalid(S_spec_err_Dec[i])

            mask = S_disk_Dec[i].mask
            mask_Dec.append(mask)

        mask_Dec = np.array(mask_Dec)
        freq_Aug = []
        for i in range(len(S_disk_Aug)):
            freq_Aug.append(freq)
        freq_Aug = np.array(freq_Aug)

        freq_Sept = []
        for i in range(len(S_disk_Sept)):
            freq_Sept.append(freq)
        freq_Sept = np.array(freq_Sept)

        freq_Nov = []
        for i in range(len(S_disk_Nov)):
            freq_Nov.append(freq)
        freq_Nov = np.array(freq_Nov)

        freq_Dec = []
        for i in range(len(S_disk_Dec)):
            freq_Dec.append(freq)
        freq_Dec = np.array(freq_Dec)

        freq_Aug = np.ma.masked_array(freq_Aug, mask_Aug)
        freq_Sept = np.ma.masked_array(freq_Sept, mask_Sept)
        freq_Nov = np.ma.masked_array(freq_Nov, mask_Nov)
        freq_Dec = np.ma.masked_array(freq_Dec, mask_Dec)

        inplt_full_beam_Aug = np.ma.masked_array(inplt_full_beam_Aug, mask_Aug)
        inplt_full_beam_Sept = np.ma.masked_array(inplt_full_beam_Sept, mask_Sept)
        inplt_full_beam_Nov = np.ma.masked_array(inplt_full_beam_Nov, mask_Nov)
        inplt_full_beam_Dec = np.ma.masked_array(inplt_full_beam_Dec, mask_Dec)

        Aug_RMS = np.ma.masked_array(Aug_RMS, mask_Aug)
        Sept_RMS = np.ma.masked_array(Sept_RMS, mask_Sept)
        Nov_RMS = np.ma.masked_array(Nov_RMS, mask_Nov)
        Dec_RMS = np.ma.masked_array(Dec_RMS, mask_Dec)

        mask_Dec = np.delete(mask_Dec, 8, axis=0)
        Dec_RMS = np.delete(Dec_RMS, 8, axis=0)
        S_disk_Dec = np.delete(S_disk_Dec, 8, axis=0)
        S_spec_Dec = np.delete(S_spec_Dec, 8, axis=0)
        S_disk_err_Dec = np.delete(S_disk_err_Dec, 8, axis=0)
        S_spec_err_Dec = np.delete(S_spec_err_Dec, 8, axis=0)
        freq_Dec = np.delete(freq_Dec, 8, axis=0)
        inplt_full_beam_Dec = np.delete(inplt_full_beam_Dec, 8, axis=0)
        
    elif local_load==True:
        
        Dec_RMS = np.delete(Dec_RMS, 8, axis=0)
        S_disk_Dec = np.delete(S_disk_Dec, 8, axis=0)
        S_spec_Dec = np.delete(S_spec_Dec, 8, axis=0)
        S_disk_err_Dec = np.delete(S_disk_err_Dec, 8, axis=0)
        S_spec_err_Dec = np.delete(S_spec_err_Dec, 8, axis=0)
        inplt_full_beam_Dec = np.delete(inplt_full_beam_Dec, 8, axis=0)
        
        
    les_04_Dec = np.argwhere(inplt_full_beam_Dec < lim_beam)
    les_04_Nov = np.argwhere(inplt_full_beam_Nov < lim_beam)
    les_04_Sept = np.argwhere(inplt_full_beam_Sept < lim_beam)
    les_04_Aug = np.argwhere(inplt_full_beam_Aug < lim_beam)

    for i in range(len(les_04_Dec)):
        S_disk_Dec[les_04_Dec[i][0], les_04_Dec[i][1]] = np.nan
        S_spec_Dec[les_04_Dec[i][0], les_04_Dec[i][1]] = np.nan
        S_disk_err_Dec[les_04_Dec[i][0], les_04_Dec[i][1]] = np.nan
        S_spec_err_Dec[les_04_Dec[i][0], les_04_Dec[i][1]] = np.nan

    for i in range(len(les_04_Sept)):
        S_disk_Sept[les_04_Sept[i][0], les_04_Sept[i][1]] = np.nan
        S_spec_Sept[les_04_Sept[i][0], les_04_Sept[i][1]] = np.nan
        S_disk_err_Sept[les_04_Sept[i][0], les_04_Sept[i][1]] = np.nan
        S_spec_err_Sept[les_04_Sept[i][0], les_04_Sept[i][1]] = np.nan

    for i in range(len(les_04_Aug)):
        S_disk_Aug[les_04_Aug[i][0], les_04_Aug[i][1]] = np.nan
        S_spec_Aug[les_04_Aug[i][0], les_04_Aug[i][1]] = np.nan
        S_disk_err_Aug[les_04_Aug[i][0], les_04_Aug[i][1]] = np.nan
        S_spec_err_Aug[les_04_Aug[i][0], les_04_Aug[i][1]] = np.nan

    for i in range(len(les_04_Nov)):
        S_disk_Nov[les_04_Nov[i][0], les_04_Nov[i][1]] = np.nan
        S_spec_Nov[les_04_Nov[i][0], les_04_Nov[i][1]] = np.nan
        S_disk_err_Nov[les_04_Nov[i][0], les_04_Nov[i][1]] = np.nan
        S_spec_err_Nov[les_04_Nov[i][0], les_04_Nov[i][1]] = np.nan
        

    S_disk_Dec = np.ma.masked_invalid(S_disk_Dec)
    S_spec_Dec = np.ma.masked_invalid(S_spec_Dec)
    
    S_disk_err_Dec = np.ma.masked_invalid(S_disk_err_Dec)
    S_spec_err_Dec = np.ma.masked_invalid(S_spec_err_Dec)

    S_disk_Nov = np.ma.masked_invalid(S_disk_Nov)
    S_spec_Nov = np.ma.masked_invalid(S_spec_Nov)
    
    S_disk_err_Nov = np.ma.masked_invalid(S_disk_err_Nov)
    S_spec_err_Nov = np.ma.masked_invalid(S_spec_err_Nov)


    S_disk_Sept = np.ma.masked_invalid(S_disk_Sept)
    S_spec_Sept = np.ma.masked_invalid(S_spec_Sept)
    
    S_disk_err_Sept = np.ma.masked_invalid(S_disk_err_Sept)
    S_spec_err_Sept = np.ma.masked_invalid(S_spec_err_Sept)


    S_disk_Aug = np.ma.masked_invalid(S_disk_Aug)
    S_spec_Aug = np.ma.masked_invalid(S_spec_Aug)
    
    S_disk_err_Aug = np.ma.masked_invalid(S_disk_err_Aug)
    S_spec_err_Aug = np.ma.masked_invalid(S_spec_err_Aug)

    if local_load==False:
        freq_Dec = np.ma.masked_array(freq_Dec, mask_Dec)
        inplt_full_beam_Dec = np.ma.masked_array(inplt_full_beam_Dec, mask_Dec)

        freq_Sept = np.ma.masked_array(freq_Sept, mask_Sept)
        inplt_full_beam_Sept = np.ma.masked_array(inplt_full_beam_Sept, mask_Sept)

        freq_Nov = np.ma.masked_array(freq_Nov, mask_Nov)
        inplt_full_beam_Nov = np.ma.masked_array(inplt_full_beam_Nov, mask_Nov)

        freq_Aug = np.ma.masked_array(freq_Aug, mask_Aug)
        inplt_full_beam_Aug = np.ma.masked_array(inplt_full_beam_Aug, mask_Aug)
        

    if month == 'Aug':
        return S_disk_Aug, S_spec_Aug, inplt_full_beam_Aug, S_disk_err_Aug, S_spec_err_Aug
    elif month == 'Sept':
        return S_disk_Sept, S_spec_Sept, inplt_full_beam_Sept, S_disk_err_Sept, S_spec_err_Sept
    elif month == 'Nov':
        return S_disk_Nov, S_spec_Nov, inplt_full_beam_Nov, S_disk_err_Nov, S_spec_err_Nov
    elif month == 'Dec':
        return S_disk_Dec, S_spec_Dec, inplt_full_beam_Dec, S_disk_err_Dec, S_spec_err_Dec
    else:
        print('use correct month')


def wtd_flux_sigma(S_disk, S_spec, S_moon, inplt_full_beam, wtd=True):
    '''
    Get the weighted mean and sigma of the flux-density
    
    params:
    S_disk: flux-density of the Moon's disk
    S_spec: flux-density of the Moon's specular
    S_moon: flux-density of the Moon
    inplt_full_beam: interpolated beam response 
    
    returns:
    S_disk_wtd: weighted mean flux-density, Disk
    S_spec_wtd: weighted mean flux-density, Specular 
    S_moon_wtd: weighted mean flux_density Moon 
    Sigma_S_disk_wtd: weighted sigma flux-density, Disk
    Sigma_S_spec_wtd: weighted sigma flux-density, Specular 
    Sigma_S_moon_wtd: weighed sigma flux-density Moon
    '''
    if wtd == True:
        S_disk_wtd = np.nansum(S_disk*((inplt_full_beam)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0)

        S_spec_wtd = np.nansum(S_spec*((inplt_full_beam)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0)

        S_moon_wtd = np.nansum(S_moon*((inplt_full_beam)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0)

        Sigma_S_disk_wtd = np.sqrt(np.nansum((((inplt_full_beam)**2)*(S_disk - S_disk_wtd)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0))

        Sigma_S_spec_wtd = np.sqrt(np.nansum((((inplt_full_beam)**2)*(S_spec - S_spec_wtd)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0))

        Sigma_S_moon_wtd = np.sqrt(np.nansum((((inplt_full_beam)**2)*(S_moon - S_moon_wtd)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0))
    elif wtd == False:
        S_disk_wtd = np.nanmean(S_disk, axis=0)

        S_spec_wtd = np.nanmean(S_spec, axis=0)

        S_moon_wtd = np.nanmean(S_moon, axis=0)

        Sigma_S_disk_wtd = np.nanstd(S_disk, axis=0)

        Sigma_S_spec_wtd = np.nanstd(S_spec, axis=0)

        Sigma_S_moon_wtd = np.nanstd(S_moon, axis=0)
        
    return S_disk_wtd, S_spec_wtd, S_moon_wtd, Sigma_S_disk_wtd, Sigma_S_spec_wtd, Sigma_S_moon_wtd
    
def pol_fit( x, m, b,):
    '''
    line fit to the curve
    '''
    return ((x*m) - b)


def fitting_func_for_fit(S_disk, freq, S_disk_err, limit_freq, meanover):
    '''
    function to create numpy arrays for fitting the function, 
    we remove NaNs and other masked indices from the arrays here.
    
    params:
    S_disk: flux-density of the Moon's disk
    freq: frequency array
    S_disk_err:  uncertainity flux-density, Disk
    limit_freq: limit frequency upto which the data is cropped.
    meanover: bool, wheather doing over the mean value, not individual observations (sigma is not used).


    returns:
    popt: optimal parameter
    pcov: covariance 
    fit_Y: fitted curve/line
    fit_freq: fitting frequeny 
    fit_S_disk_flux: fitting S_disk flux-density  
    fit_sigma_S_disk: fitting sigma_S_disk 
    fit_Y_errs: fitting function errors

    '''
    
    fit_S_disk_flux=[]
    fit_freq=[]
    fit_S_disk_err=[]
    
    S_disk_copy = np.copy(S_disk)
    S_disk_err_copy = np.copy(S_disk_err)
    
    S_disk_copy[FM_index] = np.nan
    S_disk_err_copy[FM_index] = np.nan

    zero_indx = np.ma.where(S_disk_copy==0.)[0]
    S_disk_copy[zero_indx] = np.nan
    S_disk_err_copy[zero_indx] = np.nan

    freq_grt = np.where(freq>limit_freq)[0]
    S_disk_copy[freq_grt] = np.nan
    S_disk_err_copy[freq_grt] = np.nan
    
    S_disk_copy = np.ma.masked_invalid(S_disk_copy)
    S_disk_err_copy = np.ma.masked_invalid(S_disk_err_copy)

    nan_mask_index = np.where(np.isnan(S_disk_copy.data)==True)[0]
    
    fit_freq.append(np.delete(freq, nan_mask_index))
    fit_freq = np.array(fit_freq)
    fit_freq = fit_freq.flatten('F')
    
    fit_S_disk_flux.append(np.delete(S_disk_copy, nan_mask_index))
    fit_S_disk_flux = np.array(fit_S_disk_flux)
    fit_S_disk_flux = fit_S_disk_flux.flatten('F')
    
    fit_S_disk_err.append(np.delete(S_disk_err_copy, nan_mask_index))
    fit_S_disk_err = np.array(fit_S_disk_err)
    fit_S_disk_err = fit_S_disk_err.flatten('F')
    
    if len(fit_S_disk_flux)==0:
        popt = np.nan*np.zeros(2)
        pcov = np.nan*np.zeros(shape=(2,2))
        fit_Y = np.nan*np.zeros(len(freq))
        fit_S_disk_flux = fit_Y
        fit_S_disk_err = fit_Y
        fit_Y_errs = popt
        
    else:
        try:
            if meanover==False:
                popt, pcov = curve_fit(f=pol_fit, xdata=fit_freq, ydata=fit_S_disk_flux, sigma=fit_S_disk_err,\
                                      absolute_sigma=True)
                fit_Y = pol_fit(freq, *popt)
                fit_Y_errs = np.sqrt(np.diag(pcov))
            elif meanover==True:
                popt, pcov = curve_fit(f=pol_fit, xdata=fit_freq, ydata=fit_S_disk_flux,)
                fit_Y = pol_fit(freq, *popt)
                fit_Y_errs = np.sqrt(np.diag(pcov))
        except RuntimeError:
            print('not a fit')
            popt = np.nan*np.zeros(2)
            pcov = np.nan*np.zeros(shape=(2,2))
            fit_Y = np.nan*np.zeros(len(freq))
            fit_S_disk_flux = fit_Y
            fit_S_disk_err = fit_Y
            fit_Y_errs = popt
    return popt, pcov, fit_Y, fit_freq, fit_S_disk_flux, fit_S_disk_err, fit_Y_errs, nan_mask_index

def S_moon(S_disk, S_spec, S_disk_err, S_spec_err, fit_Y, fit_Y_errs):
    '''
    Get the flux-density of the Moon from the disk and specular flux-density
    
    params:
    S_disk: flux-density of the Moon's disk
    S_spec: flux-density of the Moon's specular
    S_disk_err: uncertainity in Moon's disk flux-density (from individual observation)
    S_spec_err: uncertainity in the Specular flux-density (from individual observation)
    fit_Y: fitting function 
    fit_Y_errs: fitting function errors

    returns:
    S_moon: flux-density of the Moon (these will be individual observations)
    S_moon_err: uncertainity in the Moon's flux-density (individual observations)
    '''
    beta = -0.58
    ch = np.where(freq==100.02704871060172)[0]

    S_diffuse_at_100 = np.subtract(np.abs(S_disk[ch]), fit_Y[ch])
 
    S_spec_at_100 = S_spec[ch]
    
    A = (S_diffuse_at_100/np.abs(S_spec_at_100))
    Re = A*(freq/(freq[ch])) **(-beta)
    plt.plot(Re, 'b')
    S_moon = np.ma.subtract(S_disk, Re*S_spec)

    dm, dc = fit_Y_errs
    
    dY_fit = np.sqrt((100*dm)**2 + dc**2)

    S_disk_err_at_100 = S_disk_err[ch]
    S_spec_err_at_100 = S_spec_err[ch]
    
    S_diffuse_err_at_100 = np.sqrt(S_disk_err_at_100**2 + dY_fit**2 )

    Re_err = (100**beta)*np.sqrt(((S_diffuse_at_100/S_spec_at_100)**2)*((S_diffuse_err_at_100/\
                                                                           S_diffuse_at_100)**2 +\
                                                (S_spec_err_at_100/S_spec_at_100)**2 ))
    
    
    S_moon_err = np.sqrt(S_disk_err**2 + (Re_err*S_spec_err)**2)

    return S_moon, S_moon_err, Re*S_spec, Re_err*S_spec_err

def S_moon_FM(S_disk, S_spec, S_disk_err, S_spec_err, S_FM, freq_FM, fit_Y, fit_Y_errs):
    '''
    Get the flux-density of the Moon from the disk and specular flux-density
    
    params:
    S_disk: flux-density of the Moon's disk
    S_spec: flux-density of the Moon's specular
    S_disk_err: uncertainity in Moon's disk flux-density (from individual observation)
    S_spec_err: uncertainity in the Specular flux-density (from individual observation)
    S_FM: simulated FM flux-density of the Moon's disk
    freq_FM: operating frequencies of the FM station
    fit_Y: fitting function 
    fit_Y_errs: fitting function errors

    returns:
    S_moon: flux-density of the Moon (these will be individual observations)
    S_moon_err: uncertainity in the Moon's flux-density (individual observations)
    '''
    beta = -0.58
    ch = np.where(freq==100.02704871060172)[0]
    ch2 = np.where(freq_FM==99.86300599114352)[0]
    #ch = np.where(freq_FM==99.86300599114352)[0]

    #S_FM_mean100 = np.nanmean(S_FM[c], axis=0)[ch2]

    S_diffuse_at_100_ = np.subtract(np.abs(S_disk[ch]), fit_Y[ch])
    S_diffuse_at_100 = np.abs(S_FM[ch2])
    S_spec_at_100 = S_spec[ch]
    S_disk_err_at_100 = S_disk_err[ch]
    S_spec_err_at_100 = S_spec_err[ch]
    A = (S_diffuse_at_100/np.abs(S_spec_at_100))
    dm, dc = fit_Y_errs
    
    dY_fit = np.sqrt((100*dm)**2 + dc**2)
    S_diffuse_err_at_100 = np.sqrt(S_disk_err_at_100**2 + dY_fit**2 )
    
    deltaA = np.sqrt((S_disk_err_at_100/S_disk[ch])**2 +\
                     (S_spec_err_at_100/S_spec_at_100)**2) * (S_diffuse_at_100_/S_spec_at_100)
    Re = A*(freq/(freq[ch])) **(-beta)
   
    plt.plot(Re, 'r')
    
    S_moon = np.ma.subtract(S_disk, Re*S_spec)

    S_diffuse_err_at_100 = np.sqrt(S_disk_err_at_100**2 + dY_fit**2 )

    Re_err = (100**beta)*np.sqrt(((S_diffuse_at_100/S_spec_at_100)**2)*((S_diffuse_err_at_100/\
                                                                           S_diffuse_at_100)**2 +\
                                                (S_spec_err_at_100/S_spec_at_100)**2 ))
    
    
    S_moon_err = np.sqrt(S_disk_err**2 + (Re_err*S_spec_err)**2)

    return S_moon, S_moon_err, Re*S_spec, Re_err*S_spec_err

def T_gal_minus_moon_func(nu, T_Gal, alpha, Toff):
    '''
    Fitting funcation of the Synchrotron Galactic Sky - Moon Temp.
    
    params:
    nu: frequency
    T_Gal: Galactic Temperature
    alpha: Synchrotron spectral index
    Toff: Offset temeperature, Moon's Temperature in some sense
    
    returns: Galactic - Moon temperature
    '''
    return T_Gal*((nu/150.)**alpha) - Toff

def T_gal_intpld_with_fittng_full(skymodel, month, freq, mtype, version, model_error):
    '''
    Used the Reflected Galactic temperature from the Moon and the Moon occulted sky temprature from the GSM models
    
    params:
    
    skymodel: name of the skymodel; existing data has: GSM, GSM2016, LFSM, Haslam
    month: month of the observation
    
    returns:
    T_ref_gal: galactic temp array
    T_ref_gal_mean: Mean reflected temp. from the Moon
    T_ref_gal_std: Sigma of the mean reflected temp.
    T_ref_gal_inplt: MWA full band interpolated Galactic Temp
    Sigma_T_ref_gal_inplt: Sigma interploted Galactic Temp

    T_moon_back: Moon occulted patch temp. array
    T_moon_back_mean: Moon occulted patch average temp.
    T_moon_back_std: Sigma of the Moon occulted patch temp.
    T_moon_back_inplt: MWA full band interpolated Moon occulted patch temp
    Sigma_T_moon_back_inplt: Sigma interploted Moon occulted patch Temp
    freq_array: frequency array correspond to the temp array
    '''
    
    if mtype=='old':
        if version=='1':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s'%(skymodel, month))
        elif version=='2':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s2'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s2'%(skymodel, month))
    elif mtype=='new':
        Temp_refl = np.loadtxt('../../sky_temp_new/Ref_T_Gal_%s_%s'%(skymodel, month))*0.07
        Temp_moon_back = np.loadtxt('../../sky_temp_new/GSM_T_Gal_%s_%s'%(skymodel, month))
    
    freq_array = np.linspace(70, 230, 32)

    T_ref_gal_inplt = []
    T_moon_back_inplt = []
    
    
    for i in range(len(Temp_refl)):
        T_ref_gal_full = interp1d(freq_array, Temp_refl[i], bounds_error=False, fill_value= 'extrapolate')
        T_moon_back_full = interp1d(freq_array, Temp_moon_back[i], bounds_error=False, fill_value= 'extrapolate')
        T_ref_gal_inplt.append(T_ref_gal_full(freq))
        T_moon_back_inplt.append(T_moon_back_full(freq))

    T_ref_gal_inplt = np.array(T_ref_gal_inplt)
    T_moon_back_inplt = np.array(T_moon_back_inplt)
    
    popt_gal = []
    pcov_gal = []
    alpha = []
    
    popt_refgal = []
    pcov_refgal = []
    beta = []
    
    for i in range(len(T_moon_back_inplt)):
        p = curve_fit(f=T_gal_alpha, xdata=freq, ydata=T_moon_back_inplt[i],\
                      sigma=T_moon_back_inplt[i]*model_error, absolute_sigma=True)
        
        p1 = curve_fit(f=T_gal_alpha, xdata=freq, ydata=T_ref_gal_inplt[i],\
                      sigma=T_ref_gal_inplt[i]*model_error, absolute_sigma=True)
        
        popt_gal.append(p[0])
        pcov_gal.append(p[1])
        
        popt_refgal.append(p1[0])
        pcov_refgal.append(p1[1])
        
    popt_gal = np.array(popt_gal)
    pcov_gal = np.array(pcov_gal)
    
    popt_refgal = np.array(popt_refgal)
    pcov_refgal = np.array(pcov_refgal)

    return popt_gal, pcov_gal, popt_refgal, pcov_refgal, T_moon_back_inplt, T_ref_gal_inplt

def T_gal_intpld_with_fittng_coarse(skymodel, month, freq, mtype, version, model_error):
    '''
    Used the Reflected Galactic temperature from the Moon and the Moon occulted sky temprature from the GSM models
    
    params:
    
    skymodel: name of the skymodel; existing data has: GSM, GSM2016, LFSM, Haslam
    month: month of the observation
    
    returns:
    T_ref_gal: galactic temp array
    T_ref_gal_mean: Mean reflected temp. from the Moon
    T_ref_gal_std: Sigma of the mean reflected temp.
    T_ref_gal_inplt: MWA full band interpolated Galactic Temp
    Sigma_T_ref_gal_inplt: Sigma interploted Galactic Temp

    T_moon_back: Moon occulted patch temp. array
    T_moon_back_mean: Moon occulted patch average temp.
    T_moon_back_std: Sigma of the Moon occulted patch temp.
    T_moon_back_inplt: MWA full band interpolated Moon occulted patch temp
    Sigma_T_moon_back_inplt: Sigma interploted Moon occulted patch Temp
    freq_array: frequency array correspond to the temp array
    '''
    
    if mtype=='old':
        if version=='1':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s'%(skymodel, month))
        elif version=='2':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s2'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s2'%(skymodel, month))
    elif mtype=='new':
        Temp_refl = np.loadtxt('../../sky_temp_new/Ref_T_Gal_%s_%s'%(skymodel, month))*0.07
        Temp_moon_back = np.loadtxt('../../sky_temp_new/GSM_T_Gal_%s_%s'%(skymodel, month))
    
    freq_array = np.linspace(70, 230, 32)
    freq = np.linspace(72, 232, 32)

    T_ref_gal_inplt = []
    T_moon_back_inplt = []
    
    
    for i in range(len(Temp_refl)):
        T_ref_gal_full = interp1d(freq_array, Temp_refl[i], bounds_error=False, fill_value= 'extrapolate')
        T_moon_back_full = interp1d(freq_array, Temp_moon_back[i], bounds_error=False, fill_value= 'extrapolate')
        T_ref_gal_inplt.append(T_ref_gal_full(freq))
        T_moon_back_inplt.append(T_moon_back_full(freq))

    T_ref_gal_inplt = np.array(T_ref_gal_inplt)
    T_moon_back_inplt = np.array(T_moon_back_inplt)
    
    popt_gal = []
    pcov_gal = []
    alpha = []
    
    popt_refgal = []
    pcov_refgal = []
    beta = []
    
    for i in range(len(T_moon_back_inplt)):
        p = curve_fit(f=T_gal_alpha, xdata=freq, ydata=T_moon_back_inplt[i],\
                      sigma=T_moon_back_inplt[i]*model_error, absolute_sigma=True)
        
        p1 = curve_fit(f=T_gal_alpha, xdata=freq, ydata=T_ref_gal_inplt[i],\
                      sigma=T_ref_gal_inplt[i]*model_error, absolute_sigma=True)
        
        popt_gal.append(p[0])
        pcov_gal.append(p[1])
        
        popt_refgal.append(p1[0])
        pcov_refgal.append(p1[1])
        
    popt_gal = np.array(popt_gal)
    pcov_gal = np.array(pcov_gal)
    
    popt_refgal = np.array(popt_refgal)
    pcov_refgal = np.array(pcov_refgal)

    return popt_gal, pcov_gal, popt_refgal, pcov_refgal, T_moon_back_inplt, T_ref_gal_inplt

def T_gal_intpld(skymodel, month, mtype, version):
    '''
    Used the Reflected Galactic temperature from the Moon and the Moon occulted sky temprature from the GSM models
    
    params:
    
    skymodel: name of the skymodel; existing data has: GSM, GSM2016, LFSM, Haslam
    month: month of the observation
    
    returns:
    T_ref_gal: galactic temp array
    T_ref_gal_mean: Mean reflected temp. from the Moon
    T_ref_gal_std: Sigma of the mean reflected temp.
    T_ref_gal_inplt: MWA full band interpolated Galactic Temp
    Sigma_T_ref_gal_inplt: Sigma interploted Galactic Temp

    T_moon_back: Moon occulted patch temp. array
    T_moon_back_mean: Moon occulted patch average temp.
    T_moon_back_std: Sigma of the Moon occulted patch temp.
    T_moon_back_inplt: MWA full band interpolated Moon occulted patch temp
    Sigma_T_moon_back_inplt: Sigma interploted Moon occulted patch Temp
    freq_array: frequency array correspond to the temp array
    '''
    
    if mtype=='old':
        if version=='1':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s'%(skymodel, month))
        elif version=='2':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s2'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s2'%(skymodel, month))
    elif mtype=='new':
        Temp_refl = np.loadtxt('../../sky_temp_new/Ref_T_Gal_%s_%s'%(skymodel, month))*0.07
        Temp_moon_back = np.loadtxt('../../sky_temp_new/GSM_T_Gal_%s_%s'%(skymodel, month))
    
    freq_array = np.linspace(70, 230, 32)
    freq = np.linspace(72, 232, 32)
    
    T_ref_gal_inplt = []
    T_moon_back_inplt = []
    
    
    for i in range(len(Temp_refl)):
        T_ref_gal_full = interp1d(freq_array, Temp_refl[i], bounds_error=False, fill_value= 'extrapolate')
        T_moon_back_full = interp1d(freq_array, Temp_moon_back[i], bounds_error=False, fill_value= 'extrapolate')
        T_ref_gal_inplt.append(T_ref_gal_full(freq))
        T_moon_back_inplt.append(T_moon_back_full(freq))

    T_ref_gal_inplt = np.array(T_ref_gal_inplt)
    T_moon_back_inplt = np.array(T_moon_back_inplt)
    
    

    return T_moon_back_inplt, T_ref_gal_inplt

def T_Gal_fitting_func(S_Moon, S_Moon_err, T_ref_gal, Sigma_T_ref_gal, freq, limit_freq, meanover): 
    '''
    Get the final Moon occulted background temp of the Moon
    params:

    S_Moon: flux-density of the Moon
    S_Moon_err: uncertainity on the flux-density of the Moon
    T_ref_gal: reflected galactic background temp. We will interplote the Galactic temp at finer freq.
    Sigma_T_ref_gal: sigma of the reflected galactic background
    limit_freq: limit frequency upto which the data is cropped.
    meanover: bool, wheather doing over the mean value, not individual observations.

    returns:
    popt: fitting optimal parameters 
    pcov: fitting covariance 
    errs: fitting errors 
    T_gal: Galactic Temperature
    T_gal_error: Uncertainity in Galactic Temperature
    '''
    

    freq_grt = np.where(freq>limit_freq)[0]
    S_Moon[freq_grt] = np.nan
    S_Moon_err[freq_grt] = np.nan
    
    nan_index = np.where(np.isnan(S_Moon.data)==True)[0]
    S_Moon_fit = np.delete(S_Moon, nan_index)
    S_Moon_err_fit = np.delete(S_Moon_err, nan_index)
    freq_fit = np.delete(freq, nan_index)
    T_ref_gal_fit  = np.delete(T_ref_gal, nan_index)
    Sigma_T_ref_gal_fit = np.delete(Sigma_T_ref_gal, nan_index)
    
    zero_index = np.where(S_Moon_err_fit == 0.)[0]
    S_Moon_fit = np.delete(S_Moon_fit, zero_index)
    S_Moon_err_fit = np.delete(S_Moon_err_fit, zero_index)
    freq_fit = np.delete(freq_fit, zero_index)
    T_ref_gal_fit  = np.delete(T_ref_gal_fit, zero_index)
    Sigma_T_ref_gal_fit = np.delete(Sigma_T_ref_gal_fit, zero_index)

    delta_t = ((1e-26*9*1e16)*(S_Moon))/((2*1.38*1e-23)*(7.365*1e-5)*freq**2)
    delta_t = delta_t/10**12
    
    delta_t_fit = ((1e-26*9*1e16)*(S_Moon_fit))/((2*1.38*1e-23)*(7.365*1e-5)*freq_fit**2)
    delta_t_fit = delta_t_fit/10**12
    
    delta_t_error = ((1e-26*9*1e16)*(S_Moon_err))/((2*1.38*1e-23)*(7.365*1e-5)*freq**2)
    delta_t_error = delta_t_error/10**12
    
    delta_t_error_fit = ((1e-26*9*1e16)*(S_Moon_err_fit))/((2*1.38*1e-23)*(7.365*1e-5)*freq_fit**2)
    delta_t_error_fit = delta_t_error_fit/10**12

    dT_CMB = 0.00057
    T_CMB = 2.72548
    T_G_M = T_ref_gal - delta_t - T_CMB
    T_G_M_fit = T_ref_gal_fit - delta_t_fit - T_CMB

    T_G_M_error = np.sqrt(Sigma_T_ref_gal**2 + delta_t_error**2 + dT_CMB**2)
    
    T_G_M_error_fit = np.sqrt(Sigma_T_ref_gal_fit**2 + delta_t_error_fit**2 + dT_CMB**2)
    
    nan_mask_index = np.where(np.isnan(S_Moon_fit) == True)[0]
    S_Moon_fit = np.delete(S_Moon_fit, nan_mask_index,)
    S_Moon_err_fit = np.delete(S_Moon_err_fit, nan_mask_index,)
    T_G_M_fit = np.delete(T_G_M_fit, nan_mask_index)
    freq_fit =  np.delete(freq_fit, nan_mask_index)
    T_G_M_error_fit = np.delete(T_G_M_error_fit, nan_mask_index)
    delta_t_fit = np.delete(delta_t_fit, nan_mask_index)
    delta_t_error_fit = np.delete(delta_t_error_fit, nan_mask_index)
    

    print(T_G_M_fit.shape)
    
    if len(T_G_M_fit)==0:
        popt = np.nan*np.zeros(3)
        pcov = np.nan*np.zeros(shape=(3,3))
        errs = popt
        T_gal = np.nan*np.zeros(len(delta_t))
        T_gal_error = T_gal
        
        return popt, pcov, errs, T_gal, T_gal_error, freq_fit, T_G_M_fit, T_G_M_error_fit

    else:
        try:
            if meanover==False:
                popt, pcov, = curve_fit(f=T_gal_minus_moon_func, xdata=freq_fit, ydata=T_G_M_fit,\
                                        sigma=T_G_M_error_fit, absolute_sigma=False)

                errs = np.sqrt(np.diag(pcov))

                T_gal = popt[0] - delta_t - T_CMB
                T_gal_error = np.sqrt(errs[0]**2 + delta_t_error**2)
                
                return popt, pcov, errs, T_gal, T_gal_error, freq_fit, T_G_M_fit, T_G_M_error_fit

            elif meanover==True:
                popt, pcov, = curve_fit(f=T_gal_minus_moon_func, xdata=freq_fit, ydata=T_G_M_fit)

                errs = np.sqrt(np.diag(pcov))

                T_gal = popt[0] - delta_t - T_CMB
                T_gal_error = np.sqrt(errs[0]**2 + delta_t_error**2)
            
        
                return popt, pcov, errs, T_gal, T_gal_error, freq_fit, T_G_M_fit, T_G_M_error_fit

        except RuntimeError:
            print('not a fit')
            popt = np.nan*np.zeros(3)
            pcov = np.nan*np.zeros(shape=(3,3))
            errs = popt
            T_gal = np.nan*np.zeros(len(delta_t))
            T_gal_error = T_gal

            return popt, pcov, errs, T_gal, T_gal_error, freq_fit, T_G_M_fit, T_G_M_error_fit


        
def T_Gal_wtd_sigma_func(T_gal, inplt_full_beam, wtd):
    '''
    Get the weighted Galactic Temperature and Sigma from all observations
    
    params:
    T_gal: Galactic Temp. arr
    inplt_full_beam: interploted beam response 
    
    returns:
    T_gal_wtd: weighted Galactic Temp.
    Sigma_T_gal_wtd: weighted sigma Galactic Temp.
    '''
    if wtd == True:
        T_gal_wtd = np.nansum(T_gal*((inplt_full_beam)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0)

        Sigma_T_gal_wtd = np.sqrt(np.nansum((((inplt_full_beam)**2)*(T_gal-T_gal_wtd)**2), axis=0)/\
                                                        np.nansum((inplt_full_beam)**2, axis=0))
    elif wtd == False:
        T_gal_wtd = np.nanmean(T_gal, axis=0)
        Sigma_T_gal_wtd = np.nanstd(T_gal, axis=0)

    return T_gal_wtd, Sigma_T_gal_wtd



def Gal_Back(skymodel, month, mtype, version):

    if mtype=='old':
        if version=='1':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s'%(skymodel, month))
        elif version=='2':
            Temp_refl = np.loadtxt('../../old_sky_temp/GSM_mean_temp_from_Moon_%s_%s2'%(skymodel, month))
            Temp_moon_back = np.loadtxt('../../old_sky_temp/Moon_back_mean_temp_%s_%s2'%(skymodel, month))
    elif mtype=='new':
        Temp_refl = np.loadtxt('../../sky_temp_new/Ref_T_Gal_%s_%s'%(skymodel, month))*0.07
        Temp_moon_back = np.loadtxt('../../sky_temp_new/GSM_T_Gal_%s_%s'%(skymodel, month))
        
        
    Temp_moon_back_mean = np.nanmean(Temp_moon_back, axis=0)
    Temp_refl_mean = np.nanmean(Temp_refl, axis=0)

    return Temp_moon_back_mean, Temp_refl_mean

def T_gal_alpha(nu, T_Gal, alpha):
    '''
    Fitting funcation of the Synchrotron Galactic Sky
    
    params:
    nu: frequency
    T_Gal: Galactic Temperature
    alpha: Synchrotron spectral index
    
    returns: Galactic
    '''
    return T_Gal*((nu/150)**alpha)

def dT_gal_ref_gal(T, freq, T150, alpha, d_T150, d_alpha):
    dt = T * np.sqrt((d_alpha* np.log(freq/150.))**2 + (d_T150/T150)**2)
    
    return dt

def dT_gal_ref_gal_mean(dt, dalpha):
    N = dt.shape[0]
    dt_mean = np.sqrt(np.sum(dt**2, axis=0)/N)
    dalpha_mean = np.sqrt(np.sum(dalpha**2, axis=0)/N)
    return dt_mean, dalpha_mean


ncoarse = 5
nfine = 768

## dtv 180 - 210 MHz
## fm 80-110 MHz
## fm 100 MHz index
## 150 MHz index

FM_index = np.arange(354, 856) # showing the corresponding indices of the channels
dtv_index = np.arange(2355, 3356)
FM_index  = np.arange(362, 909)
ch_100_MHz_index = 660
ch_150_MHz_index = 1879
ch_200_MHz_index =  3098
ch_mid_ch_index = np.arange(1182,2360)
orbcomm_index = np.arange(1440, 1709)
orbcomm_index_updated = np.arange(1182, 1539)
## freq in MHz
freq = np.linspace(72.96, 230.4, 768*5)