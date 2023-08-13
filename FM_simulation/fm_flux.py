try:
    import lal
    from lal import gpstime
    import numpy as np
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import get_moon
    from astropy.coordinates import EarthLocation, AltAz
    
    np.seterr(divide='ignore', invalid='ignore')
    lal.gpstime.GPS_EPOCH = lal.EPOCH_J2000_0_JD

except IOError:
    raise IOError('repository absent')


class FM_CALC(object):
    """
    Evaluates the reflected FM from the FM catalog

    Args:
        object (object): 
    """
    
    def __init__(self, catalog_path, telescope_loc=None):
        
        """
        Initiation

        Args:
            catalog_path (str): path to the FM catalog file
            telescope_loc (tuple, , units: (degree, degrees, meter), optional): tuple of telescope location 
            (telescope_lat, telescope_lon, telescope_elev). Defaults to None and uses MWA's location parameters.
        """
        
        self.FM_catlog = np.load(catalog_path)
        
        if telescope_loc == None:
            # using MWA's location
            telescope_loc = (-26.70331940, 116.67081524, 377.83)
        self.telescope_loc = telescope_loc
    
    def get_altaz_n_dist(self, obsID, FM_loc=None, save_as_array=True, savefilename=None, savepath=None):
        
        """
        Uses astropy to get the altaz and distance of the telescope, FM station from the Moon at give GPSTIME.

        Args:
            obsID (int): GPSTIME at which the Moon's sky location is obtained.
            FM_loc (1D tuple, 1D list, 1D, 2d-array, units: (degree, degrees, meter)): (latitude, longitude, height)
            of the FM station or a location at which radio transmitter is. height can be put to zero meters. 
            if FM_loc is None, then uses class instance
            save_full_array (bool, optional): True if saving full altaz_and_distance as numpy array. Defaults to True.
            savefilename (str): if save_full_array is True, then provide filename to be save with.
            savepath (str): if save_full_array is True, then file provide location for the save.
            
        Returns:
        
            altitude, azimuth, distance of the Moon from telescope's location (tuple, units: (degrees, degrees, meters)),
            altitude, azimuth, distance of the Moon from FM station's location (tuple, units: (degrees, degrees, meters)
            
        """

        telescope_loc = EarthLocation(lat=self.telescope_loc[0]*u.deg, lon=self.telescope_loc[1]*u.deg,\
                                        height=self.telescope_loc[2]*u.meter)
        if FM_loc == None:
            FM_loc = self.FM_catlog[:,1:4] # index 1, 2 corresponds to lat, long of FM stations in degrees
            FM_loc[:,2] = 0 # putting heights to 0 meters
            
        if FM_loc.ndim > 1:
            FM_station_loc = EarthLocation(lat=FM_loc[:,0]*u.deg, lon=FM_loc[:,1]*u.deg, height=FM_loc[:,2]*u.m)
        elif FM_loc.ndim == 1:
            FM_station_loc = EarthLocation(lat=FM_loc[0]*u.deg, lon=FM_loc[1]*u.deg, height=FM_loc[2]*u.m)

        Moon_at_given_time = Time(int(obsID), format='gps')
        altaz_telescope = AltAz(obstime=Moon_at_given_time, location=telescope_loc)
        altaz_FM_station = AltAz(obstime=Moon_at_given_time, location=FM_station_loc)
        altaz_telescope = get_moon(Moon_at_given_time,).transform_to(altaz_telescope)
        altaz_FM_station = get_moon(Moon_at_given_time,).transform_to(altaz_FM_station)
  
        altaz_dis = np.empty(shape=(len(FM_loc)+1, 3)) 
        # adding +1 to store telescope's altaz, indices 0,1,2 corrresponds, altitude, azimuth, distance from Moon

        altaz_dis[0] = altaz_telescope.az.deg, altaz_telescope.alt.deg, altaz_telescope.distance.m
        altaz_dis[1:].T[:] = altaz_FM_station.az.deg, altaz_FM_station.alt.deg, altaz_FM_station.distance.m   
            
        if save_as_array == True:
            np.save(savepath+savefilename, altaz_dis) #(units,: deg, deg, meter)
            
        else:
            return altaz_dis

    def get_station_RFIcontributions(self, obsID, altaz_array=None, randomise_zero_power_stations=True,\
                                save_as_array=True, savefilename=None, savepath=None):
        
        """
        Get the FM stations contributing in the reflected RFI from the Moon at the given observation time.

        Args:

            obsID (int): GPSTIME at which the Moon's sky location is obtained.
            altaz_array (bool, optional): if True then provide altaz array information, otherwise uses self.instance
            altaz_filename (str, optional): path of the altaz file.
            randomise_zero_power_stations (bool, optional): if randomise the zero/missing power stations based on the non-zeros powered stations
            save_as_array (bool, optional): True if want to save the station array as file
            savefilename (str, optional): if True, provide the filename to save, format:.npy
            savepath    (str, optional): path to save the file
            
        Raises:
            RuntimeError: uses function under main instances
            TypeError: bool required

        Returns:
        
            file or ndarray: FM contribution from where the Moon lying above the horizon during the scheduled observation from given telescope.
            
        """
            
        ## FM catalog includes FM transmitters across the earth.[station ID, lat(deg.), long(deg.), freq(MHz), power(KW)]
        FM_catalog = self.FM_catlog 
        
        altaz_stations = altaz_array[1:,:]
        altaz_telescope = altaz_array[0]
        
        missing_freq_index = np.where(FM_catalog[:,4]==0.)[0]
        FM_catalog = np.delete(FM_catalog, missing_freq_index,axis=0)
        altaz_stations = np.delete(altaz_stations, missing_freq_index, axis=0)
        
        if altaz_telescope[1]>0: # Moon is above the horizon at the telescope's location for given obsID (GPSTIME)
            
            contributing_stations_index = np.where(altaz_stations[:,1]>=0.)[0] # Moon above horizon at the location of FM stations
            altaz_stations = altaz_stations[contributing_stations_index] 
            FM_catalog = FM_catalog[contributing_stations_index]
            
            if type(randomise_zero_power_stations) != bool:
                raise TypeError('not a bool type')
            
            elif bool(randomise_zero_power_stations) == True:
                ## checking 0 KW power stations (missing data)
                
                nonzero_power_stations_ind = np.where(FM_catalog[:,4]!=0.)[0] 
                mean_power = np.nanmean(FM_catalog[:,4][nonzero_power_stations_ind])
                std_power = np.nanstd(FM_catalog[:,4][nonzero_power_stations_ind]) 
                 
                zero_power_stations_ind = np.where(FM_catalog[:,4]!=0.)[0] 
                
                rand_power = np.random.normal(loc=mean_power, scale=std_power, size=len(zero_power_stations_ind)) 
                ## giving unhealthy stations a random power
                FM_catalog[zero_power_stations_ind, 4] = np.abs(rand_power) # to stations
                    
            elif bool(randomise_zero_power_stations) == False:
                
                # discarding zero powered stations
                FM_catalog = np.delete(FM_catalog, zero_power_stations_ind, axis=0) 
                altaz_stations = np.delete(altaz_stations, zero_power_stations_ind, axis=0)
            # stored data as station ID, lat, long, freq, power, alt, az, distance
            stations = np.concatenate([FM_catalog, altaz_stations], axis=1) # typical shape N x 8
            ## adding telescope location in the zeroth index of the array # shape required 1 x 8
            # stored data as np.nan, lat, long, height, np.nan alt, az, distance (frequency is replaced by height)
            telescope_parameters = np.array([np.nan, self.telescope_loc[0],\
                                                        self.telescope_loc[1],\
                                                            self.telescope_loc[2],\
                                                                np.nan,\
                                                                    altaz_telescope[0], \
                                                                        altaz_telescope[1],\
                                                                            altaz_telescope[2]])
            telescope_parameters = telescope_parameters.reshape(1, len(telescope_parameters))
            stations = np.concatenate([telescope_parameters, stations], axis=0)
            
            if save_as_array == True:
                np.save(savepath+savefilename, stations)   
            else:   
                return stations
        else:
            print("The Moon is below the horizon at telescope's at given GPSTIME!")
            pass


    def get_FM_RFI_flux(self, obsID, station_BW=180, div=5, bandpass='constant', stations_array=None,\
                        save_as_array=True, savefilename=None, savepath=None):
        
        """
        This function outputs the expected reflected power received at the location of the observing telescope from all of 
        the contributing FM stations at the time of the observation.

        Args:
            obsID (_type_): _description_
            station_BW (int, optional): assumed bandwidth of the FM stations, (assumes all stations have same bandwidth). Defaults to 180 kHz.
            div (int, optional): division factor decides the number of frequency samples considered in the assumed bandwidth. Defaults to 5.
            bandpass (str, optional): assumes the bandpass structure of the FM stations assuming 180KHz. Available options, constant, Gaussian
            stations_array (optional): FM contribution from where the Moon lying above the horizon during 
            the scheduled observation from given telescope. Defaults to None.
            save_as_array (bool, optional): True if want to save the station array as file
            savefilename (str, optional): if True, provide the filename to save, format:.npy
            savepath    (str, optional): path to save the file

        Raises:
            TypeError: only self instance supported

        Returns:
            array: 
            freq (array): frequency array FM catalog. units: MHz
            flux-density (array): reflected RFI FM flux-density at the location of the telescope, units: Jy
            sigma-flux-density (array): errors based on the power in the flux-density, units: Jy
        """

        moon_radius = 1737.4*(10**3) *u.m
        MWA_A_eff = 22.2 *u.m *u.m
        transmitter_BW = 180000.0 *u.Hertz    
        Radar_Cross_Section = 0.081*(np.pi * (moon_radius**2))
    
        power = np.empty(len(stations_array[1:])* div) ## input all the stations at given obsid
        freq = np.empty(len(stations_array[1:])* div) ## store corresponding frequency of stations
        
        for st_count in range(1, len(power), div): # indexing over 5 elements
            try:
                # altaz array stores the [alt, az, distance] from 
                distance1 = np.average(stations_array[1:,-1])*u.m # distance station-Moon (meters)

                P_at_Moon = (((stations_array[st_count,4]*u.kilowatt).to(u.watt)/((distance1 - moon_radius)**2))*Radar_Cross_Section)
                
                freq_dev = (np.linspace(-station_BW/2, station_BW/2, div)*u.kHz).to(u.MHz) # in MHz
                freq[st_count:st_count+div] = stations_array[st_count:st_count+div][:,4]*u.MHz + freq_dev
                
                if bandpass == 'constant':
                    power[st_count:st_count+div] =  P_at_Moon
                    
                elif bandpass == 'Gaussian':
                    power_dev = np.array([.707, .95, 1., 0.95, .707])
                    power[st_count:st_count+div] =  P_at_Moon*power_dev
                
                ## I can assume that power follows a Gaussian-kind pattern,
                ## instead of being constant throughout the 180kHz bandwidth
                ## here we need to understand that the station power peaks at the station frequency
                ## and assume that it was fell off FWHM at 90kHz (half around the bandwidth of 180kHz)
            except:
                pass
        
        argIndx = np.argsort(freq.value) # sorting low to high freq
        freq = freq[argIndx]
        power = power[argIndx]
        
        freqFM_uniq, uniq_indx = np.unique(freq.value, return_index=True) # unique indicies of FM frequencies
        power_sum = np.empty(len(uniq_indx)) ## summing similar frequency power
        power_var = np.empty(len(uniq_indx))
        
        for sum_index in range(len(power_sum)):
            if sum_index == len(power_sum)-1:
                # last element 
                power_sum[sum_index] = np.sum(power[uniq_indx[sum_index]: len(freq)])
                power_var[sum_index] = np.std(power[uniq_indx[sum_index]: len(freq)])
            else:
                power_sum[sum_index] = np.sum(power[uniq_indx[sum_index]:uniq_indx[sum_index+1]])
                power_var[sum_index] = np.std(power[uniq_indx[sum_index]:uniq_indx[sum_index+1]])
        
        distance2 = stations_array[0][-1]*u.m
        # flux = power (Watts) / (distance^2 * bandwidth * MWA_area)
        flux_density = power_sum/((4*np.pi*(distance2-moon_radius)**2)* transmitter_BW*MWA_A_eff) 
        flux_density_std = power_var/((4*np.pi*(distance2-moon_radius)**2)* transmitter_BW*MWA_A_eff) 
        flux_density = (flux_density).to(u.Jy)
        flux_density_std = flux_density_std.to(u.Jy)  
        freq_flux = np.array([freqFM_uniq.value, flux_density.value, flux_density_std.value])
        
        if save_as_array == True:
            np.save(savepath+savefilename, freq_flux)
            
        else:   
            return freqFM_uniq, flux_density, flux_density_std

    def spec_RFI_st(self):
        
        #stt = self.station_RFIcontributions()
        #altaz_stat = stt[1]
        #mwa_altaz = stt[2]
        #i_eq_r_indx = []
        #for i in range(stt[3]):
        #    ## 4 deg
        #    x = np.isclose(mwa_altaz[1], altaz_stat[i][1], rtol=0.001, atol=1)
        #    if x == True:
        #        i_eq_r_indx.append(i)
        #    else:
        #        pass
        #    
        #i_eq_r_indx =np.array(i_eq_r_indx)
        #return i_eq_r_indx
        pass # TBD
