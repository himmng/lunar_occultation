try:
    import lal
    import sys
    import time
    import ephem
    import numpy as np
    import astropy.units as u
    
    from lal import gpstime
    from astropy.io import fits
    from astropy.time import Time
    from datetime import datetime, timedelta
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz,\
          get_sun, get_moon

except ImportError:
    raise ImportError

lal.gpstime.GPS_EPOCH = lal.EPOCH_J2000_0_JD


def zen_an(obsid, lat, lon,):
    '''
    This function outputs the altitude and azimuth angles [in deg.] 
    between the zenith of FM radio stations and the Moon, and MWA and the Moon
    at the time of the observation
    
    parameters: 
    obsid: GPS Time of the observation.
    lat: Latitude of the FM station [units: deg.]
    lon: Longitude of the FM station [units: deg.]
    
    returns: Altitude, Azimuth of MWA with Moon[units: deg.], 
    and Altitude, Azimuth of the FM station with Moon[units: deg.]
    '''
    
    
    
    mwa_latitude_dec_deg = -26.70331940
    mwa_longitude_dec_deg = 116.67081524
    mwa_elevation = 377.83
    ut = obsid # in UTC datetime.datetime
    mwa_loc = EarthLocation(lat=mwa_latitude_dec_deg*u.deg, \
                            lon=mwa_longitude_dec_deg*u.deg, \
                            height=mwa_elevation*u.m)
    ant_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)

    Moon_time = Time(ut, format='gps')
    ant_time = Time(ut, format='gps')
    
    altaz_mwa = AltAz(obstime=Moon_time, location=mwa_loc)
    altaz_ant = AltAz(obstime=ant_time, location=ant_loc)

    altaz_mwa = get_moon(Moon_time,).transform_to(altaz_mwa)
    altaz_ant = get_moon(ant_time,).transform_to(altaz_ant)

    return altaz_mwa, altaz_ant

def station_contributions(obsid):
    '''
    This function chooses the FM stations for which 
    the Moon is above the horizon at the time of the observation.
    
    Already provided parametes are the FMlist at the given observation, and the Full FM catalog.
    FMlist contains the information of the ALTITUDE and AZIMUTH of the FM stations at the time of the observation.
    FM catalog contains the information of the station power, frequency, etc.
    parameters:
    obsid: GPS Time of the observation
    
    returns:
    stations{above horizon}, Altitude, Azimuth of all FM stations, Altitude, Azimuth of MWA, station count
    '''
    
    altaz_ant = np.load(path_FMlist+'altaz_%d.npy'%obsid)
    ## this array holds altitude-azimuth info. of the FM transmitters (index 0 is MWA's alt-az )

    stations = np.load('../FM/full_earth_FM_moonrak.npy') 
    ## includes FM transmitters across the earth.[id, lat, long, freq[MHz], power]
    
    altaz_mwa = altaz_ant[0] ## alt-az of moon from MWA
    
    indx = np.where(altaz_ant[1:,1]>=0.)[0] ## altitude > 0. moon is above horizon
    altaz_ant = altaz_ant[indx] 
    stations = stations[indx] # using alt >0 deg stations
    
   
    ind = np.where(stations[:,4]==0)[0] ## checking 0 KW power stations 
    stations_non0 = np.delete(stations[:,4], ind,) 
    ##deleting 0 power stations and then average power amongst non zero stations
    
    stations_non0_avg = np.average(stations_non0) ## taking average power
    
    stations_non0_std = np.std(stations_non0)  ## taking Std. deviation 
    

    for i in range(len(ind)):
        rand_power = np.random.normal(loc=stations_non0_avg, scale=stations_non0_std) 
        ## giving a random power to zero power stations
        stations[ind[i]][4] = np.abs(rand_power) # to stations
        
    return stations, altaz_ant, altaz_mwa, len(stations)


def Power_at_Moon(stations, altaz_ant, pattern):
    '''
    This function outputs the expected reflected power receieved at the MWA location from all of 
    the contributing FM stations during the time of the observation.
    
    parameters:
    
    stations: The stations is an array that has [stationID, Lat, Long, Operating Freq,
    Power [KW]] stored in it.
    altaz_ant: This array has [alt, az, distance] of all the FM stations.
   
    returns:
    
    '''

    moon_albedo = 0.07
    moon_solid_angle = 7.365*10**(-5)
    moon_radius = 1737.4*(10**3)
    MWA_A_eff = 22.2
        
    transmitter_BW = 180000.0                 
    MWA_bandwidth = 30720000
    Gain = 1.0
    
    Radar_Cross_Section = 0.081*(np.pi * (moon_radius**2))
    
    power = np.zeros(len(stations)*5) ## input all the stations at given obsid
    
    freq = np.zeros(len(stations)*5) ## store corresponding frequency of stations
    
    for i in range(0, len(stations), 5):
        
        # altaz array stores the [alt, az, distance] from 

        distance1 = np.average(altaz_ant[:,2]) # distance station-Moon (meters)

        P_at_Moon = ((stations[i][4]*(10**3))/((distance1 - moon_radius)**2))\
                    *Radar_Cross_Section # power of station in Watts
        
        # power at Moon
        if pattern == 'constant':
            power[i] =  P_at_Moon
            power[i+1] =  P_at_Moon
            power[i+2] =  P_at_Moon
            power[i+3] =  P_at_Moon
            power[i+4] =  P_at_Moon

            freq[i] = stations[i][3]- 0.080
            freq[i+1] = stations[i][3]-0.040
            freq[i+2] = stations[i][3] # freq in MHz
            freq[i+3] = stations[i][3]+0.040
            freq[i+4] = stations[i][3]+0.080
        elif pattern == 'Gaussian':
            power[i] =  P_at_Moon*.707
            power[i+1] =  P_at_Moon*.95
            power[i+2] =  P_at_Moon
            power[i+3] =  P_at_Moon*.95
            power[i+4] =  P_at_Moon*.707

            freq[i] = stations[i][3]- 0.080
            freq[i+1] = stations[i][3]-0.040
            freq[i+2] = stations[i][3] # freq in MHz
            freq[i+3] = stations[i][3]+0.040
            freq[i+4] = stations[i][3]+0.080
        ## I can assume that power follows a gaussian pattern,
        ## instead of being constant throughout the 180kHz bandwidth
        ## here we need to understand that the station power peaks at the station frequency
        ## and assume that it was fell off FWHM at 90kHz (half around the bandwidth of 180kHz)
    
    rem = np.where(freq==0.0)[0]
    
    power = np.delete(power, rem)
    freq = np.delete(freq, rem)
    
    rem = np.where(freq<80.)[0]
    power = np.delete(power, rem)
    freq = np.delete(freq, rem)
    argIndx = np.argsort(freq) # sorting low to high freq
    freq = freq[argIndx]
    power = power[argIndx]
    
    freqFM_uni, uniq_indx = np.unique(freq, return_index=True) # unique indicies of FM frequencies
    
    power_summ = np.zeros(len(uniq_indx)) ## summing similar frequency power
    
    for i in range(len(power_summ)):
        if i == len(power_summ)-1:
            # last element 
            power_summ[i] = np.sum(power[uniq_indx[i]: len(freq)])
        else:
            power_summ[i] = np.sum(power[uniq_indx[i]:uniq_indx[i+1]])  # summing power within 0.1 MHz
    
    return power_summ, freqFM_uni

    

def flux_calc(power, freq, altaz_mwa):
    
    moon_albedo = 0.07
    moon_solid_angle = 7.365*10**(-5)
    moon_radius = 1737.4*(10**3)
    bandwidth = 180*(10**3)               ### assuming stations emits over 180KHz band
    MWA_A_eff = 22.2
    distance2 = altaz_mwa[2]    # distance mwa-Moon (meters)
    
    # flux = power (Watts) / (distance^2 * bandwidth * MWA_area)
    flux = power/((4*np.pi*(distance2-moon_radius)**2)*bandwidth*MWA_A_eff) 
    
    flux = flux * (10**26)
    return flux