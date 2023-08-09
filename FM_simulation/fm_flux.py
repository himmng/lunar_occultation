
import sys
import lal
import time
import numpy as np
from lal import gpstime
import astropy.units as u
from astropy.time import Time
from astropy import units as u
from datetime import datetime, timedelta
from astropy.coordinates import get_sun, get_moon
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

np.seterr(divide='ignore', invalid='ignore')
lal.gpstime.GPS_EPOCH = lal.EPOCH_J2000_0_JD



def zen_an(obsid, lat, lon,):

    mwa_latitude_dec_deg = -26.70331940
    mwa_longitude_dec_deg = 116.67081524
    mwa_elevation = 377.83
    ut = obsid # in UTC datetime.datetime
    print(ut)
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

def moon_mask(iml):
    
    moon_albedo = 0.07
    moon_solid_angle = 7.365*10**(-5)
    earth_solid_angle = 8.625*10**(-4)
    moon_radius = 1737.4*(10**3)

    pix_size_deg=0.0085
    moon_diameter_arcmin = 33.29
    moon_diameter_deg = moon_diameter_arcmin/60
    moon_radius_deg = moon_diameter_deg/2. 
    #area of moon in deg_sq
    moon_area_deg_sq = np.pi*(moon_radius_deg)**2

    steradian_in_sq_deg = (180/np.pi)**2
    #solid angle subtended by Moon in Steradians
    Omega = moon_area_deg_sq/steradian_in_sq_deg

    # moon mask
    image_length = iml
    image_height = iml
    moon_radius_pix = np.round(moon_radius_deg/pix_size_deg)
    moon_mask = np.zeros((image_length,image_height))
    a,b = (image_length/2)-1, (image_height/2)-1
    y,x = np.ogrid[-a:image_length-a, -b:image_height-b]
    mask = x*x + y*y <= moon_radius_pix*moon_radius_pix
    moon_mask[mask]=1
    moon_mask = moon_mask.flatten('F')
    
    return moon_mask

def station_contributions(obsid):
    
    altaz_ant = np.load('data/Aug2015/altaz_%d.npy'%obsid)
    ## this array holds altitude-azimuth info. (index 0 is MWA's alt-az )

    stations = np.load('full_earth_FM_moonrak.npy') 
    ## includes FM transmitters across the earth.[id, lat, long, freq[MHz], power]
    
    altaz_mwa = altaz_ant[0] ## alt-az of moon from MWA
    
    indx = np.where(altaz_ant[1:,1]>=0.)[0] ## altitude > 0. moon is above horizon
    altaz_ant = altaz_ant[indx] 
    stations = stations[indx] # using alt >0 deg stations
    
   
    ind = np.where(stations[:,4]==0)[0] ## checking 0 KW power stations 
    stations_non0 = np.delete(stations[:,4], ind,) 
    ##deleting 0 power stations for average power amongst non0
    
    stations_non0_avg = np.average(stations_non0) ## taking average power
    
    stations_non0_std = np.std(stations_non0)  ## taking Std. deviation 
    

    for i in range(len(ind)):
        rand_power = np.random.normal(loc=stations_non0_avg, scale=stations_non0_std) 
        ## giving a random power
        stations[ind[i]][4] = np.abs(rand_power) # to stations
        
    return stations, altaz_ant, altaz_mwa, len(stations)


def Power_at_Moon(stations, altaz_ant, altaz_mwa):

    moon_albedo = 0.07
    moon_solid_angle = 7.365*10**(-5)
    earth_solid_angle = 8.625*10**(-4)
    moon_radius = 1737.4*(10**3)
    
    index = []
    power_summed =[]

    power = np.zeros(len(stations))
    Powers_Recieved_per_Hz = np.zeros(shape=len(stations))
    freq = np.zeros(len(stations))
    distance2 = np.zeros(len(stations))
    transmitter_BW = 180000.0
    MWA_bandwidth = 30720000
    Gain = 1.0
    Radar_Cross_Section = 0.081 * np.pi * moon_radius**2
    dilution_factor = transmitter_BW/MWA_bandwidth
    
    for i in range(len(stations)):
         
        distance1 = altaz_ant[i][2] # distance station-Moon (meters)
           
        distance2 = altaz_mwa[2]    # distance mwa-Moon (meters)
        
        P_at_Moon = ((stations[i][4]*(10**3))/(4*np.pi*(distance1 - moon_radius)**2))\
                    *Radar_Cross_Section*moon_albedo # power of station in KW
        
        power[i] =  P_at_Moon
        
        freq[i] = stations[i][3] * 10**6 # freq in Hz
        
              
    
    
    freq_int = np.around(freq, decimals=1 ) # making FM catalog freq in 0.1 MHz 
    
    argIndx = np.argsort(freq_int) # sorting low to high freq
    freqFM = freq_int[argIndx]
    power = power[argIndx]

    freqFM_uni, uniq_indx = np.unique(freqFM, return_index=True) # unique indicies of FM frequencies
    power_summ = np.zeros(len(uniq_indx)) ## summing similar frequency power
    for i in range(len(power_summ)):
        if i == len(power_summ)-1:
            power_summ[i] = np.sum(power[uniq_indx[i]: len(freqFM)]) # summing power within 0.1 MHz
        else:
            power_summ[i] = np.sum(power[uniq_indx[i]:uniq_indx[i+1]])
    
    return freqFM_uni, power_summ, #Powers_Recieved_per_Hz_sum

    

def flux_calc(power, freq, altaz_mwa):
    
    moon_albedo = 0.07
    moon_solid_angle = 7.365*10**(-5)
    earth_solid_angle = 8.625*10**(-4)
    moon_radius = 1737.4*(10**3)
    bandwidth = 10*(10**3)               ### assuming stations emits over 10KHz band
    Aeff_MWA = 22
    
    # 2 pi for reflection reflecting to half hemisphere
    distance2 = altaz_mwa[2]    # distance mwa-Moon (meters)
    flux = power/(4*np.pi*((distance2-moon_radius)**2)*bandwidth)
    flux = flux * (10**26)
    return flux
    


def spec_RFI_st(j):
    stt = station_contributions(j)
    altaz_stat = stt[1]
    mwa_altaz = stt[2]
    i_eq_r_indx = []
    for i in range(stt[3]):
        ## 4 deg
        x = np.isclose(mwa_altaz[1], altaz_stat[i][1], rtol=0.001, atol=1)
        if x == True:
            i_eq_r_indx.append(i)
        else:
            pass
        
    i_eq_r_indx =np.array(i_eq_r_indx)
    return i_eq_r_indx


ind = int(sys.argv[1])
obsids = np.loadtxt('../obsIDs_Aug2015', dtype=np.int32)

on_ind = int(2*ind)
off_ind = int(on_ind + 1)

stations, altaz_ant, altaz_mwa, st_count = station_contributions(obsids[on_ind])
freq, Power= Power_at_Moon(stations=stations, altaz_ant=altaz_ant, altaz_mwa=altaz_mwa)
flux = flux_calc(power=Power, freq=freq, altaz_mwa=altaz_mwa)

np.savetxt('data/Aug2015/freq_%d'%obsids[on_ind], freq)
np.savetxt('data/Aug2015/power_%d'%obsids[on_ind], Power)
np.savetxt('data/Aug2015/flux_%d'%obsids[on_ind], flux)
np.savetxt('data/Aug2015/altaz_ant_%d'%obsids[on_ind], altaz_ant)
np.savetxt('data/Aug2015/altaz_mwa_%d'%obsids[on_ind], altaz_mwa)
