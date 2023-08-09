try:
    import os
    import sys
    import lal
    import json
    import math
    import time
    import ephem
    import mpdaf
    import numpy as np
    import healpy as hp
    import mwa_hyperbeam
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import astropy.coordinates as ac

    from lal import gpstime
    from astropy.io import fits
    from astropy.wcs import WCS
    from matplotlib import colors
    from erfa import hd2ae, ae2hd
    from astropy.time import Time
    from astropy import units as u
    from astropy import time as ttm
    from healpy.pixelfunc import pix2ang
    from skyfield.api import Topos, load, utc
    from datetime import date, datetime, timedelta
    from astropy.coordinates import ICRS, Galactic, FK4, FK5, Angle,\
          SkyCoord, EarthLocation, solar_system_ephemeris,\
        AltAz, Latitude, Longitude, get_sun, get_moon 
    from pygdsm import GSMObserver2016, GSMObserver, LFSMObserver, GlobalSkyModel,\
    GlobalSkyModel2016, LowFrequencySkyModel, HaslamObserver, HaslamSkyModel

except ImportError:
    
    raise ImportError

lal.gpstime.GPS_EPOCH = lal.EPOCH_J2000_0_JD


#normalise vectors to unit vectors:
def magnitude(v):
    return np.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def normalise(v):
    vmag = magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def skyf_chgctr(obsIDs, n):
    #in decimal degrees (for skyview)
    mwa_latitude_dec_deg = -26.70331940
    mwa_longitude_dec_deg = 116.67081524
    mwa_elevation = 377.83
    
    radec= []
    for i in range(0, len(obsIDs)):
        ut = lal.gpstime.gps_to_utc(int(obsIDs[i])) #UTC time correspond to GPS-Time (OBSID)
        ut_add = timedelta(seconds=n)  #for adding extra time to the phase centre 
        ut_tot = ut + ut_add           ## adding middle time for observation
        ut_tot = ut_tot.replace(tzinfo=utc)
        ts = load.timescale()
        t = ts.from_datetime(ut_tot)

        planets = load('de421.bsp')
        earth=planets['earth']
        
        MWA = earth + Topos(latitude_degrees=mwa_latitude_dec_deg,
        longitude_degrees=mwa_longitude_dec_deg ,elevation_m=mwa_elevation)
        
        MWA2 = EarthLocation(lat=-26.70331940*u.deg, lon=116.67081524*u.deg, height=377.83*u.m)
        new = ac.get_moon(time=ttm.Time(val=ut_tot, format='datetime', location=MWA2),location=MWA2,)
        
        new = SkyCoord(new, equinox='J2000')
        moon = planets['moon']
        astrometric_moon_centre_ephem=MWA.at(t).observe(moon)
        moon_centre_ra, moon_centre_dec, moon_centre_distance = astrometric_moon_centre_ephem.radec()
        
        moon_centre_distance = moon_centre_distance.to(u.km)
        
        ra = '' ## format in HH:MM:SS
        
        ra = '%d'%moon_centre_ra.hms()[0] +\
        ':'+'%d'%moon_centre_ra.hms()[1]+\
        ':'+'%.2f'%moon_centre_ra.hms()[2]

        dec = '' ## format in DD:MM:SS
        
        dec = '%d'%moon_centre_dec.dms()[0]+':' +\
        '%d'%abs(moon_centre_dec.dms()[1])+':'+\
        '%.2f'%abs(moon_centre_dec.dms()[2])
        
        RAm = mpdaf.obj.hms2deg(ra) ## Moon's RA in Deg. units
        DECm = mpdaf.obj.dms2deg(dec) ## Moon's DEC in Deg. units
        radec.append([RAm, DECm, float(moon_centre_distance.to(u.km)/u.km)
                     ])
    radec = np.array(radec)
    return radec

def submit_job(obs):
    os.system('mwa_client -c %s'%obs) # .csv file as obs

def status_check():
    os.system('giant squid -j')
    
def read_obsID(path, filename, mn, mx): #reading the ASVO jobID from the file
    '''
    path: provide path to the file as string
    filename: provide name of the file as string
    The jobs must be submitted in a single run
    mn: first job ID
    mx: last job ID
    '''
    with open(path+filename,) as f:
        data = f.read()
    js = json.loads(data)
    jobID = np.arange(mn, mx+1, 1)
    # saving obsids into a file for further process
    obsID = ''
    for i in range(len(jobID)):
        obsID = obsID+'%s'%js.get('%s'%jobID[i]).get('obsid')+'\n'
    f1 = open('obsID', 'w')
    f1.writelines(obsID)
    f1.close()
    print('obsID file written!')


def zen_an(obsid):
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

    Moon_time = Time(ut, format='gps')
    
    altaz_moon = AltAz(obstime=Moon_time, location=mwa_loc)

    altaz_moon = get_moon(Moon_time,).transform_to(altaz_moon)
    m = get_moon(Moon_time,)
    return altaz_moon, m

def observatory(radec, location):
    
    mwa_latitude_dec_deg = -26.70331940 ## mwa latitude deg.
    mwa_longitude_dec_deg = 116.67081524 ## mwa long. deg.
    mwa_elevation = 377.83
    
    if location=='Moon':
        earth_mwa_RA_deg = []
        earth_mwa_dec_deg = []  

        #Define the RA and DEC of the MWA site on Earth as viewed from the Moon 
        earth_mwa_RA_deg=radec[0]-180.0
        earth_mwa_dec_deg=-1.0*radec[1]


        moon_observatory_lat=earth_mwa_dec_deg

        moon_observatory_lon = float(earth_mwa_RA_deg)

        latitude, longitude, elevation = moon_observatory_lat, moon_observatory_lon, 0
        
    elif location == 'MWA':
        
        latitude, longitude, elevation = mwa_latitude_dec_deg, mwa_longitude_dec_deg, mwa_elevation
        
    
    return latitude, longitude, elevation


def skymap_estimation(obsIDs, obsID_index, skymodel, location, month, n):

    step = 10
    
    freq_array = np.linspace(70, 230, 32)
    
    moon_diameter_arcmin=33.29
        
    moon_diameter_deg = 0.0166667 * moon_diameter_arcmin
        
    moon_radius_deg = moon_diameter_deg*2.
    moon_radius_radians = np.deg2rad(moon_radius_deg)

    radec = skyf_chgctr(obsIDs=obsIDs, n=n)
    
    radec = radec[obsID_index]
    
    if skymodel=='GSM2016':
        
        ov = GSMObserver2016()
        NSIDE_interim= 1024
        
    elif skymodel=='GSM':
        
        NSIDE_interim= 512
        ov = GSMObserver()
        
    elif skymodel=='LFSM':
        
        NSIDE_interim= 256
        ov = LFSMObserver()
        
    elif skymodel=='Haslam':
        
        NSIDE_interim= 512
        ov = HaslamObserver()
        
    if location=='Moon':
            
        latitude, longitude, elevation = observatory(radec=radec, location='Moon')
        ov.lon = np.deg2rad(longitude)
        ov.lat = np.deg2rad(latitude)
        ov.elev = elevation
        ov.date = gpstime.gps_to_utc(int(obsIDs[obsID_index])) 
        
        
        disk_averaged_temp_array=np.zeros(len(freq_array))
        moon_map=np.ma.zeros(hp.nside2npix(NSIDE_interim))
        moon_map_full = np.ma.zeros(shape=(len(freq_array), len(moon_map)))
        index = np.arange(0, len(moon_map))
        
        for freq_index, freq_MHz in enumerate(freq_array):

            gsm_map_from_moon=ov.generate(freq_MHz)

            long_udeg = longitude * u.deg
            lat_udeg = latitude * u.deg
            rot_custom = hp.Rotator(rot=[long_udeg.to_value(u.deg), lat_udeg.to_value(u.deg)], inv=True)
            
            rotated_gsm = rot_custom.rotate_map_pixel(gsm_map_from_moon)
            
            moon_map_full[freq_index] = rotated_gsm
                
            disk_averaged_temp_array[freq_index]= np.nanmean(rotated_gsm)
            
        
        return moon_map_full, disk_averaged_temp_array
        
    elif location=='MWA':
        
        latitude, longitude, elevation = observatory(radec=radec, location='MWA')
        ov.lon = np.deg2rad(longitude)
        ov.lat = np.deg2rad(latitude)
        ov.elev = elevation
        ov.date = gpstime.gps_to_utc(int(obsIDs[obsID_index]))
    
        gsm_map_from_MWA_arr = np.zeros(shape=(len(freq_array),hp.nside2npix(NSIDE_interim)))
        gsm_map_from_MWA_meanTemp_arr = np.zeros(len(freq_array))
        #gsm_map_from_MWA_arr_copy = np.zeros(gsm_map_from_MWA_arr.shape)
        for freq_index, freq_MHz in enumerate(freq_array):

            
            gsm_map_from_MWA = ov.generate(freq_MHz)

            long_udeg = longitude * u.deg
            lat_udeg = latitude * u.deg
            rot_custom = hp.Rotator(rot=[long_udeg.to_value(u.deg), lat_udeg.to_value(u.deg)], inv=True)
            
            rotated_gsm = rot_custom.rotate_map_pixel(gsm_map_from_MWA)
            
            #gsm_map_from_MWA_copy = np.copy(rotated_gsm)


            RA = radec[0]
            DEC = radec[1]

            mfits = fits.open('/home/himanshu/pawsey/phase1/data/%s2015/cal_ms/%d.metafits'%\
                              (month, obsIDs[obsID_index]))[0]
            LST = mfits.header['LST']
            HA = LST - RA

            if DEC <= 0. :
                theta_moon = np.pi/2.  + np.deg2rad(np.abs(DEC))
            elif DEC > 0. :
                theta_moon = np.pi/2. - np.deg2rad(np.abs(DEC))

            if HA < 0. :
                phi_moon = np.deg2rad(longitude) + np.deg2rad(np.abs(HA))
            elif HA >= 0. :
                phi_moon = np.deg2rad(longitude) - np.deg2rad(np.abs(HA))

        

            vec = hp.ang2vec(theta= theta_moon, phi=phi_moon,)

            ipix_disc_moon = hp.query_disc(nside=NSIDE_interim,\
                                           vec=vec,\
                                           radius=moon_radius_radians)
            
            map_index = np.arange(0, len(gsm_map_from_MWA)) 
            
            discarded_index = np.delete(map_index, ipix_disc_moon)

            gsm_map_from_MWA_meanTemp_arr[freq_index] = np.nanmean(rotated_gsm[ipix_disc_moon])
            
            rotated_gsm[discarded_index] = np.nan #max(rotated_gsm)
            
            gsm_map_from_MWA_arr[freq_index] = rotated_gsm
            

        return rotated_gsm, gsm_map_from_MWA_arr, gsm_map_from_MWA_meanTemp_arr
    

def skymap_estimation_v2(obsIDs, radec_old, radec, skymodel, location, month):
    
    step = 10
    freq_array = np.linspace(70, 230, 32)
    
    
    if skymodel=='GSM2016':
        ov = GSMObserver2016()
        NSIDE_interim= 1024
    elif skymodel=='GSM':
        NSIDE_interim= 512
        ov = GSMObserver()
    elif skymodel=='LFSM':
        ov = LFSMObserver()
    elif skymodel=='Haslam':
        ov = HaslamObserver()
        
    if location=='Moon':
            
        latitude, longitude, elevation = observatory(radec=radec_old, location='Moon')
        ov.lon = longitude*(np.pi/180.)
        ov.lat = latitude*(np.pi/180.)
        ov.elev = elevation*(np.pi/180.)
        ov.date = gpstime.gps_to_utc(int(obsIDs))
            
        
        disk_averaged_temp_array=np.zeros(len(freq_array))
        moon_map=np.zeros(hp.nside2npix(NSIDE_interim))
        moon_map_full = np.zeros(shape=(len(freq_array), len(moon_map) ))
        index = np.arange(0, len(moon_map))
        mfits = fits.open('/home/himanshu/pawsey/phase1/data/%s2015/metafits/%d.metafits'\
                          %(month, obsIDs))
        for freq_index, freq_MHz in enumerate(freq_array):

            gsm_map_from_moon=ov.generate(freq_MHz)
            
            ra, dec = radec ## RA-DEC of the Moon
            
            RAm = mpdaf.obj.hms2deg(ra) ## Moon's RA in Deg. units
            DECm = mpdaf.obj.dms2deg(dec) ## Moon's DEC in Deg. units
            print(RAm, DECm)
            HAm = RAm + 180 - mfits[0].header['LST']
            
            HAm = HAm*(np.pi/180.)
            DECm = DECm*(np.pi/180.) + (np.pi/2.)
            print(RAm, DECm)
            
            azm, zam =  hd2ae(ha=HAm, dec=DECm, phi=-latitude*(np.pi/180.))
            print(azm, zam)
            zenith_theta= zam #np.pi/2# - DEC*(np.pi/180.)

            zenith_phi= azm #HA*(np.pi/180.)

            zenith_vector=hp.ang2vec(zenith_theta, zenith_phi)

            pixel_theta, pixel_phi=hp.pix2ang(NSIDE_interim, index)        

            moon_normal_vector=hp.ang2vec(pixel_theta, pixel_phi)

            earth_moon_vector=zenith_vector*radec_old[2]

            incident_vector=-earth_moon_vector+moon_normal_vector

            incident_vector_unit=incident_vector/np.linalg.norm(incident_vector,axis=1, keepdims=True)

            reflected_vector = 2.0*np.reshape(np.sum(moon_normal_vector*incident_vector_unit,axis=1), \
                           (len(moon_normal_vector), 1))*moon_normal_vector - incident_vector_unit


            reflected_vector=reflected_vector*-1.0

            gsm_pixel_mapped=hp.vec2pix(NSIDE_interim,reflected_vector[:,0],\
                                                    reflected_vector[:,1],reflected_vector[:,2])

            gsm_pixel_temp=gsm_map_from_moon[gsm_pixel_mapped]

            dot_product=np.sum(-moon_normal_vector*incident_vector_unit, axis=1)

            gsm_pixel_temp_reflected= gsm_pixel_temp*dot_product*0.07

            moon_map[index]=gsm_pixel_temp_reflected
            
            
            moon_map[moon_map < 0] = np.nan

            disk_averaged_temp=np.nanmean(moon_map)

            disk_averaged_temp_array[freq_index]=disk_averaged_temp
                
            moon_map_full[freq_index] = moon_map
                
            moon_map_full = np.ma.masked_invalid(moon_map_full)
            #disk_averaged_temp_array_full.append(disk_averaged_temp_array)
            
            #disk_averaged_temp_array_full = np.array(disk_averaged_temp_array_full)
            print('done', freq_index)
        
        return moon_map_full, disk_averaged_temp_array
        
    elif location=='MWA':
        #NSIDE_interim= 1024   
        latitude, longitude, elevation = observatory(radec=radec_old, location='MWA')
        ov.lon = longitude*(np.pi/180.)
        ov.lat = latitude*(np.pi/180.)
        ov.elev = elevation*(np.pi/180.)
            
        moon_diameter_arcmin=33.29
        moon_diameter_deg=moon_diameter_arcmin/60.


        moon_radius_deg=moon_diameter_deg/2.
            
        ov.date = lal.gpstime.gps_to_utc(int(obsIDs))
        
        gsm_map_from_MWA_mean = np.zeros(len(freq_array))
        mfits = fits.open('/home/himanshu/pawsey/phase1/data/%s2015/metafits/%d.metafits'\
                                  %(month, obsIDs))
        for freq_index, freq_MHz in enumerate(freq_array):
                
            gsm_map_from_MWA = ov.generate(freq_MHz)
            index = np.arange(0, len(gsm_map_from_MWA))
            pixel_long, pixel_lat = hp.pix2ang(NSIDE_interim,index, lonlat=True,)  
                

            HA = radec_old[0] - mfits[0].header['LST']
            DEC = radec_old[1]
            DEC = 45.-DEC

            discard_long_0 = np.where(pixel_long<HA-moon_diameter_deg)[0]
            discard_long_1 = np.where(pixel_long>HA+moon_diameter_deg)[0]
            discard_long = np.concatenate([discard_long_0, discard_long_1])

            discard_lat_0 = np.where(pixel_lat<DEC-moon_diameter_deg)[0]
            discard_lat_1 = np.where(pixel_lat>DEC+moon_diameter_deg)[0]
            discard_lat = np.concatenate([discard_lat_0, discard_lat_1])

            discard = np.array(list(set(index) - set(discard_long)- set(discard_lat)))

            discard_rest = np.array(list(set(index)-set(discard)))
                
            gsm_map_from_MWA[discard_rest] = np.nan
                
            gsm_map_from_MWA = np.ma.masked_invalid(gsm_map_from_MWA)
                
            gsm_map_from_MWA_mean[freq_index] = np.nanmean(gsm_map_from_MWA)
            print('done', freq_index)
        return gsm_map_from_MWA, gsm_map_from_MWA_mean ## it is the blocked background temperature

def T_map_gen(obsIDs, skymodel, location1, location2, month, n=116):
    
    step = 10
    mids=4
    radec = skyf_chgctr(obsIDs=obsIDs, n=n,)


    moon_back_map = []
    moon_back_temp_mean = []
    
    GSM_map_from_moon = []
    GSM_mean_temp_from_moon = []
    
    for obsID_index in range(mids, len(obsIDs)-10, step):

        moon_back = skymap_estimation(obsIDs=obsIDs, obsID_index=obsID_index,\
                                    skymodel=skymodel, location=location1, month=month, n=n)

        GSM_moon = skymap_estimation(obsIDs=obsIDs, obsID_index=obsID_index,\
                                    skymodel=skymodel, location=location2, month=month, n=n)



        moon_back_temp_mean.append(moon_back[2])

        GSM_mean_temp_from_moon.append(GSM_moon[1])
        

        #np.savetxt('sky_temp_new/GSM_T_Gal_%s_%s_%d'%(skymodel, month, obsID_index), moon_back_temp_mean)
        #np.savetxt('sky_temp_new/Ref_T_Gal_%s_%s_%d'%(skymodel, month, obsID_index), GSM_mean_temp_from_moon)
        print('done', obsID_index)
        
    moon_back_temp_mean = np.ma.array(moon_back_temp_mean)
    GSM_mean_temp_from_moon = np.ma.array(GSM_mean_temp_from_moon)
    
    np.savetxt('sky_temp_new/GSM_T_Gal_%s_%s'%(skymodel, month), moon_back_temp_mean)
    np.savetxt('sky_temp_new/Ref_T_Gal_%s_%s'%(skymodel, month), GSM_mean_temp_from_moon)
    print('completed, %s'%skymodel)