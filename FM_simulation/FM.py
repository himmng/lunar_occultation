from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun, get_moon
from astropy import units as u
from astropy.time import Time
import time
import numpy as np
import sys
from lal import gpstime
import lal
import numpy as np
lal.gpstime.GPS_EPOCH = lal.EPOCH_J2000_0_JD


def zen_an(obsid, lat, lon,):

    mwa_latitude_dec_deg = -26.70331940
    mwa_longitude_dec_deg = 116.67081524
    mwa_elevation = 377.83

    ut = lal.gpstime.gps_to_utc(int(obsid))

    mwa_loc = EarthLocation(lat=mwa_latitude_dec_deg*u.deg, lon=mwa_longitude_dec_deg*u.deg, height=mwa_elevation*u.m)
    ant_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)

    Moon_time = Time(ut)
    ant_time = Time(ut)
    
    altaz_moon = AltAz(obstime=Moon_time, location=mwa_loc)
    altaz_ant = AltAz(obstime=ant_time, location=ant_loc)

    zen_ang_mwa = get_moon(Moon_time,).transform_to(altaz_moon)
    zen_ang_ant = get_moon(ant_time,).transform_to(altaz_ant)

    return zen_ang_mwa, zen_ang_ant

#########################################################################

#start = time.time()
ind = int(sys.argv[1])

obsID = np.loadtxt('../obsIDs_Dec2015', dtype=np.int32)

on_ind = int(2*ind)
off_ind = int(on_ind + 1)

on_obsid = obsID[on_ind]
off_obsid = obsID[off_ind]

FM_loc = np.load('full_earth_FM_moonrak.npy')

ang_dis = np.zeros(shape=(len(FM_loc)+1, 3))

#print(on_obsid)

for j in range(0, len(FM_loc)):

    zz = zen_an(obsid=on_obsid, lat=float(FM_loc[j][1]), lon=float(FM_loc[j][2]),)

    if j == 0:

        ang_dis[j][0] = zz[0].az.deg
        ang_dis[j][1] = zz[0].alt.deg
        ang_dis[j][2] = zz[0].distance.m

        ang_dis[j+1][0] = zz[1].az.deg
        ang_dis[j+1][1] = zz[1].alt.deg
        ang_dis[j+1][2] = zz[1].distance.m
    
    else:
        j = j+1
        ang_dis[j][0] = zz[1].az.deg
        ang_dis[j][1] = zz[1].alt.deg
        ang_dis[j][2] = zz[1].distance.m

np.save('data/Dec2015/altaz_%d'%on_obsid, ang_dis)

#end = time.time()

#print('Time taken,' (end-start))