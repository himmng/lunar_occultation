import numpy as np
from fm_flux import FM_CALC

catalog_file = '/home/himanshu/Dropbox/phd/milestone2/lunar_occultation/FM_simulation/full_earth_FM_moonrak.npy'

FM = FM_CALC(catalog_path = catalog_file,)

altaz_dis = FM.get_altaz_n_dist(obsID=1324432943, FM_loc=None, save_as_array=False, savefilename=None, savepath=None)

stations = FM.get_station_RFIcontributions(obsID=1324432943, altaz_array=altaz_dis,\
        randomise_zero_power_stations=True, save_as_array=False, savefilename=None, savepath=None)

#print(altaz_dis.shape)
print(stations, stations.shape)
