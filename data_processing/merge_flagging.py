# From Kariuki: sivio on git
# Useful MS manipulation functions.
import sys
import numpy as np
from casacore import *
from casacore.tables import table, maketabdesc, makearrcoldesc
from casacore.tables import table, tablesummary
c = 299792458


def get_data(tbl, col="DATA"):
    """Grab data from a CASA measurement set (MS)
    Parameters
    ----------
    tbl : object
        Casacore table object
    col : str, optional
        The required MS column, by default "DATA"
    Returns
    -------
    array
        required data
    """
    data = tbl.getcol(col)
    return data


def get_uvw(tbl):
    """Grab the UVW data from a CASA measurement set (MS)
    Parameters
    ----------
    tbl : object
        Casacore table object
    Returns
    -------
    array
        required data
    """
    uvw = tbl.getcol("UVW")
    return uvw

def get_flags(tbl):
    """Grab the FLAG column from a CASA measurement set (MS)
    Parameters
    ----------
    tbl : object
        Casacore table object
    Returns
    -------
    array
        flags
    """
    flags = tbl.getcol("FLAG")
    return flags
    
def get_bl_num(tbl):
    """Grab the baseline number column from a CASA measurement set (MS)
    Parameters
    ----------
    tbl : object
        Casacore table object
    Returns
    -------
    array
        baseline number array
    """
    bl_num = tbl.getcol("BASELINE")
    return bl_num
    

def get_phase_center(tbl):
    """
    Grabs the phase centre of the observation in RA and Dec
    Parameters
    ----------
    tbl : casacore table.
        The casacore mset table opened with readonly=False.
    Returns
    -------
    float, float.
        RA and Dec in radians.
    """
    ra0, dec0 = tbl.FIELD.getcell("PHASE_DIR", 0)[0]
    return ra0, dec0


def get_channels(tbl, ls=True):
    """Get frequency or wavelength of an observation
    Parameters
    ----------
    tbl : casacore table.
        The casacore MS table.
    ls : bool, optional
        Convert to wavelength, by default True
    Returns
    -------
    [type]
        [description]
    """
    if ls:
        chans = c / tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    else:
        chans = tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    return chans


def get_ant12(mset):
    """[summary]
    Parameters
    ----------
    mset : string
        Path to measurement set
    Returns
    -------
    tuple(array, array)
        array antenna ID correlations
    """
    ms = table(mset, readonly=False, ack=False)
    antenna1 = ms.getcol("ANTENNA1")
    antenna2 = ms.getcol("ANTENNA2")
    return antenna1, antenna2


def put_col(tbl, col, dat):
    """add data 'dat' to the column 'col'"""
    tbl.putcol(col, dat)


def add_col(tbl, colnme):
    """Add a column 'colnme' to the MS"""
    col_dmi = tbl.getdminfo("DATA")
    col_dmi["NAME"] = colnme
    shape = tbl.getcell("DATA", 0).shape
    tbl.addcols(
        maketabdesc(
            makearrcoldesc(colnme, 0.0 + 0.0j, valuetype="complex", shape=shape)
        ),
        col_dmi,
        addtoparent=True,
    )


def get_lmns(tbl, ra_rad, dec_rad, phase_center_shift=0):
    """
    Calculating l, m, n values from ras,decs and phase centre.
    Parameters
    ----------
    tbl : casacore table.
        The casacore MS table.
    ra_rad : array
        Right ascensions in radians
    dec_rad : [type]
        Declinations in radians
    phase_center_shift : int, optional
        Dont use this!, by default 0
    Returns
    -------
    tuple(array, array, array)
        ls, ms, ns
    """
    ra0, dec0 = get_phase_center(tbl)

    if phase_center_shift != 0:
        #print(f"shifting pahse center from {np.rad2deg(ra0)},{np.rad2deg(dec0)} \
        #    to {np.rad2deg(ra0)+phase_center_shift}, {np.rad2deg(dec0)+phase_center_shift} for testing.\
        #    Will ruin stuff!!!!!")
        ra0 += np.deg2rad(phase_center_shift)
        dec0 += np.deg2rad(phase_center_shift)

    ra_delta = ra_rad - ra0
    ls = np.cos(dec_rad) * np.sin(ra_delta)
    ms = np.sin(dec_rad) * np.cos(dec0) - np.cos(dec_rad) * np.sin(dec0) * np.cos(
        ra_delta
    )
    ns = np.sqrt(1 - ls ** 2 - ms ** 2) - 1

    return ls, ms, ns


def get_bl_lens(mset):
    """Calculate the baseline length for each DATA row in the measurement set
    Parameters
    ----------
    mset : string
        Path to measurement set
    Returns
    -------
    array
        baseline vectors
    """
    t = table(mset + "/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()

    tt = table(mset)
    ant1 = tt.getcol("ANTENNA1")
    ant2 = tt.getcol("ANTENNA2")
    tt.close()

    bls = np.zeros(len(ant1))
    for i in range(len(ant1)):
        p = ant1[i]
        q = ant2[i]
        pos1, pos2 = pos[p], pos[q]
        bls[i] = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    return bls


def get_bl_vectors(mset, refant=0):
    """
    Gets the antenna XYZ position coordinates and recalculates them with the reference antenna as the origin.
    Parameters
    ----------
    mset : Measurement set. \n
    refant : int, optional
        The reference antenna ID, by default 0. \n
    Returns
    -------
    array
        XYZ coordinates of each antenna with respect to the reference antenna.
    """
    # First get the positions of each antenna recorded in XYZ coordinates from the MS
    t = table(mset + "/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()

    no_ants = len(pos)
    print("The mset has %s antennas." % (no_ants))

    bls = np.zeros((no_ants, 3))
    for i in range(no_ants):  # calculate and fill bls with distances from the refant
        pos1, pos2 = pos[i], pos[refant]
        bls[i] = np.array([pos2 - pos1])
    return bls


def merge_flagging(ms_ON, ms_OFF, readonly = True):

    ms_table_ON = table(ms_ON,readonly=readonly)
    ms_table_OFF = table(ms_OFF,readonly=readonly)

    on_flag = get_flags(ms_table_ON)
    off_flag = get_flags(ms_table_OFF)

    indx_ON = np.argwhere(on_flag==True)
    indx_OFF = np.argwhere(off_flag==True)

    for i in range(len(indx_ON)):
        off_flag[indx_ON[i][0]][indx_ON[i][1]][indx_ON[i][2]] = True

    for i in range(len(indx_OFF)):
        on_flag[indx_OFF[i][0]][indx_OFF[i][1]][indx_OFF[i][2]] = True
    
    ms_table_ON.putcol('FLAG', on_flag,)
    ms_table_OFF.putcol('FLAG', off_flag,)

    ms_table_ON.flush()
    ms_table_OFF.flush()
    print('Done merging flags!')


ON_Moon = sys.argv[1]
OFF_Moon = sys.argv[2]
merge_flagging(ms_ON = ON_Moon, ms_OFF = OFF_Moon, readonly = False)