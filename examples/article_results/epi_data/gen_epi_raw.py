"""Sample file to make the raw data file generation process semi-visible, the original file is not available to the public

1. read in TRC file (Original EEG record)
2. get seizure and control time-frames
3. cut out chunks of LFP data
4. save out the raw chunks into bz2-compressed pickle file

"""
import numpy as np
import neo
from tqdm import tqdm

import pickle
import bz2


def sec_2_index(secs, fr=2048, offset=0):
    """

    :param numpy.ndarray secs: time in seconds to be converted
    :param float fr: dampling frequency
    :param float offset: offset in seconds
    :return: time in samples
    :rtype: numpy.ndarray
    """
    return (secs*fr - offset*fr).astype(int)


fn = "/home/phrenico/Projects/Data/MeMo/EEG_2800.TRC"
reader = neo.io.MicromedIO(fn)
channels = np.array([reader.header['signal_channels'][i][0] for i in range(106)])
fr = reader.get_signal_sampling_rate()
data = reader.get_analogsignal_chunk()


# get seizure times
times_sec_control = np.load("control_times.npy")
times_sec_seizure = np.load("seizure_times.npy")


# Start the procedure
indices = list(range(64))  + list(range(65, 89))
L = 10  # length of segment in seconds


control_chunks_dict = {}
for time in tqdm(times_sec_control):
    data_chunk = data[sec_2_index(time):sec_2_index(time+10), :].astype(float)

    control_chunks_dict[time / fr] = data_chunk

seizure_chunk_dict = {}
for time in tqdm(times_sec_seizure):
    data_chunk = data[sec_2_index(time):sec_2_index(time+10), :].astype(float)[:, :]
    seizure_chunk_dict[time / fr] = data_chunk

intermittent_save_path = "./"
# with open(intermittent_save_path+'raw_data_dicts.pkl', 'wb') as f:  # save without bz2 compression
with bz2.BZ2File(intermittent_save_path+'raw_data_dicts.pckl', 'wb') as f:
    pickle.dump({'control': control_chunks_dict,
                 'seizure': seizure_chunk_dict,
                 'channels': channels,
                 'samplefreq': fr}, f)