import mne
from src.utils.utils import CHNAMES_MAPPING, CH_NAMES, get_feat_names, apply_mor_data_hack_fix, chunk, FREQ_BANDS_PH, FREQ_BANDS
import numpy as np
import warnings
from scipy.signal import iirnotch, BadCoefficients
from mne.filter import estimate_ringing_samples
import mne_connectivity
import mne_features
from pyriemann.estimation import Covariances
import glob
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of the log message
    handlers=[
        logging.FileHandler('app.log'),  # Redirect logs to a file
        logging.StreamHandler()           # Also show logs in the console (optional)
    ]
)

DESIRED_FS = 100 # my tak robiliśmy ale można inaczej
CROP_LEN_SAMP = 600 # długość wycinka sygnału w próbkach
MAX_LIM = 800 # odrzucamy wycinki powyżej 800 µV
MIN_LIM = 1 # próg w µV na minimalną wartość miary MAD dla kanału - powoduje odrzucanie płaskich kanałów

class ChannelsError(Exception):
    """Exception raised for errors in the channel renaming or reordering."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def standardize_electrode_names_and_order(edf):
    i = 0
    while i < len(CHNAMES_MAPPING):
        try:
            # Ujednolicamy nazwy elektrod i porządek
            edf = edf.rename_channels(CHNAMES_MAPPING[i])
            edf = edf.reorder_channels(CH_NAMES)
            break
        except ValueError:
            # Jeśli wystąpi błąd, przechodzimy do kolejnej próby
            i += 1

    if i == len(CHNAMES_MAPPING):
        raise ChannelsError(
            f"Channel renaming or reordering failed. Available channels: {edf.info['ch_names']}. "
            f"Required channels: {CH_NAMES}."
        )

    return edf

def create_filter(iir_params, f_pass, f_stop, fs, btype):
    """
    Create an IIR filter with given parameters.

    Arguments:
    - iir_params: Filter parameters dictionary.
    - f_pass: Pass frequency for lowpass/highpass filters.
    - f_stop: Stop frequency for lowpass/highpass filters.
    - fs: Sampling frequency.
    - btype: Filter type ('highpass', 'lowpass')

    Returns:
    - Filter parameters in SOS format.
    """
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            iir_params = mne.filter.construct_iir_filter(iir_params,
                                            f_pass=f_pass,
                                            f_stop=f_stop,
                                            sfreq=fs,
                                            btype=btype,
                                            return_copy=False)
            return iir_params
    except BadCoefficients:
        raise BadCoefficients(f'Bad filter coefficients for {btype} filter')

def create_notch_filter(notch_freq, Q, fs):
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            b, a = iirnotch(w0=notch_freq, Q=Q, fs=fs)
            padlen = estimate_ringing_samples((b, a))
            iir_params = {'output': 'ba', 'b': b, 'a': a, 'padlen': padlen}
            return iir_params
    except BadCoefficients:
        raise BadCoefficients(f'Bad filter coefficients for notch filter')

def get_filters(fs, filter_specs):
    """
    Returns a list of filter parameters in SOS format based on provided specifications.

    Arguments:
    - fs: Sampling frequency.
    - filter_specs: List of dictionaries containing filter parameters.

    Returns:
    - filters: List of filter parameters in SOS format.
    """
    filters = []

    for filter_spec in filter_specs:
        filter_type = filter_spec.get('type', None)

        if filter_type == 'highpass':
            f_stop = filter_spec.get('f_stop', 0.1)
            f_pass = filter_spec.get('f_pass', 0.5)
            iir_params = dict(ftype='butter', output='sos', gpass=1, gstop=20)
            iir_params = create_filter(iir_params, f_pass, f_stop, fs, 'highpass')
            filters.append(iir_params)

        elif filter_type == 'lowpass':
            f_stop = filter_spec.get('f_stop', 50)
            f_pass = filter_spec.get('f_pass', 40)
            iir_params = dict(ftype='butter', output='sos', gpass=1, gstop=20)
            iir_params = create_filter(iir_params, f_pass, f_stop, fs, 'lowpass')
            filters.append(iir_params)

        elif filter_type == 'notch':
            notch_freq = filter_spec.get('notch_freq', 50)
            notch_widths = filter_spec.get('notch_widths', 10)
            Q = notch_freq / notch_widths
            iir_params = create_notch_filter(notch_freq, Q, fs)
            filters.append(iir_params)

        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    return filters

def filter_eeg(edf):
    fs = 250 #edf.info['sfreq']
    filter_specs = [
        {'type': 'highpass', 'f_pass': 0.5, 'f_stop': 0.1},
        {'type': 'lowpass', 'f_pass': 40, 'f_stop': 50},
        {'type': 'notch', 'notch_freq': 50, 'notch_widths': 10}
    ]
    filters = get_filters(fs, filter_specs)
    for filter in filters:
        # l_freq and h_req must be numbers, but filtering will use filter defined in iirparams
        edf = edf.filter(iir_params=filter, l_freq=0, h_freq=0, method='iir')
    return edf

def average_rereference(edf):
    reference = -np.sum(edf._data, axis=0) / (edf._data.shape[0]+1)
    edf._data += reference

    return edf

def preprocess_edf(edf):
    """
    Funkcja preprocess_edf:
    - Ujednolica nazwy elektrod oraz ich kolejność.
    - Jeśli elektrody nie spełniają wymagań (np. brakuje ich wystarczającej liczby lub nie pasują do standardu), funkcja rzuca wyjątek.
    """
    with mne.utils.use_log_level("error"):
        edf = standardize_electrode_names_and_order(edf)
        edf = edf.set_montage("standard_1020", match_case=False)
        edf = filter_eeg(edf)

        #fs = edf.info['sfreq']
        #if int(fs) != int(DESIRED_FS):
        #    edf = edf.resample(sfreq=DESIRED_FS)

        edf_ref = average_rereference(edf.copy())

    return edf, edf_ref

def cut_events(edf, return_idx_list=False):
    data = edf.get_data(units="uV")

    # Calculate the number of crops
    tmax = data.shape[-1]
    num_crops = (tmax // CROP_LEN_SAMP)

    # Initialize arrays
    crops = np.zeros((num_crops, data.shape[0], CROP_LEN_SAMP))
    IDX = np.arange(0, num_crops * CROP_LEN_SAMP, CROP_LEN_SAMP)

    # Slice data into crops
    for i, idx in enumerate(IDX):
        crops[i, :, :] = data[:, idx:idx + CROP_LEN_SAMP]

    if return_idx_list:
        return crops, IDX
    else:
        return crops

def make_selection(crops_with_ref, crops_without_ref, time_crops):
    # Odrzucamy za duże wartości
    reject_max = np.sum(np.abs(crops_with_ref) > MAX_LIM, axis=(-1, -2))

    # Odrzucamy płaskie sygnały
    min_abs_med = np.min(np.median(np.abs(crops_without_ref - np.median(crops_without_ref, axis=-1, keepdims=True)), axis=-1),
                         axis=-1)

    reject_min = min_abs_med < MIN_LIM

    reject_max = np.where(reject_max != 0, 1, 0)

    # Łączymy wyniki
    reject = np.logical_or(reject_min, reject_max)
    not_reject = np.logical_not(reject)
    logging.info(
        f"Total: {len(reject):3d} | Rejected - max: {np.sum(reject_max):3d}, min: {np.sum(reject_min):3d}, total: {np.sum(reject):3d} | Kept: {np.sum(not_reject):3d}")

    crops_with_ref = crops_with_ref[not_reject]
    time_crops = list(np.array(time_crops)[not_reject])

    return crops_with_ref, time_crops

def get_phase(crops):
    mne.set_log_level("CRITICAL")
    n_channels = crops.shape[1]
    len_t = crops.shape[2]

    phase = np.zeros(
        (len(crops), len(FREQ_BANDS_PH), int(n_channels * (n_channels - 1) / 2))
    )

    for i, arr in enumerate(crops):
        n_fragments = len_t // DESIRED_FS
        l_fragment = len_t // n_fragments
        x = np.zeros((n_fragments, n_channels, l_fragment))
        start = 0
        for part in range(n_fragments):
            x[part, :, :] = arr[:, start:start + l_fragment]
            start += l_fragment
        tmp_ph = mne_connectivity.spectral_connectivity_epochs(x, method='coh', sfreq=DESIRED_FS,
                                                               mode='multitaper', faverage=True, # Można się zastanowić czy multitaper
                                                               fmin=FREQ_BANDS_PH[:, 0],
                                                               fmax=FREQ_BANDS_PH[:, 1])
        a = tmp_ph.get_data('dense')  # extract the connectivity matrix values
        for idx_band, band in enumerate(FREQ_BANDS_PH):
            b = a[:, :, idx_band]  # extract connectivity for each band
            # the proper values are in the lower triangular matrix
            phase[i, idx_band, :] = b[np.tril_indices(n_channels, k=-1)]

    phase = phase.reshape(phase.shape[0], phase.shape[1] * phase.shape[2])
    return phase

def get_pow(cut):
    mne.set_log_level("CRITICAL")
    n_channels = cut.shape[1]

    pows = np.zeros((len(cut), n_channels * len(FREQ_BANDS)))

    for i, arr in enumerate(cut):
        P_tmp = mne_features.univariate.compute_pow_freq_bands(
            DESIRED_FS, arr, FREQ_BANDS, normalize=False, psd_method='multitaper')  # JZ normalize= False to return not normalized power
        P_tmp = P_tmp.reshape((n_channels, len(FREQ_BANDS)))
        norm = np.sum(P_tmp)  # JZ normalize given band across electrodes and freqs
        P_tmp = P_tmp / norm
        pows[i] = P_tmp.flatten()

    return pows

def get_covs(cut):
    cov = Covariances()
    covs = cov.fit_transform(cut)
    return covs

def combined_classic_transformation(crops, time_crops, preprocessed_edf):
    if len(crops) > 0:
        phase = np.median(get_phase(crops), axis=0)
        pows = np.median(get_pow(crops), axis=0)
        covs = np.median(get_covs(crops), axis=0)
        covs = covs[np.tril_indices(len(CH_NAMES), k=0)]
        return np.hstack((phase, pows, covs))
    else:
        return None

def get_edf_path(examination_id):
    base_path = "datasets/gemein/raw/gemein/raw/*/*/01_tcp_ar/"
    patient_id, sesion_id, t_id = examination_id.split("_")
    patient_group = patient_id[3:6]
    path_pattern = base_path + f"{patient_group}/{patient_id}/{sesion_id}_*/{examination_id}.edf"
    files = glob.glob(path_pattern)
    return files[0]

def map_edf_to_samples(examination_id, institution_id, edf_dir_pth, data_group):
    mne.set_log_level("CRITICAL")
    if data_group == "ELM19":
        edf_pth = f"{edf_dir_pth}/{examination_id}.edf"
    elif data_group == "TUH":
        edf_pth = get_edf_path(examination_id)

    raw_edf = mne.io.read_raw_edf(edf_pth, preload=True)

    # Szpital MOR ma złe jednostki - trzeba sprowadzać do standardowych
    if institution_id == 'MOR':
        raw_edf = apply_mor_data_hack_fix(raw_edf, edf_pth, institution_id)

    try:
        preprocessed_edf_without_ref, preprocessed_edf = preprocess_edf(raw_edf.copy())
    except ValueError as e:
        print(f"[WARNING] Skipping due to error: {e}")
        return None

    crops, time_crops = cut_events(preprocessed_edf, return_idx_list=True)
    # Potrzebne do wykrycia płaskich sygnałów później
    crops_without_ref = cut_events(preprocessed_edf_without_ref)

    crops, time_crops = make_selection(crops, crops_without_ref, time_crops)

    return combined_classic_transformation(crops, time_crops, preprocessed_edf)

def map_edf_to_samples_with_idx(examination_id, institution_id, idx, edf_dir_pth, data_group="ELM19"):
    return idx, map_edf_to_samples(examination_id, institution_id, edf_dir_pth, data_group)

