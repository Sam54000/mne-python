"""Reading tools from BCI2000 .dat files."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import re
import struct
from datetime import datetime, timezone

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ...annotations import Annotations
from ...channels.montage import make_standard_montage
from ...utils import fill_doc, logger, verbose, warn
from ..base import BaseRaw

CH_TYPE_MAPPING = {
    "EEG": FIFF.FIFFV_EEG_CH,
    "SEEG": FIFF.FIFFV_SEEG_CH,
    "ECOG": FIFF.FIFFV_ECOG_CH,
    "DBS": FIFF.FIFFV_DBS_CH,
    "EOG": FIFF.FIFFV_EOG_CH,
    "ECG": FIFF.FIFFV_ECG_CH,
    "EMG": FIFF.FIFFV_EMG_CH,
    "BIO": FIFF.FIFFV_BIO_CH,
    "RESP": FIFF.FIFFV_RESP_CH,
    "TEMP": FIFF.FIFFV_TEMPERATURE_CH,
    "MISC": FIFF.FIFFV_MISC_CH,
    "STIM": FIFF.FIFFV_STIM_CH,
}


@fill_doc
class RawBCI2K(BaseRaw):
    """Raw object from BCI2000 .dat file.

    Parameters
    ----------
    input_fname : path-like
        Path to the BCI2000 .dat file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : ``'auto'`` | str | list of str | int | list of int
        Defaults to ``'auto'``, which means that channels named ``'status'`` or
        ``'trigger'`` (case insensitive) are set to STIM. If str (or list of
        str), all channels matching the name(s) are set to STIM. If int (or
        list of ints), the channels corresponding to the indices are set to
        STIM.
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    include : list of str | str
        Channel names to be included. A str is interpreted as a regular
        expression. 'exclude' must be empty if include is assigned.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    mne.io.read_raw_bci2k : Recommended way to read BCI2000 .dat files.

    Notes
    -----
    BCI2000 recordings typically store event markers in the 'State' channels.
    These are automatically detected and can be accessed via
    :func:`mne.events_from_annotations`.
    """

    @verbose
    def __init__(
        self,
        input_fname,
        eog=None,
        misc=None,
        stim_channel="auto",
        exclude=(),
        include=None,
        preload=False,
        *,
        verbose=None,
    ):
        logger.info(f"Extracting BCI2000 parameters from {input_fname}...")
        input_fname = os.path.abspath(input_fname)

        info, bci2k_info, orig_units = _get_info(
            input_fname,
            stim_channel,
            eog,
            misc,
            exclude,
            include,
        )

        logger.info("Creating raw.info structure...")
        last_samps = [bci2k_info["nsamples"] - 1]

        super().__init__(
            info,
            preload,
            filenames=[input_fname],
            raw_extras=[bci2k_info],
            last_samps=last_samps,
            orig_format="int",
            orig_units=orig_units,
            verbose=verbose,
        )

        if len(bci2k_info["state_names"]) > 0:
            logger.info("Reading state channel data for annotations...")
            statevec_len = bci2k_info["statevec_len"]
            n_samples = bci2k_info["nsamples"]

            bci2k_info["raw_states"] = np.zeros(
                (statevec_len, n_samples), dtype=np.uint8
            )
            bci2k_info["collecting_raw_states"] = True

            with open(input_fname, "rb") as fid:
                fid.seek(bci2k_info["header_len"])

                bytes_per_frame = bci2k_info["bytes_per_frame"]
                bytes_per_channel = bci2k_info["bytes_per_channel"]
                n_source_ch = bci2k_info["orig_source_ch"]
                signal_bytes = n_source_ch * bytes_per_channel

                for i in range(n_samples):
                    frame_data = fid.read(bytes_per_frame)
                    if len(frame_data) < bytes_per_frame:
                        break
                    state_data = frame_data[signal_bytes:]
                    if len(state_data) == statevec_len:
                        bci2k_info["raw_states"][:, i] = np.frombuffer(
                            state_data, dtype=np.uint8
                        )

            bci2k_info["collecting_raw_states"] = False

            onset, duration, desc = _get_annotations_bci2k(
                bci2k_info, self.info["sfreq"]
            )

            self.set_annotations(
                Annotations(
                    onset=onset, duration=duration, description=desc, orig_time=None
                )
            )

    def _read_segment_file(
        self,
        data: np.ndarray,
        idx: np.ndarray,
        fi: int,
        start: int,
        stop: int,
        cals: np.ndarray,
        mult: np.ndarray,
    ):
        """Read a chunk of raw data."""
        return _read_segment_file(
            data,
            idx,
            fi,
            start,
            stop,
            self._raw_extras[fi],
            self.filenames[fi],
            cals,
            mult,
        )


def _read_segment_file(
    data: np.ndarray,
    idx: np.ndarray,
    fi: int,
    start: int,
    stop: int,
    bci2k_info: dict,
    fname: str,
    cals: np.ndarray,
    mult: np.ndarray,
    verbose: bool = False,
):
    """Read a chunk of raw data."""
    if isinstance(idx, slice):
        idx = np.arange(bci2k_info["source_ch"])[idx]

    n_channels = len(idx)
    n_samples = stop - start

    with open(fname, "rb") as fid:
        fid.seek(bci2k_info["header_len"])
        offset = start * bci2k_info["bytes_per_frame"]
        fid.seek(offset, 1)
        n_bytes = n_samples * bci2k_info["bytes_per_frame"]
        raw_data = fid.read(n_bytes)

    bytes_per_frame = bci2k_info["bytes_per_frame"]
    temp_buffer = np.zeros((n_channels, n_samples), dtype=np.float64)

    for i in range(n_samples):
        start_byte = i * bytes_per_frame
        end_byte = start_byte + bytes_per_frame
        frame_data = raw_data[start_byte:end_byte]
        signal_bytes = bci2k_info["source_ch"] * bci2k_info["bytes_per_channel"]
        signal_data = frame_data[:signal_bytes]
        state_data = frame_data[signal_bytes:]

        if (
            hasattr(bci2k_info, "collecting_raw_states")
            and bci2k_info["collecting_raw_states"]
        ):
            bci2k_info["raw_states"][:, start + i] = np.frombuffer(
                state_data, dtype=np.uint8
            )

        data_format = bci2k_info["data_format"]
        fmt_map = {"int16": "h", "int32": "l", "float32": "f"}
        ch_fmt = fmt_map.get(data_format, "h")
        signal_fmt = ch_fmt * bci2k_info["source_ch"]
        values = struct.unpack(signal_fmt, signal_data)

        for j, ch_idx in enumerate(idx):
            if ch_idx < len(values):
                temp_buffer[j, i] = values[ch_idx]

    if mult is not None:
        data[:] = mult @ temp_buffer
    else:
        if cals is not None:
            cals = np.array([bci2k_info["cals"][i] for i in idx])
            temp_buffer *= cals[:, np.newaxis]
        data[:] = temp_buffer

    return data


def _decode_units(value_str):
    """Decode units from BCI2000 parameter strings.

    Parameters
    ----------
    value_str : str
        The string value from BCI2000 parameters, e.g., "10muV" or "0.01V"

    Returns
    -------
    value : float
        The numeric value
    unit : str
        The unit string (e.g., 'V', 'µV')
    scale : float
        The scaling factor to apply
    """
    value_str = str(value_str)

    units = ""
    s = value_str
    while len(s) and s[-1] not in "0123456789.":
        units = s[-1] + units
        s = s[:-1]

    if len(s) == 0:
        return None, "V", 1.0

    try:
        value = float(s)
    except ValueError:
        return None, "V", 1.0

    units_lower = units.lower()
    if units_lower == "muv" or units_lower == "µv":
        return value, "µV", 1.0
    elif units_lower == "mv":
        return value, "mV", 1e3
    elif units_lower == "v":
        return value, "V", 1e6
    else:
        return value, "V", 1.0


def _get_info(fname, stim_channel, eog, misc, exclude, include):
    """Extract info from BCI2000 .dat file."""
    bci2k_info = _read_bci2k_header(fname, exclude, include)
    info = _empty_info(bci2k_info["sfreq"])
    ch_names = bci2k_info["ch_names"]
    ch_types = ["eeg"] * len(ch_names)

    stim_ch_idxs = _check_stim_channel(stim_channel, ch_names)

    for idx in stim_ch_idxs:
        ch_types[idx] = "stim"

    if eog is not None:
        eog_ch_idxs = _check_channel_list(eog, ch_names)
        for idx in eog_ch_idxs:
            ch_types[idx] = "eog"

    if misc is not None:
        misc_ch_idxs = _check_channel_list(misc, ch_names)
        for idx in misc_ch_idxs:
            ch_types[idx] = "misc"

    params = bci2k_info["params"]
    unit_scale = np.ones(len(ch_names))
    orig_units = {}

    source_ch_gain = params.get("SourceChGain", [])
    if len(source_ch_gain) == len(ch_names):
        for ch_name in ch_names:
            orig_units[ch_name] = "V"

        for i, (ch_name, gain_str) in enumerate(zip(ch_names, source_ch_gain)):
            value, unit, scale = _decode_units(gain_str)
            if value is not None:
                unit_scale[i] = scale
                orig_units[ch_name] = unit
    else:
        with open(fname, "rb") as fid:
            fid.seek(bci2k_info["header_len"])
            n_samples = int(bci2k_info["sfreq"])
            max_start = bci2k_info["nsamples"] - n_samples

            if max_start <= 0:
                n_samples = bci2k_info["nsamples"]
                start_sample = 0
            else:
                start_sample = np.random.randint(0, max_start)

            logger.info(
                f"Reading {n_samples} samples (1 second) starting at sample"
                f"{start_sample}"
            )

            fid.seek(
                bci2k_info["header_len"] + start_sample * bci2k_info["bytes_per_frame"]
            )

            n_bytes = n_samples * bci2k_info["bytes_per_frame"]
            raw_data = fid.read(n_bytes)
            values = []
            bytes_per_frame = bci2k_info["bytes_per_frame"]

            for i in range(n_samples):
                start_byte = i * bytes_per_frame
                end_byte = start_byte + bytes_per_frame
                frame_data = raw_data[start_byte:end_byte]
                signal_bytes = bci2k_info["source_ch"] * bci2k_info["bytes_per_channel"]
                signal_data = frame_data[:signal_bytes]
                data_format = bci2k_info["data_format"]
                fmt_map = {"int16": "h", "int32": "l", "float32": "f"}
                ch_fmt = fmt_map.get(data_format, "h")
                signal_fmt = ch_fmt * bci2k_info["source_ch"]
                values.extend(struct.unpack(signal_fmt, signal_data))

            values = np.array(values).reshape(-1, len(ch_names))

            orig_units = {}
            for ch_name in ch_names:
                orig_units[ch_name] = "V"

            for i, (ch_name, ch_type) in enumerate(zip(ch_names, ch_types)):
                if ch_type == "eeg":
                    ch_data = values[:, i]
                    data_range = np.abs(ch_data).mean()
                    if data_range > 1:
                        unit_scale[i] = 1e-6
                        orig_units[ch_name] = "µV"

    bci2k_info["cals"] *= unit_scale

    for idx, (ch_name, ch_type) in enumerate(zip(ch_names, ch_types)):
        info["chs"].append(
            {
                "cal": bci2k_info["cals"][idx],
                "ch_name": ch_name,
                "coil_type": FIFF.FIFFV_COIL_EEG,
                "kind": CH_TYPE_MAPPING.get(ch_type.upper(), FIFF.FIFFV_EEG_CH),
                "loc": np.zeros(12),
                "logno": idx + 1,
                "range": 1.0,
                "scanno": idx + 1,
                "unit": FIFF.FIFF_UNIT_V,
                "unit_mul": 0,
            }
        )

    info["nchan"] = len(ch_names)
    info["ch_names"] = ch_names
    info["sfreq"] = bci2k_info["sfreq"]

    if bci2k_info["meas_date"] is not None:
        info["meas_date"] = bci2k_info["meas_date"]

    try:
        montage = make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False)
    except Exception:
        logger.info("Could not set standard montage.")

    return info, bci2k_info, orig_units


def _read_bci2k_header(fname, exclude, include):
    """Read BCI2000 header information from .dat file."""
    bci2k_info = {}

    with open(fname, "rb") as fid:
        line = fid.readline().decode("utf-8").strip().split()

        header_info = {}
        for i in range(0, len(line), 2):
            if i + 1 < len(line):
                key = line[i].rstrip("=")
                value = line[i + 1]
                header_info[key] = value

        header_len = int(header_info.get("HeaderLen", 0))
        source_ch = int(header_info.get("SourceCh", 0))
        statevec_len = int(header_info.get("StatevectorLen", 0))
        data_format = header_info.get("DataFormat", "int16")

        fmt_map = {"int16": "h", "int32": "l", "float32": "f"}
        fmt = fmt_map.get(data_format, "h")
        bytes_per_channel = struct.calcsize(fmt)

        frame_fmt = fmt * source_ch + "B" * statevec_len
        unpack_sig = fmt * source_ch + "x" * statevec_len
        unpack_states = "x" * bytes_per_channel * source_ch + "B" * statevec_len

        bytes_per_frame = source_ch * bytes_per_channel + statevec_len

        line = fid.readline().decode("utf-8").strip()
        if line != "[ State Vector Definition ]":
            raise ValueError("Failed to find state vector definition section")

        state_defs = {}
        state_names = []
        while True:
            line = fid.readline().decode("utf-8").strip()
            if not line or line.startswith("["):
                break

            parts = line.split()
            if len(parts) >= 5:
                state_name = parts[0]
                state_names.append(state_name)
                state_defs[state_name] = {
                    "length": int(parts[1]),
                    "start_val": int(parts[2]),
                    "byte_pos": int(parts[3]),
                    "bit_pos": int(parts[4]),
                }

        if line != "[ Parameter Definition ]":
            raise ValueError("Failed to find parameter definition section")

        params = {}
        while fid.tell() < header_len:
            line = fid.readline().decode("utf-8").strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 3:
                param_name = parts[2].rstrip("=")
                param_value = parts[3] if len(parts) > 3 else ""
                params[param_name] = param_value

        sfreq = float(params.get("SamplingRate", "0").rstrip("Hz"))

        ch_names = []
        if "ChannelNames" in params:
            ch_names = params["ChannelNames"].split()
        else:
            ch_names = [f"Ch{i + 1}" for i in range(source_ch)]

        orig_source_ch = source_ch
        orig_ch_names = ch_names.copy()

        if include is not None:
            if isinstance(include, str):
                include = [include]
            include_idx = []
            for pattern in include:
                pattern = re.compile(pattern)
                include_idx.extend(
                    [i for i, ch in enumerate(ch_names) if pattern.match(ch)]
                )
            include_idx = sorted(set(include_idx))
            if len(include_idx) == 0:
                raise ValueError("No channels match the inclusion pattern.")
            ch_names = [ch_names[i] for i in include_idx]
        elif exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            exclude_idx = []
            for pattern in exclude:
                pattern = re.compile(pattern)
                exclude_idx.extend(
                    [i for i, ch in enumerate(ch_names) if pattern.match(ch)]
                )
            exclude_idx = sorted(set(exclude_idx))
            ch_names = [ch for i, ch in enumerate(ch_names) if i not in exclude_idx]

        meas_date = None
        if "StorageTime" in params:
            try:
                date_str = params["StorageTime"]
                for fmt in ["%a %b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        meas_date = datetime.strptime(date_str, fmt)
                        meas_date = meas_date.replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
            except Exception:
                logger.warning("Could not parse measurement date.")

        fid.seek(0, 2)
        file_size = fid.tell()
        data_size = file_size - header_len
        n_samples = data_size // bytes_per_frame

        cals = np.ones(len(ch_names))

        source_ch_gain = []
        if "SourceChGain" in params:
            source_ch_gain = params["SourceChGain"].split()
            if include is not None:
                source_ch_gain = [source_ch_gain[i] for i in include_idx]
            elif exclude:
                source_ch_gain = [
                    gain
                    for i, gain in enumerate(source_ch_gain)
                    if i not in exclude_idx
                ]

        bci2k_info = {
            "header_len": header_len,
            "source_ch": len(ch_names),
            "orig_source_ch": orig_source_ch,
            "statevec_len": statevec_len,
            "data_format": data_format,
            "bytes_per_channel": bytes_per_channel,
            "frame_fmt": frame_fmt,
            "unpack_sig": unpack_sig,
            "unpack_states": unpack_states,
            "bytes_per_frame": bytes_per_frame,
            "state_defs": state_defs,
            "state_names": state_names,
            "params": params,
            "sfreq": sfreq,
            "ch_names": ch_names,
            "orig_ch_names": orig_ch_names,
            "nsamples": n_samples,
            "meas_date": meas_date,
            "cals": cals,
            "source_ch_gain": source_ch_gain,
        }

    return bci2k_info


def _check_stim_channel(stim_channel, ch_names):
    """Check stim channel selection."""
    if stim_channel is None:
        return []

    if isinstance(stim_channel, str):
        if stim_channel == "auto":
            stim_channel = []
            for i, ch in enumerate(ch_names):
                if ch.lower() in ["status", "trigger"]:
                    stim_channel.append(i)
        else:
            stim_channel = [i for i, ch in enumerate(ch_names) if ch == stim_channel]
    elif isinstance(stim_channel, int):
        stim_channel = [stim_channel]
    elif isinstance(stim_channel, list | tuple):
        if all(isinstance(item, str) for item in stim_channel):
            stim_channel = [i for i, ch in enumerate(ch_names) if ch in stim_channel]
        elif all(isinstance(item, int) for item in stim_channel):
            stim_channel = list(stim_channel)
        else:
            raise ValueError("stim_channel must be all str or all int")
    else:
        raise ValueError("stim_channel must be str, int, list, or None")

    return stim_channel


def _check_channel_list(ch_list, ch_names):
    """Check if channels exist and return indices."""
    if isinstance(ch_list, str):
        ch_list = [ch_list]

    ch_indices = []
    for ch in ch_list:
        if isinstance(ch, str):
            if ch in ch_names:
                ch_indices.append(ch_names.index(ch))
            else:
                warn(f"Channel {ch} not found.")
        elif isinstance(ch, int):
            if 0 <= ch < len(ch_names):
                ch_indices.append(ch)
            else:
                warn(f"Channel index {ch} out of range.")

    return ch_indices


def _get_annotations_bci2k(bci2k_info, sfreq):
    """Extract annotations from BCI2000 state channels.

    This function extracts state changes from BCI2000 state channels and
    converts them into MNE annotations. The approach is based on how
    the BCpy2000 project handles BCI2000 state data.

    Parameters
    ----------
    bci2k_info : dict
        Dictionary with BCI2000 file information.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    onset : array
        Annotation onset times in seconds.
    duration : array
        Annotation durations in seconds.
    description : array of str
        Annotation descriptions with format "StateChannelName=Value".
    """
    state_defs = bci2k_info["state_defs"]
    state_names = bci2k_info["state_names"]

    if not state_names:
        return np.array([]), np.array([]), np.array([], dtype=str)

    raw_states = bci2k_info["raw_states"]

    events = []
    descriptions = []

    for state_name in state_names:
        if "Time" in state_name:
            continue

        state_def = state_defs[state_name]

        byte_pos = state_def["byte_pos"]
        bit_pos = state_def["bit_pos"]
        n_bits = state_def["length"]

        n_bytes = int((bit_pos + n_bits + 7) / 8)
        byte_slice = slice(byte_pos, byte_pos + n_bytes)

        masks = np.ones(n_bytes, dtype=np.uint8) * 255

        start_mask = 255 & (255 << bit_pos)
        masks[0] &= start_mask

        extra_bits = n_bytes * 8 - n_bits - bit_pos
        if extra_bits > 0:
            end_mask = 255 & (255 >> extra_bits)
            masks[-1] &= end_mask

        state_bytes = raw_states[byte_slice, :]

        for i in range(n_bytes):
            state_bytes[i, :] &= masks[i]

        multiplier = np.array(
            [256**i / (1 << bit_pos) for i in range(n_bytes)], dtype=np.float64
        )
        state_values = np.sum(state_bytes * multiplier[:, np.newaxis], axis=0)

        state_changes = np.diff(np.concatenate([[0], state_values]))
        change_indices = np.where(state_changes != 0)[0]

        for idx in change_indices:
            value = int(state_values[idx])
            if value > 0:
                events.append(idx)
                descriptions.append(f"{state_name}={value}")

    if events:
        events = np.array(events)
        descriptions = np.array(descriptions)

        sort_indices = np.argsort(events)
        events = events[sort_indices]
        descriptions = descriptions[sort_indices]

        onset = events / sfreq

        duration = np.zeros_like(onset)

        return onset, duration, descriptions
    else:
        return np.array([]), np.array([]), np.array([], dtype=str)


@fill_doc
def read_raw_bci2k(
    input_fname,
    eog=None,
    misc=None,
    stim_channel="auto",
    exclude=(),
    include=None,
    preload=False,
    *,
    verbose=None,
) -> RawBCI2K:
    """Read BCI2000 .dat file.

    Parameters
    ----------
    input_fname : path-like
        Path to the BCI2000 .dat file.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    stim_channel : ``'auto'`` | str | list of str | int | list of int
        Defaults to ``'auto'``, which means that channels named ``'status'`` or
        ``'trigger'`` (case insensitive) are set to STIM. If str (or list of
        str), all channels matching the name(s) are set to STIM. If int (or
        list of ints), the channels corresponding to the indices are set to
        STIM.
    exclude : list of str
        Channel names to exclude. This can help when reading data with
        different sampling rates to avoid unnecessary resampling.
    include : list of str | str
        Channel names to be included. A str is interpreted as a regular
        expression. 'exclude' must be empty if include is assigned.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawBCI2K
        A Raw object containing BCI2000 data.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """
    return RawBCI2K(
        input_fname=input_fname,
        eog=eog,
        misc=misc,
        stim_channel=stim_channel,
        exclude=exclude,
        include=include,
        preload=preload,
        verbose=verbose,
    )
