import os

import numpy as np
import xarray as xr
from src.metrics.spectral import conn_spec_average
from src.session import session
from tqdm import tqdm


## Functions
def load_session_data(sid, monkey, align):
    # Instantiate class
    ses = session(
        raw_path=os.path.expanduser("/home/INT/lima.v/data/GrayLab/"),
        monkey=monkey,
        date=sid,
        session=1,
        slvr_msmod=False,
        only_unique_recordings=False,
        align_to=align,
        evt_dt=[-0.65, 2.0],
    )

    # Read data from .mat files
    ses.read_from_mat()

    # Load XYZ coordinates
    coords = np.concatenate(
        (ses.get_xy_coords(), ses.recording_info["depth"][:, None]), axis=1
    )

    # Filtering by trials
    data_task = ses.filter_trials(trial_type=[1], behavioral_response=[1])
    data_fixation = ses.filter_trials(trial_type=[2], behavioral_response=None)

    attrs_task, attrs_fixation = data_task.attrs, data_fixation.attrs

    stim = np.hstack((attrs_task["stim"], attrs_fixation["stim"]))
    t_cue_on = np.hstack((attrs_task["t_cue_on"], attrs_fixation["t_cue_on"]))
    t_cue_off = np.hstack((attrs_task["t_cue_off"], attrs_fixation["t_cue_off"]))
    t_match_on = np.hstack((attrs_task["t_match_on"], attrs_fixation["t_match_on"]))

    np.nan_to_num(stim, nan=6, copy=False)

    data = xr.concat((data_task, data_fixation), "trials")
    data.attrs = attrs_task
    data.attrs["stim"] = stim
    data.attrs["t_cue_on"] = t_cue_on
    data.attrs["t_cue_off"] = t_cue_off
    data.attrs["t_match_on"] = t_match_on
    data.attrs["x"] = coords[:, 0]
    data.attrs["y"] = coords[:, 1]
    data.attrs["z"] = coords[:, 2]

    # ROIs with channels
    rois = [
        f"{roi}_{channel}" for roi, channel in zip(data.roi.data, data.channels_labels)
    ]
    data = data.assign_coords({"roi": rois})
    # data.attrs = attrs
    data.values *= 1e6

    # return node_xr_remove_sca(data)
    return data


def create_generalized_surrogates(X, n_boot=10000):

    T, R, N = X.shape  # Extract dimensions

    # Trials onsets
    t_match_on = (data.attrs["t_match_on"] - data.attrs["t_cue_on"]) / data.fsample
    t_match_on = np.round(t_match_on, 1)

    X = X.values

    # Sample trials
    count = 0
    trials_surr = []
    while count < n_boot:
        out = np.random.choice(
            range(T),
            size=2,
            replace=False,
        )
        trials_surr += [out]
        count = count + 1
    trials_surr = np.stack(trials_surr)

    # Sample channels
    count = 0
    channels_surr = []
    while count < n_boot:
        out = np.random.choice(
            range(R),
            size=2,
            replace=False,
        )
        channels_surr += [out]
        count = count + 1
    channels_surr = np.stack(channels_surr)

    data_surr = []
    for c_i, c_j, trial_i, trial_j in np.concatenate(
        (channels_surr, trials_surr), axis=1
    ):
        x = X[trial_i, c_i]
        y = X[trial_j, c_j]
        data_surr += [np.stack((x, y))]

    # Get shortest onset time for each pair of sampled trials
    onsets = []
    for t1, t2 in trials_surr:
        onsets += [min(t_match_on[t1], t_match_on[t2])]
    onsets = np.stack(onsets)

    return np.stack(data_surr), onsets


## Load data

data = load_session_data("141017", "lucy", "cue")
rois = data.roi.values
s, t = "F1_63", "F1_83"
roi_idx = np.logical_or(rois == s, rois == t)
data = data.isel(roi=roi_idx)
attrs = data.attrs
task = data.stim < 6
data = data.isel(trials=task)
n_trials, n_roi, n_times = data.shape

n_boot = 1000
n_trials = data.sizes["trials"]


def _loop():

    data_surr, t_match_on = create_generalized_surrogates(data.copy(), n_boot=n_trials)

    data_surr = xr.DataArray(
        data_surr,
        dims=("trials", "roi", "time"),
        coords={"time": data.time.values},
    )

    data_surr.attrs = attrs

    epoch_data = []

    for i in range(data_surr.sizes["trials"]):
        stages = [
            [-0.4, 0.0],
            [0, 0.4],
            [0.5, 0.9],
            [0.9, 1.3],
            [t_match_on[i] - 0.4, t_match_on[i]],
        ]

        temp = []

        for t_i, t_f in stages:
            temp += [data_surr[i].sel(time=slice(t_i, t_f)).data]

        epoch_data += [np.stack(temp, axis=-2)]

    epoch_data = xr.DataArray(
        np.stack(epoch_data),
        dims=("trials", "roi", "epochs", "time"),
        coords={
            "trials": data_surr.trials,
            "roi": ["x", "y"],
        },
        attrs=data_surr.attrs,
    )

    coh = []
    for i in range(epoch_data.sizes["epochs"]):
        coh += [
            conn_spec_average(
                epoch_data.sel(epochs=i),
                sfreq=data.attrs["fsample"],
                fmin=0.1,
                fmax=80,
                roi="roi",
                n_jobs=1,
                bandwidth=5,
                verbose=False,
            )
        ]

    coh = xr.concat(coh, "epochs")
    coh.attrs = attrs

    del data_surr, epoch_data
    return coh.squeeze()


out = [_loop() for _ in tqdm(range(n_boot))]
out = xr.concat(out, "boot")
out.to_netcdf(f"coh_surr_{s}_{t}.nc")
