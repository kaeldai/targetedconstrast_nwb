import os
import glob
import json
import logging
import pickle
import numpy as np
import h5py
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.image as mpimg

from base import lazy_property


logger = logging.getLogger(__name__)
LOCAL_TIMEZONE = pytz.timezone("US/Pacific")


class OphysSession(object):
    def __init__(self, session_id, lims_reader):
        self.session_id = session_id
        self.imaging_plane_name = 'imaging_plane_1'
        self.calcium_indicator = "GCaMP6f"
        self.plane_segmentation_name = "PlaneSegmentation"
        self.ophys_module_name = "brain_observatory_pipeline"
        self.image_segmentation_name = "ImageSegmentation"

        self._stimulus_table_dir = '/allen/programs/braintv/workgroups/nc-ophys/VisualCoding/stimulus_tables'
        self._events_dir = '/allen/programs/braintv/workgroups/nc-ophys/VisualCoding/events'
        self._running_speeds_dir = '/allen/programs/braintv/workgroups/nc-ophys/VisualCoding/contrast_running_speeds'
        self._lims_reader = lims_reader

        # event times and amplitudes are linked together
        self._event_times = None
        self._event_amps = None
        self._event_indices = None

        # Need to track of all (incl. invalid) ROIs and the order the appear in from objectlist.txt
        self._all_oris_order = None

    @lazy_property
    def description(self):
        return 'OphysSession'

    @lazy_property
    def start_time(self):
        session_start_time = self.stimulus_pkl['start_time']
        return datetime.utcfromtimestamp(session_start_time).astimezone(LOCAL_TIMEZONE)

    @lazy_property
    def stimulus_pkl(self):
        stim_pkl_path = self._lims_reader.get_file(session_id=self.session_id, file_type='StimulusPickle')
        with open(stim_pkl_path, 'rb') as f:
            try:
                return pickle.load(f)
            except UnicodeDecodeError:
                # For pkl files encoded using python 2
                f.seek(0)
                return pickle.load(f, encoding="latin1")

    @lazy_property
    def session_metadata(self):
        session_metadata = self._lims_reader.get_session_metadata(self.session_id)
        session_dict = session_metadata.to_dict()
        storage_dir = session_metadata['storage_directory']
        if not os.path.exists(storage_dir):
            msg = 'session storage_directory {} not found'.format(storage_dir)
            logger.exception(msg)
            raise ValueError(msg)

        # TODO: Check that one and only one row is returned

        # TODO: Check there's only one experiment

        return session_dict

    @lazy_property
    def experiment_metadata(self):
        experiment_metadata = self._lims_reader.get_experiment_metadata(self.session_id)
        experiment_dict = experiment_metadata.to_dict()
        return experiment_dict

    @lazy_property
    def stimulus_table(self):
        stim_table_path = os.path.join(self._stimulus_table_dir, '{}_stim_table.csv'.format(self.session_id))
        stim_table_df = pd.read_csv(stim_table_path, index_col=0)

        stim_table_df = stim_table_df.rename(columns={
            'Start': 'start_time',
            'start': 'start_time',
            'End': 'stop_time',
            'end': 'stop_time',
            'TF': 'temporal_frequency',
            'SF': 'spatial_frequency',
            'Contrast': 'contrast',
            'Ori': 'direction',
            'orientation': 'direction'
        })
        ### Note from Saskia, wanted to remove stimulus_name
        # stim_table_df['stimulus_name'] = 'contrast tuning'

        stim_table_df = stim_table_df.drop('sweep_number', axis=1)
        ### Note from Saskia, Don't use stimulus block
        # if 'stimulus_block' not in stim_table_df:
        #     # Adds the appropiate stimulus block number to each interval. Assign value of 1 to every row where
        #     # 'stimulus_name' changes, then do cumulative sum down the rows
        #     stim_table_df['stimulus_block'] = (
        #             stim_table_df['stimulus_name'] != stim_table_df['stimulus_name'].shift(1)
        #     ).cumsum()
        #     stim_table_df['stimulus_block'] -= 1  # blocks should be 0 indexed

        return stim_table_df

    @lazy_property
    def emission_wavelength(self):
        # TODO: See if this value exists in the database or another known file, <exp_id>.json is not used anywhere else
        experiment_id = self.experiment_metadata['experiment_id']
        experiment_directory = self.experiment_metadata['storage_directory']
        exp_json_path = os.path.join(experiment_directory, '{}.json'.format(experiment_id))
        if not os.path.exists(exp_json_path):
            raise ValueError('Could not find experiment json file {}'.format(exp_json_path))

        with open(exp_json_path, 'r') as fp:
            exp_json = json.load(fp)
            planes = exp_json['planes']
            assert(len(planes) == 1)
            return float(planes[0]['emission-wavelength'])

    @lazy_property
    def excitation_lambda(self):
        # TODO: See if this exists in lims or a file
        return 910.0

    @lazy_property
    def imaging_rate(self):
        # TODO: See if this exists in lims or a file
        return 30.0

    @lazy_property
    def image_height(self):
        return int(self.experiment_metadata['image_height'])

    @lazy_property
    def image_width(self):
        return int(self.experiment_metadata['image_width'])

    @lazy_property
    def wheel_radius(self):
        # TODO: See if this exists in lims or a file
        return 5.5036

    @lazy_property
    def depth(self):
        return self.experiment_metadata["depth"]

    @lazy_property
    def fov(self):
        return 400, 400

    @lazy_property
    def roi_metrics(self):
        roi_metrics_path = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysSegmentationObjects')
        assert(os.path.exists(roi_metrics_path))
        roi_metrics = pd.read_csv(roi_metrics_path)

        tracs_json_path = self._lims_reader.get_file(session_id=self.session_id,
                                                     file_type='OphysExtractedTracesInputJson')
        assert(os.path.exists(tracs_json_path))

        traces_json = json.load(open(tracs_json_path, 'r'))

        # image_height = traces_json["image"]["height"]
        # image_width = traces_json["image"]["width"]

        roi_locations_list = []
        for roi in traces_json['rois']:
            mask = roi["mask"]
            roi_locations_list.append([roi["id"], roi["x"], roi["y"], roi["width"], roi["height"], roi["valid"], mask])

        roi_locations = pd.DataFrame(data=roi_locations_list,
                                     columns=["id", "x", "y", "width", "height", "valid", "mask"])
        roi_names = np.sort(roi_locations.id.values)
        roi_locations["unfiltered_cell_index"] = [np.where(roi_names == rid)[0][0] for rid in roi_locations.id.values]

        specimen_ids = []
        for row in roi_metrics.index:
            minx = roi_metrics.iloc[row][" minx"]
            miny = roi_metrics.iloc[row][" miny"]
            wid = roi_metrics.iloc[row][" maxx"] - minx + 1
            hei = roi_metrics.iloc[row][" maxy"] - miny + 1
            id_vals = roi_locations[
                (roi_locations.x == minx)
                & (roi_locations.y == miny)
                & (roi_locations.width == wid)
                & (roi_locations.height == hei)
            ].id.values
            assert (len(id_vals) == 1)
            specimen_ids.append(id_vals[0])

        roi_metrics["cell_specimen_id"] = specimen_ids
        roi_metrics["id"] = roi_metrics['cell_specimen_id'].values

        roi_metrics = pd.merge(roi_metrics, roi_locations, on="id")

        # Due to an issue that sometimes occurs with l0_true_false events we need to keep of all the ROIs and the
        #  order they appear before we filter out the invalid ones.
        self._all_oris_order = np.array(roi_metrics['id'])
        roi_metrics = roi_metrics[roi_metrics.valid]

        cell_index = [np.where(np.sort(roi_metrics.cell_specimen_id.values) == rid)[0][0]
                      for rid in roi_metrics.cell_specimen_id.values]
        roi_metrics['cell_index'] = cell_index
        return roi_metrics

    @lazy_property
    def roi_masks(self):
        roi_masks = {}
        for i, roi_id in enumerate(self.roi_metrics['cell_specimen_id'].values):
            roi_attrs = self.roi_metrics[self.roi_metrics['id'] == roi_id].iloc[0]
            mask = np.asarray(roi_attrs['mask'])
            binary_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            mask_y = int(roi_attrs['y'])
            mask_x = int(roi_attrs['x'])
            mask_height = int(roi_attrs['height'])
            mask_width = int(roi_attrs['width'])
            binary_mask[mask_y:(mask_y + mask_height), mask_x:(mask_x + mask_width)] = mask
            roi_masks[int(roi_id)] = binary_mask

        return roi_masks

    @property
    def all_rois(self):
        if self._all_oris_order is None:
            _ = self.roi_metrics

        return self._all_oris_order

    @lazy_property
    def neuropil_r(self):
        neuropil_corr_path = self._lims_reader.get_file(session_id=self.session_id, file_type='NeuropilCorrection')
        neuropil_corr_df = h5py.File(neuropil_corr_path, 'r')
        neuropil_r = neuropil_corr_df['r'][()]
        return neuropil_r[self.valid_roi_indices]

    @lazy_property
    def valid_roi_indices(self):
        # WARNING: the index
        return np.sort(self.roi_metrics['unfiltered_cell_index'].values)

    @lazy_property
    def max_projection(self):
        max_intensity_png = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysMaxIntImage')
        return mpimg.imread(max_intensity_png)

    @lazy_property
    def time_sync(self):
        time_sync_path = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysTimeSynchronization')
        return h5py.File(time_sync_path, 'r')

    @lazy_property
    def twop_timestamps(self):
        return np.array(self.time_sync['twop_vsync_fall'], dtype=np.float64)

    @lazy_property
    def raw_traces(self):
        raw_traces_path = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysRoiTraces')
        raw_traces_h5 = h5py.File(raw_traces_path, 'r')
        raw_traces = raw_traces_h5['data'][()]
        return raw_traces[self.valid_roi_indices, :]

    @lazy_property
    def neuropil_traces(self):
        neuropil_traces_path = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysNeuropilTraces')
        neuropil_traces_h5 = h5py.File(neuropil_traces_path, 'r')
        neuropil_traces = neuropil_traces_h5['data'][()]
        return neuropil_traces[self.valid_roi_indices, :]

    @lazy_property
    def demixed_traces(self):
        demixed_traces_path = self._lims_reader.get_file(session_id=self.session_id, file_type='DemixedTracesFile')
        demixed_traces_h5 = h5py.File(demixed_traces_path, 'r')
        demixed_traces = demixed_traces_h5['data'][()]
        return demixed_traces[self.valid_roi_indices, :]

    @lazy_property
    def dff_traces(self):
        dff_traces_path = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysDffTraceFile')
        dff_h5 = h5py.File(dff_traces_path, 'r')
        dff_traces = dff_h5['data'][()]
        return dff_traces[self.valid_roi_indices, :]

    @lazy_property
    def running_velocity(self):
        ##  dx_file = os.path.join(self._running_speeds_dir, '{}_running_speed.npy'.format(self.session_id))
        ## if not os.path.exists(dx_file):
        ##     raise FileNotFoundError('Could not find running_speeds file {}'.format(running_speed_file))
        ## dx = np.load(dx_file)

        dx = self.stimulus_pkl['items']['foraging']['encoders'][0]['dx']
        vsnync_intervals = self.stimulus_pkl['intervalsms']
        while len(vsnync_intervals) < len(dx):
            vsnync_intervals = np.insert(vsnync_intervals, 0, vsnync_intervals[0])
        vsnync_intervals /= 1000
        if len(dx) == 0:
            logger.exception('No running data')
        dxcm = ((dx / 360)*5.5036*np.pi*2) / vsnync_intervals

        twop_frames = self.time_sync['stimulus_alignment'][()]
        start = np.nanmin(twop_frames)
        endframe = int(np.nanmax(twop_frames) + 1)
        dxds = np.empty((endframe, 1))

        for i in range(endframe):
            try:
                temp = np.where(twop_frames == i)[0]
                dxds[i] = np.mean(dxcm[temp[0]:temp[-1]+1])
                if np.isinf(dxds[i]):
                    dxds[i] = 0
            except:
                if i < start:
                    dxds[i] = np.NaN
                else:
                    dxds[i] = dxds[i-1]

        dxds = dxds.flatten()

        # If running speeds array is smaller than the traces pad the end with NaNs
        padding_len = self.dff_traces.shape[1] - dxds.shape[0]
        if padding_len < 0:
            raise ValueError('Session {} running speed is greater than the dff_trace length'.format(self.session_id))

        elif padding_len > 0:
            padding = np.empty(padding_len)
            padding[:] = np.nan
            dxds = np.append(dxds, padding, axis=0)

        return dxds

    @lazy_property
    def stimulus_timestamps(self):
        intervals = self.stimulus_pkl['intervalsms'] / 1000.0
        timestamps = np.cumsum(intervals)
        timestamps = np.concatenate([[0.0], timestamps[:-1]])
        alignment = self.time_sync['stimulus_alignment'][()].astype(np.int64)
        offset = self.twop_timestamps[alignment[0]]
        return offset + timestamps

    @lazy_property
    def motion_correction(self):
        motion_xy_path = self._lims_reader.get_file(session_id=self.session_id, file_type='OphysMotionXyOffsetData')
        motion_correction_csv = pd.read_csv(motion_xy_path, header=None)
        return np.array([motion_correction_csv[1].values, motion_correction_csv[2].values])

    @lazy_property
    def eye_dlc_screen_mapping(self):
        screen_dlc_path = self._lims_reader.get_file(session_id=self.session_id, file_type='EyeDlcScreenMapping')
        if screen_dlc_path is None:
            return None

        else:
            return screen_dlc_path

    @lazy_property
    def units(self):
        return pd.DataFrame({
            'cell_id': self.roi_metrics['id'],
            'pos_x': self.roi_metrics[' cx'],
            'pos_y': self.roi_metrics[' cy'],
        })

    @property
    def event_times(self):
        if self._event_times is None:
            self._calculate_events()
        return self._event_times

    @lazy_property
    def l0_events_dff(self):
        dff_amps_exp = os.path.join(self._events_dir, '{}_*_dff.npz'.format(self.session_id))
        dff_amps_path = list(glob.glob(dff_amps_exp))
        assert(len(dff_amps_path) == 1)
        dff_amps_path = dff_amps_path[0]
        dff_amps = np.load(dff_amps_path)
        return dff_amps['dff'][self.valid_roi_indices, :]

    @property
    def event_amps(self):
        if self._event_amps is None:
            self._calculate_events()

        return self._event_amps

    @lazy_property
    def l0_events_true_false(self):
        events_exp = os.path.join(self._events_dir, '{}_*_False_True_events.npz'.format(self.session_id))
        events_path = list(glob.glob(events_exp))
        assert(len(events_path) == 1)
        events_path = events_path[0]
        tf_events = np.load(events_path)['ev']

        if tf_events.shape[0] < len(self.all_rois):
            dff_amps_exp = os.path.join(self._events_dir, '{}_*_dff.npz'.format(self.session_id))
            dff_amps_path = list(glob.glob(dff_amps_exp))[0]
            # dff_amps_path = dff_amps_path[0]
            dff_amps = np.load(dff_amps_path)['dff']
            nan_rows = np.isnan(dff_amps).any(axis=1)

            # n_nan_rows = np.count_nonzero(np.isnan(dff_amps).any(axis=1))
            assert(tf_events.shape[0] + np.count_nonzero(nan_rows) == len(self.all_rois))

            missing_rows_offsets = np.cumsum(nan_rows)
            fixed_valid_rois = self.valid_roi_indices - missing_rows_offsets[self.valid_roi_indices]
            return tf_events[fixed_valid_rois, :]

        return tf_events[self.valid_roi_indices, :]

    @property
    def event_indices(self):
        if self._event_indices is None:
            self._calculate_events()

        return self._event_indices

    @property
    def has_eye_tracking(self):
        return self.eye_dlc_screen_mapping is not None

    @lazy_property
    def aligned_eye_tracking_data(self):
        if not self.has_eye_tracking:
            return None
        else:
            dlc_h5 = h5py.File(self.eye_dlc_screen_mapping, 'r')

            pupil_area = dlc_h5['raw_pupil_areas']['values'][()]
            eye_area = dlc_h5['raw_eye_areas']['values'][()]
            pos = pd.read_hdf(self.eye_dlc_screen_mapping, 'raw_screen_coordinates_spherical')

            eye_frames = self.time_sync['eye_tracking_alignment'][()].astype(np.int)
            eye_frames = eye_frames[np.where(eye_frames > 0)]

            eye_area_sync = eye_area[eye_frames]
            pupil_area_sync = pupil_area[eye_frames]
            x_pos_sync = pos['x_pos_deg'].values[eye_frames]
            y_pos_sync = pos['y_pos_deg'].values[eye_frames]

            # correcting dropped camera frames
            test = eye_frames[np.isfinite(eye_frames)]
            test = test.astype(int)
            temp2 = np.bincount(test)
            dropped_camera_frames = np.where(temp2 > 2)[0]
            for a in dropped_camera_frames:
                null_2p_frames = np.where(eye_frames == a)[0]
                eye_area_sync[null_2p_frames] = np.NaN
                pupil_area_sync[null_2p_frames] = np.NaN
                x_pos_sync[null_2p_frames] = np.NaN
                y_pos_sync[null_2p_frames] = np.NaN

            eye_sync = pd.DataFrame(data=np.vstack((eye_area_sync, pupil_area_sync, x_pos_sync, y_pos_sync)).T,
                                    columns=('eye_area', 'pupil_area', 'x_pos_deg', 'y_pos_deg'))
            return eye_sync

    def _calculate_events(self):
        events_exp = os.path.join(self._events_dir, '{}_*_False_True_events.npz'.format(self.session_id))
        events_path = list(glob.glob(events_exp))
        assert(len(events_path) == 1)
        events_path = events_path[0]
        tf_events = np.load(events_path)

        dff_amps_exp = os.path.join(self._events_dir, '{}_*_dff.npz'.format(self.session_id))
        dff_amps_path = list(glob.glob(dff_amps_exp))
        assert(len(dff_amps_path) == 1)
        dff_amps_path = dff_amps_path[0]
        dff_amps = np.load(dff_amps_path)

        cells_counts = []
        all_amps = []
        all_times = []

        for _, row in self.roi_metrics.iterrows():
            row_idx = row['unfiltered_cell_index']
            event_indices = np.nonzero(tf_events['ev'][row_idx, :])[0]
            cell_amps = dff_amps['dff'][row_idx, event_indices]
            cell_times = self.twop_timestamps[event_indices]
            assert(cell_amps.shape[0] == cell_times.shape[0])

            cells_counts.append(cell_amps.shape[0])
            all_amps.append(cell_amps)
            all_times.append(cell_times)

        self._event_indices = np.cumsum(cells_counts)
        self._event_times = np.concatenate(all_times)
        self._event_amps = np.concatenate(all_amps)

        assert(self._event_times.shape[0] == self._event_amps.shape[0] == self._event_indices[-1])


class OphysSessionAtHome(OphysSession):
    def __init__(self, session_id, lims_reader):
        super(OphysSessionAtHome, self).__init__(session_id, lims_reader)
        self._stimulus_table_dir = '/data/visual_coding/stimulus_tables/'
        self._events_dir = '/data/visual_coding/events'
        self._running_speeds_dir = '/data/visual_coding/contrast_running_speeds'

    @lazy_property
    def session_metadata(self):
        session_metadata = self._lims_reader.get_session_metadata(self.session_id)
        session_dict = session_metadata.to_dict()
        session_dict['storage_directory'] = '/data/visual_coding/' + session_dict['storage_directory'][35:]
        storage_dir = session_dict['storage_directory']

        if not os.path.exists(storage_dir):
            msg = 'session storage_directory {} not found'.format(storage_dir)
            logger.exception(msg)
            raise ValueError(msg)

        return session_dict

    @lazy_property
    def experiment_metadata(self):
        exp_dict = super(OphysSessionAtHome, self).experiment_metadata
        exp_dict['storage_directory'] = '/data/visual_coding/' + exp_dict['storage_directory'][35:]
        return exp_dict
