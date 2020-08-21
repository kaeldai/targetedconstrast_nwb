import h5py
import numpy as np
import pandas as pd
import pynwb

import allensdk.brain_observatory.roi_masks as roi
import allensdk.brain_observatory.stimulus_info as si


pd.set_option('display.max_columns', None)


class BrainObservatoryNwb2DataSet(object):
    BRAIN_OBSERVATORY_PIPELINE = 'brain_observatory_pipeline'
    MODULE_EYE_TRACKING = 'EyeBehavior'


    def __init__(self, nwb_file):
        self.nwb_file = nwb_file
        self.nwb_root = h5py.File(self.nwb_file, 'r')

        io = pynwb.NWBHDF5IO(self.nwb_file, 'r')
        self.nwb_session = io.read()

        self._stimulus_search = None

    def get_metadata(self):
        import re

        imaging_plane_metadata = self.nwb_session.imaging_planes['imaging_plane_1'].fields

        location_match = re.match(r'area: (\w+),depth: (\d+)', imaging_plane_metadata['location'])
        if location_match is not None:
            targeted_struct = location_match.group(1)
            targeted_depth = location_match.group(2)
        else:
            targeted_struct = None
            targeted_depth = np.nan

        session_metadata = {
            'age_days': int(self.nwb_session.subject.fields['age']),
            'sex': self.nwb_session.subject.fields['sex'],
            'genotype': self.nwb_session.subject.fields['genotype'],
            'indicator': imaging_plane_metadata['indicator'],
            'excitation_lambda': imaging_plane_metadata['excitation_lambda'],
            'device_name': imaging_plane_metadata['device'].name,
            'targeted_structure': targeted_struct,
            'imaging_depth_um': targeted_depth,
            'session_start_time': self.nwb_session.session_start_time,
            'ophys_experiment_id': self.nwb_session.identifier,

            # Missing data
            'device': None,
            #  'experiment_container_id': None,  # removed by Saskia
            'fov': None,
            #  'pipeline_version': None,  # removed by Saskia
            'session_type': None,
            'specimen_name': None,
        }

        genotype = session_metadata['genotype']
        session_metadata['cre_line'] = genotype.split(';')[0] if genotype else None

        return session_metadata

    def get_session_types(self):
        raise NotImplementedError()

    def _get_stimulus_table_df(self):
        """Converts /intervals/epochs into a dataframe"""
        # TODO: Cache table?
        stim_table = pd.DataFrame({
            col.name: col.data for col in self.nwb_session.epochs.columns
            if col.name not in ['tags', 'timeseries', 'tags_index', 'timeseries_index']
        }, index=pd.Index(data=self.nwb_session.epochs.id.data))

        stim_table = stim_table.rename(columns={
            'start_time': 'start',
            'stop_time': 'end',
            'stimulus_name': 'stimulus'
        })

        # NOTE: The original BOb api stores start/end intervals as integers so I'm copying the behavior
        stim_table['start'] = stim_table['start'].astype(np.int)
        stim_table['end'] = stim_table['end'].astype(np.int)

        return stim_table

    def get_stimulus_epoch_table(self):
        stim_table_df = self._get_stimulus_table_df()
        start_times = []
        stop_times = []
        stimulus_names = []
        for grp_key, grp_df in stim_table_df.groupby(['stimulus']):
            stimulus_names.append(grp_key[1])
            start_times.append(grp_df['start'].min())
            stop_times.append(grp_df['end'].max())

        epochs_df = pd.DataFrame({
            'stimulus': stimulus_names,
            'start': start_times,
            'end': stop_times
        }, index=range(len(stimulus_names)))

        return epochs_df

    def list_stimuli(self):
        """ Return a list of the stimuli presented in the experiment

        :return: list of strings
        """
        ## Saskia wanted to get rid of the "stimulus" column.
        # stim_table_df = self._get_stimulus_table_df()
        # return list(stim_table_df['stimulus'].unique())
        raise NotImplementedError()


    def get_stimulus_table(self):
        ## 'stimuls' column was removed
        # stim_table_df = self._get_stimulus_table_df()
        # if stimulus_name not in stim_table_df['stimulus'].values:
        #     raise IOError("Could not find a stimulus named '{}'".format(stimulus_name))
        # else:
        #     return stim_table_df[stim_table_df['stimulus'] == stimulus_name]
        return self._get_stimulus_table_df()


    @property
    def stimulus_search(self):
        # stimulus_info.StimulusSearch will not work 2 files
        # TODO: Reimplement StimulusSearch to work with NWB 2 files, or give a warning?
        raise NotImplementedError()
    def get_stimulus(self, frame_ind):
        # TODO: Need to implement get_stimulus_template
        raise NotImplementedError()

    def get_stimulus_template(self, stimulus_name):
        raise NotImplementedError()

    def get_locally_sparse_noise_stimulus_template(self, stimulus, mask_off_screen=True):
        raise NotImplementedError()

    def _get_roi_responses_helper(self, cell_specimen_ids, response_series, data_interface='Fluorescence'):
        """A helper fnc for getting different types of roi_traces data"""
        raw_traces_mod = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE].data_interfaces[data_interface].roi_response_series[response_series]
        rois = raw_traces_mod.rois.table.id[()]
        if cell_specimen_ids is None:
            raw_traces = raw_traces_mod.data[()]
        else:
            intersect = np.intersect1d(rois, cell_specimen_ids, return_indices=True)
            # Check if there are cell_specimen_ids not in rois
            missing_ids = set(cell_specimen_ids) - set(intersect[0])
            if missing_ids:
                raise ValueError('cell_specimen_ids {} not found.'.format(', '.join(str(a) for a in missing_ids)))

            # Return data of corresponding rows, make sure the order is the same
            table_indices = intersect[1]
            raw_traces = np.array([raw_traces_mod.data[i, :] for i in table_indices])

        return raw_traces

    def get_fluorescence_traces(self, cell_specimen_ids=None):
        """Returns an array of fluorescence traces for all ROI and the timestamps for each datapoint

        :param cell_specimen_ids: list or array (optional). List of cell IDs to return traces for. If this is None
        (default) then all are returned
        :return: 2D numpy array, 2D numpy array. Timestamp for each fluorescence sample, Fluorescence traces for each
        cell
        """
        raw_traces = self._get_roi_responses_helper(cell_specimen_ids, 'raw_traces')
        timestamps = self.get_fluorescence_timestamps()

        return timestamps, raw_traces

    def get_fluorescence_timestamps(self):
        """Returns an array of timestamps in seconds for the fluorescence traces"""
        traces_mod = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE].data_interfaces['Fluorescence'].roi_response_series['raw_traces']
        return traces_mod.timestamps[()]

    def get_neuropil_traces(self, cell_specimen_ids=None):
        """ Returns an array of neuropil fluorescence traces for all ROIs and the timestamps for each datapoint

        :param cell_specimen_ids: list or array (optional). List of cell IDs to return traces for. If this is None
        (default) then all are returned
        :return: 2D numpy array, 2D numpy array. Timestamp for each fluorescence sample, Fluorescence traces for each
        cell
        """
        neuropil_traces = self._get_roi_responses_helper(cell_specimen_ids, 'neuropil_traces')
        timestamps = self.get_fluorescence_timestamps()

        return timestamps, neuropil_traces

    def get_neuropil_r(self, cell_specimen_ids=None):
        """Returns a scalar value of r for neuropil correction of flourescence traces

        :param cell_specimen_ids: list or array (optional). List of cell IDs to return traces for. If this is None
        (default) then results for all are returned
        :return: 1D numpy array, len(r)=len(cell_specimen_ids) Scalar for neuropil subtraction for each cell
        """
        plane_seg_df = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE].data_interfaces['ImageSegmentation']['PlaneSegmentation'].to_dataframe()
        return plane_seg_df.loc[cell_specimen_ids]['neuropil_r'].values


    def get_demixed_traces(self, cell_specimen_ids=None):
        """Returns an array of demixed fluorescence traces for all ROIs and the timestamps for each datapoint

        :param cell_specimen_ids: list or array (optional). List of cell IDs to return traces for. If this is None
        (default) then all are returned
        :return: 2D numpy array, 2D numpy array. Timestamp for each fluorescence sample, Fluorescence traces for each
        cell
        """
        demixed_traces = self._get_roi_responses_helper(cell_specimen_ids, 'demixed_traces')
        timestamps = self.get_fluorescence_timestamps()

        return timestamps, demixed_traces

    def get_corrected_fluorescence_traces(self, cell_specimen_ids=None):
        """Returns an array of demixed and neuropil-corrected fluorescence traces for all ROIs and the timestamps for
        each datapoint

        :param cell_specimen_ids: list or array (optional). List of cell IDs to return traces for. If this is None
        (default) then all are returned
        :return: 2D numpy array, 2D numpy array. Timestamp for each fluorescence sample, Fluorescence traces for each
        cell
        """
        raise NotImplementedError()

    def get_dff_traces(self, cell_specimen_ids=None):
        """

        :param cell_specimen_ids:
        :return:
        """
        dff_traces = self._get_roi_responses_helper(cell_specimen_ids, 'DfOverF')
        timestamps = self.get_fluorescence_timestamps()

        return timestamps, dff_traces

    @property
    def number_of_cells(self):
        return len(self.get_cell_specimen_ids())

    def get_roi_ids(self):
        """

        :return:
        """
        raise NotImplementedError()

    def get_cell_specimen_ids(self):
        """

        :return:
        """
        segmentation_df = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE]['ImageSegmentation']['PlaneSegmentation'].to_dataframe()
        return list(segmentation_df.index.values)

    def get_cell_specimen_indices(self, cell_specimen_ids):
        all_cell_specimen_ids = list(self.get_cell_specimen_ids())

        try:
            inds = [list(all_cell_specimen_ids).index(i)
                    for i in cell_specimen_ids]
        except ValueError as e:
            raise ValueError("Cell specimen not found (%s)" % str(e))

        return inds

    def get_max_projection(self):
        return self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE]['max_project'].data[()][0]

    def get_roi_mask_array(self, cell_specimen_ids=None):
        roi_masks = self.get_roi_mask(cell_specimen_ids)
        if len(roi_masks) == 0:
            raise IOError("no masks found for given cell specimen ids")

        roi_arr = roi.create_roi_mask_array(roi_masks)
        return roi_arr

    def get_roi_mask(self, cell_specimen_ids=None):
        segmentations_df = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE]['ImageSegmentation']['PlaneSegmentation'].to_dataframe()
        if cell_specimen_ids is not None:
            segmentations_df = segmentations_df.loc[cell_specimen_ids]

        roi_array = []
        for roi_id, roi_seg_df in segmentations_df.iterrows():
            roi_mask = roi_seg_df['image_mask']
            m = roi.create_roi_mask(roi_mask.shape[1], roi_mask.shape[0], [0, 0, 0, 0], roi_mask=roi_mask, label=roi_id)
            roi_array.append(m)

        return roi_array

    def get_running_speed(self):
        running_speed_series = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE]['RunningBehavior'].get_timeseries('running_speed')
        dxcm = running_speed_series.data[()]
        dxtime = running_speed_series.timestamps[()]
        return dxcm, dxtime

    @property
    def has_eye_tracking(self):
        return self.MODULE_EYE_TRACKING in self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE].data_interfaces

    def get_pupil_location(self, as_spherical=True):
        if not self.has_eye_tracking:
            raise ValueError('Session does not contain eye tracking data.')

        screen_coords_name = 'screen_coordinates_spherical' if as_spherical else 'screen_coordinates'
        if not screen_coords_name in self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE][self.MODULE_EYE_TRACKING].time_series:
            # Some data might not contain non-spherical coordinates
            raise ValueError('Could not find {}spherical pupil coordinates'.format('' if as_spherical else 'non-'))

        screen_coords = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE][self.MODULE_EYE_TRACKING].get_timeseries(screen_coords_name)
        coords = np.array(screen_coords.data).T
        timestamps = np.array(screen_coords.timestamps)

        return timestamps, coords

    def get_pupil_size(self):
        if not self.has_eye_tracking:
            raise ValueError('Session does not contain eye tracking data.')

        pupil_size_mod = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE][self.MODULE_EYE_TRACKING].get_timeseries('pupil_area')
        pupil_data = np.array(pupil_size_mod.data)
        timestamps = np.array(pupil_size_mod.timestamps)

        return timestamps, pupil_data

    def get_eye_area(self):
        if not self.has_eye_tracking:
            raise ValueError('Session does not contain eye tracking data.')

        eye_area_mod = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE][self.MODULE_EYE_TRACKING].get_timeseries('eye_area')
        area_data = np.array(eye_area_mod.data)
        timestamps = np.array(eye_area_mod.timestamps)

        return timestamps, area_data


    def get_motion_correction(self):
        mc_module = self.nwb_session.modules[self.BRAIN_OBSERVATORY_PIPELINE]['MotionCorrection']
        mc_df = pd.DataFrame({
            'timestamp': mc_module.timestamps[:-1],
            'x_motion': mc_module.data[0, :],
            'y_motion': mc_module.data[1, :]
        }, index=range(mc_module.data.shape[1]))

        return mc_df

    def save_analysis_dataframes(self, *tables):
        raise NotImplementedError()

    def save_analysis_arrays(self, *datasets):
        raise NotImplementedError()

    def get_units_table(self, cell_specimen_ids=None):
        # TODO: Cache table?
        units_df = self.nwb_session.units.to_dataframe()
        units_df = units_df.rename(columns={'cell_id': 'cell_specimen_id'})
        units_df = units_df.set_index('cell_specimen_id')
        if cell_specimen_ids is not None:
            units_df = units_df.loc[cell_specimen_ids]

        return units_df

    def get_event_times(self, cell_specimen_ids=None):
        units_df = self.get_units_table(cell_specimen_ids=cell_specimen_ids)
        event_times = {}
        for cell_id, units_row in units_df.iterrows():
            event_times[cell_id] = units_row['event_times']

        return event_times

    def get_event_amplitudes(self, cell_specimen_ids=None):
        units_df = self.get_units_table(cell_specimen_ids=cell_specimen_ids)
        event_amps = {}
        for cell_id, units_row in units_df.iterrows():
            event_amps[cell_id] = units_row['event_amplitudes']

        return event_amps

    def get_l0_dff_events(self, cell_specimen_ids=None):
        raw_traces = self._get_roi_responses_helper(cell_specimen_ids, 'dff_events', data_interface='l0_events')
        timestamps = self.get_fluorescence_timestamps()

        return timestamps, raw_traces

    def get_l0_true_false_events(self, cell_specimen_ids=None):
        raw_traces = self._get_roi_responses_helper(cell_specimen_ids, 'true_false_events', data_interface='l0_events')
        timestamps = self.get_fluorescence_timestamps()

        return timestamps, raw_traces


def align_running_speed(dxcm, dxtime, timestamps):
    ''' If running speed timestamps differ from fluorescence
    timestamps, adjust by inserting NaNs to running speed.

    Returns
    -------
    tuple: dxcm, dxtime
    '''
    if dxtime[0] != timestamps[0]:
        adjust = np.where(timestamps == dxtime[0])[0][0]
        dxtime = np.insert(dxtime, 0, timestamps[:adjust])
        dxcm = np.insert(dxcm, 0, np.repeat(np.NaN, adjust))
    adjust = len(timestamps) - len(dxtime)
    if adjust > 0:
        dxtime = np.append(dxtime, timestamps[(-1 * adjust):])
        dxcm = np.append(dxcm, np.repeat(np.NaN, adjust))

    return dxcm, dxtime


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # nwb_session_path = 'output/682746585.nwb'
    nwb_session_path = 'output/694856258.nwb'
    data_set = BrainObservatoryNwb2DataSet(nwb_session_path)
    selected_cell_id = np.random.choice(data_set.get_cell_specimen_ids(), size=1)[0]
    # times, dff_events = data_set.get_l0_dff_events(cell_specimen_ids=[selected_cell_id])
    # plt.figure()
    # plt.plot(times, dff_events[0])
    #
    # times, true_false_events = data_set.get_l0_true_false_events(cell_specimen_ids=[selected_cell_id])
    # plt.figure()
    # plt.plot(times, true_false_events[0], '.')
    #
    # plt.show()

    # print(data_set.get_neuropil_r(cell_specimen_ids=[selected_cell_id]))
    if data_set.has_eye_tracking:
        timestamps, coords = data_set.get_pupil_location()
        # plt.figure()
        # plt.plot(timestamps, coords[0], label='azimuth')
        # plt.plot(timestamps, coords[1], label='altitude')
        # plt.legend()
        # plt.title('Eye Position')
        # plt.ylabel('Angle (degrees)')

        timestamps, pupil_area = data_set.get_pupil_size()
        plt.figure()
        plt.plot(timestamps, pupil_area)
        plt.ylabel('area (pixels)')
        plt.title('Pupil Size')


        timestamps, eye_area = data_set.get_eye_area()
        plt.figure()
        plt.plot(timestamps, eye_area)
        plt.ylabel('area (pixels)')
        plt.title('Eye area')

        plt.show()
