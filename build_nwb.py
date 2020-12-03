import os
import sys
import argparse
import logging
import pandas as pd
import pynwb
import numpy as np
import traceback

### It would be nice the use the allensdk, but the dependencies of the allensdk are so broken at the moment that we
###   can't use latest version of pynwb.
# from allensdk.brain_observatory.nwb import \
#     add_stimulus_timestamps, \
#     add_stimulus_presentations, \
#     add_eye_gaze_mapping_data_to_nwbfile, \
#     read_eye_gaze_mappings

from lims_reader import LIMSReader, LIMSReaderAtHome
from ophys_session import OphysSession, OphysSessionAtHome

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
except Exception as exc:
    MPI_rank = 0
    MPI_size = 1


logger = logging.getLogger(__name__)
log_format = '[%(asctime)s %(levelname)s] %(message)s'


pd.set_option('display.max_columns', None)

def add_stimulus_timestamps(nwbfile, stimulus_timestamps, module_name='stimulus'):
    """Function taken from allensdk.brain_observatory.nwb"""
    stimulus_ts = pynwb.TimeSeries(
        data=stimulus_timestamps,
        name='timestamps',
        timestamps=stimulus_timestamps,
        unit='s'
    )

    stim_mod = pynwb.ProcessingModule(module_name, 'Stimulus Times processing')

    nwbfile.add_processing_module(stim_mod)
    stim_mod.add_data_interface(stimulus_ts)

    return nwbfile


def setup_table_for_epochs(table, timeseries, tag):
    """Function taken from allensdk.brain_observatory.nwb"""
    table = table.copy()
    indices = np.searchsorted(timeseries.timestamps[:], table['start_time'].values)
    if len(indices > 0):
        diffs = np.concatenate([np.diff(indices), [table.shape[0] - indices[-1]]])
    else:
        diffs = []

    table['tags'] = [(tag,)] * table.shape[0]
    table['timeseries'] = [[[indices[ii], diffs[ii], timeseries]] for ii in range(table.shape[0])]
    return table


def add_stimulus_presentations(nwbfile, stimulus_table, tag='stimulus_epoch'):
    """Adds a stimulus table (defining stimulus characteristics for each time point in a session) to an nwbfile as
    epochs. Function taken from allensdk.brain_observatory.nwb

    """
    stimulus_table = stimulus_table.copy()

    ts = nwbfile.modules['stimulus'].get_data_interface('timestamps')

    for colname, series in stimulus_table.items():
        types = set(series.map(type))
        if len(types) > 1 and str in types:
            series.fillna('', inplace=True)
            stimulus_table[colname] = series.transform(str)

    stimulus_table = setup_table_for_epochs(stimulus_table, ts, tag)
    container = pynwb.epoch.TimeIntervals.from_dataframe(stimulus_table, 'epochs')
    nwbfile.epochs = container

    return nwbfile


def add_motion_correction_cis(session, ophys_module, nwbfile):
    corrected_image_series = pynwb.image.ImageSeries(
        name="motion_corrected_movie",
        description="see external file",
        external_file=["URL"],
        starting_frame=[0],
        format="external",
        timestamps=session.twop_timestamps,
        unit="Fluorescence (a.u.)",
    )
    nwbfile.add_acquisition(corrected_image_series)

    orig_image_series = pynwb.image.ImageSeries(
        name="original_movie",
        description="see external file",
        external_file=["URL"],
        starting_frame=[0],
        format="external",
        timestamps=session.twop_timestamps,
        unit="Fluorescence (a.u.)",
    )
    nwbfile.add_acquisition(orig_image_series)

    mot_corr_traces = pynwb.TimeSeries(
        name="MotionCorrection",
        data=session.motion_correction,
        timestamps=session.twop_timestamps,
        description="Number of pixels shifts measured during motion correction",
        unit="pixels",
    )

    nwbfile.add_acquisition(mot_corr_traces)

    corr_obj = pynwb.ophys.CorrectedImageStack(
        corrected=corrected_image_series,
        original=orig_image_series,
        xy_translation=mot_corr_traces,
    )

    # corrected_image_series.parent = corr_obj
    # mot_corr_traces.parent = corr_obj
    ophys_module.add_data_interface(corr_obj)


def add_motion_correction_pm(session, ophys_module, nwbfile):
    xy_mc = pynwb.TimeSeries(
        name="MotionCorrection",
        data=session.motion_correction.T,
        timestamps=session.twop_timestamps,
        description="Number of pixels shifts measured during motion correction",
        unit="pixels",
    )

    ophys_module.add_data_interface(xy_mc)


def add_events_discrete(session, ophys_module, nwbfile):
    nwbfile.units = pynwb.misc.Units.from_dataframe(session.units, name='units')
    # events_indices = session.event_indices
    nwbfile.units.add_column(
        name='event_times',
        data=session.event_times,
        index=session.event_indices,
        description='times (s) of detected L0 events'
    )

    nwbfile.units.add_column(
        name='event_amplitudes',
        data=session.event_amps,
        index=session.event_indices,
        description='amplitudes (s) of detected L0 events'
    )


def add_events_contiguous(session, ophys_module, nwbfile, roi_table):
    nwbfile.units = pynwb.misc.Units.from_dataframe(session.units, name='units')

    l0_events = pynwb.ophys.Fluorescence(name='l0_events')
    ophys_module.add_data_interface(l0_events)

    dff_events = session.l0_events_dff
    l0_events.create_roi_response_series(
        name='dff_events',
        data=dff_events.T,
        rois=roi_table,
        unit='lumens',
        timestamps=session.twop_timestamps[:dff_events.shape[1]]
    )

    true_false_events = session.l0_events_true_false
    l0_events.create_roi_response_series(
        name='true_false_events',
        data=true_false_events.T,
        rois=roi_table,
        unit='lumens',
        timestamps=session.twop_timestamps[:true_false_events.shape[1]]
    )


def add_eye_tracking_interface(session, nwb_module):
    if not session.has_eye_tracking:
        logger.info('Session {}: No eye_tracking data.'.format(session.session_id))
        return

    eye_tracking_df = session.aligned_eye_tracking_data

    timestamps = session.twop_timestamps
    pupil_area_ts = pynwb.base.TimeSeries(
        name='pupil_area',
        data=eye_tracking_df['pupil_area'].values,
        timestamps=timestamps,
        unit='Pixels ^ 2'
    )

    eye_area_ts = pynwb.base.TimeSeries(
        name='eye_area',
        data=eye_tracking_df['eye_area'].values,
        timestamps=timestamps,
        unit='Pixels ^ 2'
    )

    screen_coord_spherical_ts = pynwb.base.TimeSeries(
        name='screen_coordinates_spherical',
        data=eye_tracking_df[['x_pos_deg', 'y_pos_deg']].values,
        timestamps=timestamps,
        unit='Degrees'
    )

    et_module = pynwb.behavior.BehavioralTimeSeries(name='EyeBehavior')
    et_module.add_timeseries(pupil_area_ts)
    et_module.add_timeseries(eye_area_ts)
    et_module.add_timeseries(screen_coord_spherical_ts)

    nwb_module.add_data_interface(et_module)
    # et_module.add_data_interface(eye_area_ts)
    # et_module.add_data_interface(screen_coord_spherical_ts)



def create_nwb_file(session, nwb_file_path):
    logger.info('Session {}: Starting'.format(session.session_id))

    nwbfile = pynwb.NWBFile(
        session_description=session.description,
        session_id=str(session.session_id),
        identifier=str(session.experiment_metadata['experiment_id']),
        session_start_time=session.start_time,
        # experiment_description=str(session.experiment_metadata['experiment_id'])
    )

    ### Build the stimulus table ###
    stim_table_df = session.stimulus_table
    nwbfile = add_stimulus_timestamps(nwbfile, stim_table_df['start_time'].values)
    nwbfile = add_stimulus_presentations(nwbfile, stim_table_df)

    ### Image Segmentation ###
    optical_channel = pynwb.ophys.OpticalChannel(
        name='optical_channel',
        description='description',
        emission_lambda=session.emission_wavelength)

    # device = pynwb.device.Device(name='Allen Institute two-photon pipeline: {}'.format(session.session_metadata['rig']))
    device = pynwb.device.Device(name='{}'.format(session.session_metadata['rig']))
    nwbfile.add_device(device)

    imaging_plane = nwbfile.create_imaging_plane(
        name=session.imaging_plane_name,
        optical_channel=optical_channel,
        description=session.imaging_plane_name,
        device=device,
        excitation_lambda=session.excitation_lambda,
        imaging_rate=session.imaging_rate,
        indicator=session.calcium_indicator,
        location='area: {},depth: {}'.format(session.experiment_metadata["area"],
                                             str(session.experiment_metadata["depth"])),
        unit="Fluorescence (au)",
        reference_frame="Intrinsic imaging home"
    )

    ophys_module = nwbfile.create_processing_module(session.ophys_module_name, 'contains optical physiology processed data')

    max_projection = pynwb.ophys.TwoPhotonSeries(
        name='max_project',
        data=[session.max_projection],
        imaging_plane=imaging_plane,
        dimension=[session.image_width, session.image_height],
        rate=30.0,
        unit='',
        # field_of_view=(session.fov[0], session.fov[1])
    )
    # nwbfile.add_acquisition(max_projection)
    ophys_module.add_data_interface(max_projection)

    img_seg = pynwb.ophys.ImageSegmentation(name=session.image_segmentation_name)
    ophys_module.add_data_interface(img_seg)
    ps = img_seg.create_plane_segmentation(
        name=session.plane_segmentation_name,
        description="Segmentation for imaging plane (de Vries et al., 2019, Nat Neurosci)",
        imaging_plane=imaging_plane
    )

    # Add a neuropil_r for each roi as a separate column to /processing/.../imagesegmentation/planeSegmentation
    ps.add_column(name='neuropil_r', description='neuropil_r')

    for (cell_id, roi_mask), neuropil_r in zip(session.roi_masks.items(), session.neuropil_r):
        ps.add_roi(id=cell_id, image_mask=roi_mask, neuropil_r=neuropil_r)
    logger.info('Session {}: {} valid ROIs.'.format(session.session_id, len(session.roi_masks)))

    rt_region = ps.create_roi_table_region(description="segmented cells with cell_specimen_ids")

    ### Fluorescence traces ###
    fluorescence = pynwb.ophys.Fluorescence()
    ophys_module.add_data_interface(fluorescence)

    timestamps = session.twop_timestamps
    raw_traces = session.raw_traces
    assert(raw_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='raw_traces',
        data=raw_traces.T,  # In the NWB guidelines times should be first dimension
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:raw_traces.shape[1]]
    )

    neuropil_traces = session.neuropil_traces
    assert(neuropil_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='neuropil_traces',
        data=neuropil_traces.T,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:neuropil_traces.shape[1]]
    )

    demixed_traces = session.demixed_traces
    assert(demixed_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='demixed_traces',
        data=demixed_traces.T,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:demixed_traces.shape[1]]
    )

    dff_traces = session.dff_traces
    assert(dff_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='DfOverF',
        data=dff_traces.T,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:dff_traces.shape[1]]
    )

    ### Running Speed ###
    running_speeds = session.running_velocity
    timestamps = session.twop_timestamps
    assert(running_speeds.shape[0] <= timestamps.shape[0])
    running_velocity = pynwb.base.TimeSeries(
        name="running_speed",
        data=running_speeds,
        timestamps=timestamps[:running_speeds.shape[0]],
        unit="cm/s",
        description="Speed of the mouse on a rotating wheel",
    )

    behavior_module = pynwb.behavior.BehavioralTimeSeries(name='RunningBehavior')
    behavior_module.add_timeseries(running_velocity)
    ophys_module.add_data_interface(behavior_module)

    ### Motion Correction ###
    # NOTE: Older versions of pynwb had a schema bug in ophys.CorrectedImageStack class that prevents it from being
    #       acccessed. Unfortunately currently allensdk has frozen pynwb at version 1.0.2. To fix we need to add the
    #       x/y corrections as a time series rather than using CorrectImageStack.
    # add_motion_correction_cis(session, ophys_module, nwbfile)
    add_motion_correction_pm(session, ophys_module, nwbfile)

    ### Subject and lab metadata ###
    sex_lu = {'F': 'F', 'M': 'M'}
    subject_metadata = pynwb.file.Subject(
        age=session.session_metadata['age'] + 'D',
        genotype=session.session_metadata['full_genotype'],
        sex=sex_lu.get(session.session_metadata['sex'], 'U'),
        species='Mus musculus',
        subject_id=str(session.session_metadata['specimen_id']),
        description=session.session_metadata['name']
    )
    nwbfile.subject = subject_metadata

    ### Events ###
    # add_events_discrete(session=session, ophys_module=ophys_module, nwbfile=nwbfile)
    add_events_contiguous(session=session, ophys_module=ophys_module, nwbfile=nwbfile, roi_table=rt_region)

    # pd.set_option('display.max_columns', None)
    # nwbfile.units = pynwb.misc.Units.from_dataframe(session.units, name='units')
    # # events_indices = session.event_indices
    # nwbfile.units.add_column(
    #     name='event_times',
    #     data=session.event_times,
    #     index=session.event_indices,
    #     description='times (s) of detected L0 events'
    # )
    #
    # nwbfile.units.add_column(
    #     name='event_amplitudes',
    #     data=session.event_amps,
    #     index=session.event_indices,
    #     description='amplitudes (s) of detected L0 events'
    # )

    # nwbfile.add_lab_meta_data({'foo': 'bar'})

    ### Eye Tracking ###
    add_eye_tracking_interface(session=session, nwb_module=ophys_module)

    logger.info('Session {}: Saving NWB to {}'.format(session.session_id, nwb_file_path))
    with pynwb.NWBHDF5IO(str(nwb_file_path), mode="w") as io:
        io.write(nwbfile)

    logger.info('Session {}: Completed'.format(session.session_id))


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_TARGETED_MANIFEST = os.path.join(SCRIPT_DIR, 'targeted_manifest.csv')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
DEFAULT_STIM_NAME = 'VisCodingTargetedContrast'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package ophys session data into NWB 2.0')
    parser.add_argument('session_ids', type=int, nargs='*',
                        help='The ophys_session_id, or space-separated list, being packaged into NWB format. If none ' 
                             'are specified will use the --manifest csv file to find ids.')

    parser.add_argument('-m', '--manifest', default=DEFAULT_TARGETED_MANIFEST,
                        help='a csv file containing column of ophys_session_ids, used along with --stimulus-names to ' 
                             'get session-ids if none are explicitly given')

    parser.add_argument('-s', '--stimulus-name', default=DEFAULT_STIM_NAME,
                        help='stimulus_name value used when using manifest or when rebuilding the lims cache ' 
                             '(ignored if using specified session_ids). Either single value or comma-separted list.')

    parser.add_argument('-o', '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='directory where nwb session file(s) will be written.')

    parser.add_argument('-c', '--create-cache', action='store_true',
                        help='Connects to the lims database (using --lims-conns) and caches relevant information into ' 
                             '--lims-cache-dir directory.')

    parser.add_argument('-f', '--force-overwrite', action='store_true',
                        help='overwrite existing nwb file(s) if exists.')

    parser.add_argument('--off-network', action='store_true',
                        help='Only use if you are testing this outside the the AIBS network')

    parser.add_argument('--lims-conns', type=str, default='limsdb2_props.json',
                        help='json file containing lims connectivity props')

    parser.add_argument('--lims-cache-dir', type=str, default='lims_cache',
                        help='Directory of csv caches of ophys_sessions, ophys_experiments and well_know_files table')

    parser.add_argument('--log-file', type=str, default=None,
                        help='path to file where log of progress will be saved. By default will only log to console')

    args = parser.parse_args()

    # logging.basicConfig(level=logging.INFO, format=log_format)
    log_formatter = logging.Formatter(log_format)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    if args.session_ids is None or len(args.session_ids) == 0:
        parsed_stim_names = args.stimulus_name.split(',')
        manifest_df = pd.read_csv(args.manifest, index_col=0)
        selected_sessions_df = manifest_df[manifest_df['stimulus_name'].isin(parsed_stim_names)]
        session_ids = selected_sessions_df['ophys_session_id'].values
        logger.info('Found {} matching session_ids in {} with stimulus_name values {}'.format(
            len(session_ids), args.manifest, ','.join(parsed_stim_names))
        )
    else:
        session_ids = args.session_ids

    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_file is not None:
        log_path = args.log_file if os.path.isabs(args.log_file) else os.path.join(args.output_dir, args.log_file)
        file_logger = logging.FileHandler(log_path)
        file_logger.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_logger)

    if args.off_network:
        session_cls = OphysSessionAtHome
        lims_reader = LIMSReaderAtHome(old_base_dir='/allen/programs/braintv/production/',
                                       new_base_dir='/data/visual_coding/',
                                       lims_conn_props=None)
    else:
        session_cls = OphysSession
        lims_reader = LIMSReader(lims_conn_props=None)

    if args.create_cache:
        cache_dir = args.lims_cache_dir
        logger.info('(Re)generating the lims cache at {}'.format(cache_dir))
        LIMSReader.build_cache(cache_dir=cache_dir,
                               limsdb_props=args.lims_conns,
                               stimulus_name=args.stimulus_name,
                               overwrite=args.force_overwrite)
        exit()

    if MPI_rank == 0:
        logger.info('>> Converting {} session(s) to NWB <<'.format(len(session_ids)))

    for ses_id in session_ids[MPI_rank::MPI_size]:
        nwb_path = os.path.join(args.output_dir, '{}.nwb'.format(ses_id))
        if not args.force_overwrite and os.path.exists(nwb_path):
            logger.warning('file {} already exists, skipping session (use -f option to overwrite).'.format(nwb_path))
            continue

        session = session_cls(ses_id, lims_reader)

        try:
            create_nwb_file(session=session, nwb_file_path=nwb_path)
        except Exception as e:
            logger.exception(msg='Session {}: Failed Conversion'.format(ses_id))

    if MPI_rank == 0:
        logger.info('>> Completed <<'.format(len(session_ids)))
