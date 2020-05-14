import os
import argparse
import logging
import pandas as pd
import pynwb
from allensdk.brain_observatory.nwb import add_stimulus_timestamps, add_stimulus_presentations, \
    add_eye_gaze_mapping_data_to_nwbfile, read_eye_gaze_mappings

from lims_reader import LIMSReader, LIMSReaderAtHome
from ophys_session import OphysSession, OphysSessionAtHome


logger = logging.getLogger(__name__)


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
        data=session.motion_correction,
        timestamps=session.twop_timestamps,
        description="Number of pixels shifts measured during motion correction",
        unit="pixels",
    )

    ophys_module.add_data_interface(xy_mc)

def add_eye_tracking(session, ophys_module):
    raw_gaze_mapping_mod = pynwb.ProcessingModule(name='raw_gaze_mapping',
                                                  description='Gaze mapping processing module raw outputs')

    raw_gaze_mapping_mod = add_eye_gaze_data_interfaces(raw_gaze_mapping_mod,
                                                        pupil_areas=eye_gaze_data["raw_pupil_areas"],
                                                        eye_areas=eye_gaze_data["raw_eye_areas"],
                                                        screen_coordinates=eye_gaze_data["raw_screen_coordinates"],
                                                        screen_coordinates_spherical=eye_gaze_data["raw_screen_coordinates_spherical"],
                                                        synced_timestamps=eye_gaze_data["synced_frame_timestamps"])

    # Container for filtered gaze mapped data
    filt_gaze_mapping_mod = pynwb.ProcessingModule(name='filtered_gaze_mapping',
                                                   description='Gaze mapping processing module filtered outputs')

    filt_gaze_mapping_mod = add_eye_gaze_data_interfaces(filt_gaze_mapping_mod,
                                                         pupil_areas=eye_gaze_data["new_pupil_areas"],
                                                         eye_areas=eye_gaze_data["new_eye_areas"],
                                                         screen_coordinates=eye_gaze_data["new_screen_coordinates"],
                                                         screen_coordinates_spherical=eye_gaze_data["new_screen_coordinates_spherical"],
                                                         synced_timestamps=eye_gaze_data["synced_frame_timestamps"])

    return (raw_gaze_mapping_mod, filt_gaze_mapping_mod)


def create_nwb_file(session, nwb_file_path):
    logging.info('Staring session {}'.format(session.session_id))
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
        reference_frame="Intrinsic imaging home",
    )

    ophys_module = nwbfile.create_processing_module(session.ophys_module_name, 'contains optical physiology processed data')

    max_projection = pynwb.ophys.TwoPhotonSeries(
        name='max_project',
        data=[session.max_projection],
        imaging_plane=imaging_plane,
        dimension=[session.image_width, session.image_height],
        rate=30.0
    )
    # nwbfile.add_acquisition(max_projection)
    ophys_module.add_data_interface(max_projection)

    img_seg = pynwb.ophys.ImageSegmentation(name=session.image_segmentation_name)
    ophys_module.add_data_interface(img_seg)
    ps = img_seg.create_plane_segmentation(
        name=session.plane_segmentation_name,
        description="Segmentation for imaging plane (de Vries et al., 2019, Nat Neurosci)",
        imaging_plane=imaging_plane,
    )

    for cell_id, roi_mask in session.roi_masks.items():
        ps.add_roi(id=cell_id, image_mask=roi_mask)

    rt_region = ps.create_roi_table_region(description="segmented cells with cell_specimen_ids")

    ### Fluorescence traces ###
    fluorescence = pynwb.ophys.Fluorescence()
    ophys_module.add_data_interface(fluorescence)

    timestamps = session.twop_timestamps
    raw_traces = session.raw_traces
    assert(raw_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='raw_traces',
        data=raw_traces,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:raw_traces.shape[1]]
    )

    neuropil_traces = session.neuropil_traces
    assert(neuropil_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='neuropil_traces',
        data=neuropil_traces,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:neuropil_traces.shape[1]]
    )

    demixed_traces = session.demixed_traces
    assert(demixed_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='demixed_traces',
        data=demixed_traces,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:demixed_traces.shape[1]]
    )

    dff_traces = session.dff_traces
    assert(dff_traces.shape[1] <= timestamps.shape[0])
    fluorescence.create_roi_response_series(
        name='DfOverF',
        data=session.dff_traces,
        rois=rt_region,
        unit='lumens',
        timestamps=timestamps[:dff_traces.shape[1]]
    )

    ### Running Speed ###
    running_velocity = pynwb.base.TimeSeries(
        name="running_speed",
        data=session.running_velocity[:len(session.stimulus_timestamps)],
        timestamps=session.stimulus_timestamps,
        unit="cm/s",
        description="Speed of the mouse on a rotating wheel",
    )

    behavior_module = pynwb.behavior.BehavioralTimeSeries()
    behavior_module.add_timeseries(running_velocity)
    ophys_module.add_data_interface(behavior_module)

    ### Motion Correction ###
    # NOTE: Older versions of pynwb had a schema bug in ophys.CorrectedImageStack class that prevents it from being
    #       acccessed. Unfortunately currently allensdk has frozen pynwb at version 1.0.2. To fix we need to add the
    #       x/y corrections as a time series rather than using CorrectImageStack.
    # add_motion_correction_cis(session, ophys_module, nwbfile)
    add_motion_correction_pm(session, ophys_module, nwbfile)

    ### Subject and lab metadata ###
    sex_lu = {'F': 'female', 'M': 'male'}
    subject_metadata = pynwb.file.Subject(
        age=session.session_metadata['age'][1:],
        genotype=session.session_metadata['full_genotype'],
        sex=sex_lu.get(session.session_metadata['sex'], 'unknown'),
        species='Mus musculus',
        subject_id=str(session.session_metadata['specimen_id']),
        description=session.session_metadata['name']
    )
    nwbfile.subject = subject_metadata

    ### Events ###
    pd.set_option('display.max_columns', None)
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

    ### Eye Tracking ###
    eye_dlc_path = session.eye_dlc_screen_mapping
    if eye_dlc_path is not None:
        eye_gazing_data = read_eye_gaze_mappings(eye_dlc_path)
        add_eye_gaze_mapping_data_to_nwbfile(nwbfile, eye_gazing_data)

    with pynwb.NWBHDF5IO(str(nwb_file_path), mode="w") as io:
        io.write(nwbfile)


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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

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

    for ses_id in session_ids:
        nwb_path = os.path.join(args.output_dir, '{}.nwb'.format(ses_id))
        if not args.force_overwrite and os.path.exists(nwb_path):
            logger.warning('file {} already exists, skipping session (use -f option to overwrite).'.format(nwb_path))
            continue

        session = session_cls(ses_id, lims_reader)
        create_nwb_file(session=session, nwb_file_path=nwb_path)
