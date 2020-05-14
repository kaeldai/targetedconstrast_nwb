import os
import pandas as pd
import pandas.io.sql as sqlio
import logging
import json
from psycopg2 import connect


logger = logging.getLogger(__name__)

# A database view of relevant metadata for any given session
VIEW_SES_METADATA = """
SELECT os.id as session_id,                                                                                                                          
       os.name,                                                                                                                                      
       os.stimulus_name,                                                                                                                             
       os.storage_directory,                                                                                                                         
       sp.external_specimen_name,                                                                                                                    
       TRIM(TRAILING '-' FROM TRIM(TRAILING sp.external_specimen_name FROM sp.name)) AS genotype,                                                    
       os.date_of_acquisition,                                                                                                                       
       e.name AS rig,                                                                                                                                
       d.date_of_birth,                                                                                                                              
       d.full_genotype,                                                                                                                              
       a.name as age,                                                                                                                                
       g.name as sex,                                                                                                                                
       os.parent_session_id,                                                                                                                         
       os.workflow_state,                                                                                                                            
       p.code AS project,                                                                                                                            
       os.stimulus_name                                                                                                                              
FROM ophys_sessions os                                                                                                                               
     LEFT JOIN ophys_experiments oe ON oe.ophys_session_id = os.id                                                                                   
     JOIN specimens sp ON sp.id = os.specimen_id                                                                                                     
     LEFT JOIN equipment e ON e.id = os.equipment_id                                                                                                 
     JOIN projects p ON p.id = os.project_id                                                                                                         
     INNER JOIN donors d on d.id = sp.donor_id                                                                                                       
     INNER JOIN ages a ON a.id = d.age_id                                                                                                            
     INNER JOIN genders g on g.id = d.gender_id                                                                                                      
"""

VIEW_EXP_METADATA = """
select oe.id,
    oe.name,
    oe.ophys_session_id,
    oe.storage_directory,
    de.depth,
    st.acronym as area,
    oe.workflow_state,
    e.name as rig,
    oe.calculated_depth
from ophys_experiments oe
    INNER JOIN ophys_sessions os ON oe.ophys_session_id = os.id
    LEFT JOIN imaging_depths de ON de.id = oe.imaging_depth_id
    LEFT JOIN structures st ON st.id = oe.targeted_structure_id
    LEFT JOIN equipment e ON e.id = os.equipment_id
"""

# A view for the well_known_files (particulary the raw/demixed/corrected traces) associated with ophys sessions.
VIEW_WKF_TRACES = """
SELECT os.id as session_id,
    oe.id AS experiment_id,
    wkf.storage_directory,
    wkf.filename,
    wkft.name AS file_type,
    os.stimulus_name
FROM well_known_files wkf
    INNER JOIN ophys_experiments oe ON wkf.attachable_id = oe.id
    INNER JOIN ophys_sessions os ON oe.ophys_session_id = os.id
    LEFT JOIN well_known_file_types wkft ON wkft.id = wkf.well_known_file_type_id
"""

# Like VIEW_WKF_TRACES but these are files generated by the ophys cell segmentations. Ideally we'd like to have
#  one view for well_known_files associated with each ophys_session, but wasn't sure how to do that with different
#  FK lookups. Instead query the well_known_files table twice and use a UNION operation.
VIEW_WKF_SEGMENTATIONS = """
SELECT os.id as session_id,
    oe.id as experiment_id,
    wkf.storage_directory,
    wkf.filename,
    wkft.name AS file_type,
    os.stimlus_name
FROM ophys_experiments oe
    LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id
    LEFT JOIN well_known_files wkf ON wkf.attachable_id = ocsr.id
    LEFT JOIN well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
    INNER JOIN ophys_sessions os on oe.ophys_session_id = os.id
"""

VIEW_WKF_SESSIONS = """
SELECT wkf.filename, 
       wkft.name 
FROM well_known_files wkf 
     INNER JOIN ophys_sessions os ON os.id = wkf.attachable_id 
     LEFT JOIN well_known_file_types wkft ON wkf.well_known_file_type_id = wkft.id 
"""


class LIMSReader(object):
    """Class for getting session and experiment metadata, plus location of well-known-files, required to generate
    NWB files for (TargetedContrast) ophys sessions.

    To make testing faster and allow for work outside the AI network, have option to cache session, experiment,
    and wkf metadata to csv files (if they exists). By default will look in caches for relevant info first then try
    to query the lims db.

    To (re)generate the cache files, log onto a machine that can read from the limsdb and run
    ```python
    LIMSReader.build_cache(cache_dir='lims_cache', overwrite=True)
    ```

    """
    session_cache_fn = 'lims_ophys_sessions.csv'
    experiment_cache_fn = 'lims_ophys_experiments.csv'
    wkf_cache_fn = 'lims_wkf.csv'

    def __init__(self, lims_cache_dir='lims_cache', lims_conn_props='limsdb2_props.json'):
        """
        :param lims_cache_dir: path to directory containing csv caches, if None then will always query the database
        :param lims_conn_props: dictionary or json file containing information to connect to lims database
        """
        if lims_cache_dir is None:
            logger.info('No lims cache directory found. Will attempt to query lims database for session metadata.')

        # The __load_cache() method will work even if lims_cache_dir
        self._lims_session_cache = self.__load_cache(lims_cache_dir, LIMSReader.session_cache_fn, 'session')
        if self._lims_session_cache is not None:
            self._lims_session_cache = self._lims_session_cache.set_index('session_id')

        self._lims_experiment_cache = self.__load_cache(lims_cache_dir, LIMSReader.experiment_cache_fn, 'experiment')
        self._lims_wkf_cache = self.__load_cache(lims_cache_dir, LIMSReader.wkf_cache_fn, 'well-known-files')

        if lims_conn_props is None:
            logger.info('No lims database json file/dictionary given, will use lims_cache_dir only.')
            self._limsdb_props = None
        elif isinstance(lims_conn_props, dict):
            self._limsdb_props = lims_conn_props
        else:
            self._limsdb_props = json.load(open(lims_conn_props, 'r'))

    def __load_cache(self, lims_cache_dir, csv_file, cache_type):
        if lims_cache_dir is None:
            return None
        else:
            file_path = os.path.join(lims_cache_dir, csv_file)
            if not os.path.exists(file_path):
                logger.info('Could not find csv cach {} under {}. Will fetch {} metdata from lims database.'.format(
                    lims_cache_dir, csv_file, cache_type))
                return None

            return pd.read_csv(file_path, index_col=False)

    def get_session_metadata(self, session_id):
        """Return ophys session metadata for a given session. Will look in the cache if available, otherwise will try
        to directly query the ophys_sessions table in lims.

        :param session_id:
        :return: A dictionary
        """
        if self._lims_session_cache is not None:
            if session_id in self._lims_session_cache.index:
                return self._lims_session_cache.loc[session_id]
            elif self._limsdb_props is not None:
                logger.debug('session {} not cached, looking in lims')
        else:
            query = VIEW_SES_METADATA + ' WHERE os.id = {}'.format(session_id)
            results_df = self.query_lims(query, self._limsdb_props)
            if len(results_df) == 0:
                raise IOError('Could not find session id {} in ophys_sessions table'.format(session_id))
            elif len(results_df) > 1:
                raise IOError('Mulitple rows with session id {} in ophys_sessions table'.format(session_id))
            else:
                return results_df.iloc[0].to_dict()

        logger.error('Could not find metadata for ophys session id {}'.format(session_id))

    def get_experiment_metadata(self, session_id):
        """Returns ophys experiment metadata for a given session id, either from the cache or directly from lims
        database. This assumes there's one and only experiment associated with an id

        :param session_id:
        :return:
        """
        if self._lims_experiment_cache is not None:
            # Check the cache csv
            exps = self._lims_experiment_cache[self._lims_experiment_cache['session_id'] == session_id]
            if len(exps) == 0:
                logger.debug('session {} not cached, looking in lims.')
            elif len(exps) > 1:
                raise ValueError('Session {} has multiple experiments.')
            else:
                return exps.iloc[0]
        elif self._limsdb_props is not None:
            # Try querying
            query = VIEW_EXP_METADATA + ' WHERE os.id = {}'.format(session_id)
            results_df = self.query_lims(query, self._limsdb_props)
            if len(results_df) == 0:
                raise IOError('Could not find any experiments for session id {}'.format(session_id))
            elif len(results_df) > 1:
                raise IOError('Mulitple experiments for session id {}'.format(session_id))
            else:
                return results_df.iloc[0].to_dict()

        logger.error('Could not find experiment metadata for ophys session id {}'.format(session_id))

    def get_file(self, session_id=None, file_type=None, experiment_id=None, stimulus_name=None):
        """Get path of well-known-file associated with a given session-id/experiment-id/file-type. This is so we
        don't have to guess the file-path names for various data.

        :param session_id: ophys_session.id
        :param experiment_id: ophys_experiments.id
        :param file_type: well_known_file_types.name,
        :param stimulus_name: ophys_sessions.stimulus_name, optional (prob. not required).
        :return: Absolute file path of file
        """
        if self._lims_wkf_cache is not None:
            wkf_mask = True
            if session_id is not None:
                wkf_mask &= self._lims_wkf_cache['session_id'] == session_id

            if file_type is not None:
                wkf_mask &= self._lims_wkf_cache['file_type'] == file_type

            if experiment_id is not None:
                wkf_mask &= self._lims_wkf_cache['experiment_id'] == experiment_id

            if stimulus_name is not None:
                wkf_mask &= self._lims_wkf_cache['stimulus_name'] == stimulus_name

            wkf_df = self._lims_wkf_cache[wkf_mask]
        elif self._limsdb_props is not None:
            conditionals = []
            if session_id is not None:
                conditionals.append('os.id = {}'.format(session_id))

            if experiment_id is not None:
                conditionals.append('oe.id = {}'.format(experiment_id))

            if file_type is not None:
                conditionals.append("wkft.name = '{}'".format(file_type))

            if stimulus_name is not None:
                conditionals.append("os.stimulus_name = '{}'".format(stimulus_name))

            where_stmt = ' WHERE {}'.format(' AND '.join(conditionals))
            query = VIEW_WKF_TRACES + where_stmt + ' UNION ' + VIEW_WKF_SEGMENTATIONS + where_stmt
            wkf_df = self.query_lims(query, self._limsdb_props)
        else:
            logger.error('Unable to access cache csv or lims db')
            wkf_df = None

        if len(wkf_df) == 0:
            return None

        wkf = wkf_df.iloc[0]
        return os.path.join(wkf['storage_directory'], wkf['filename'])

    @staticmethod
    def query_lims(query, conn_props):
        conn = connect(**conn_props)
        conn.set_session(readonly=True, autocommit=True)
        return sqlio.read_sql_query(query, conn)

    @staticmethod
    def build_cache(cache_dir, limsdb_props='limsdb2_props.json', stimulus_name='VisCodingTargetedContrast',
                    overwrite=False):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if isinstance(limsdb_props, dict):
            conn_props = limsdb_props
        else:
            conn_props = json.load(open(limsdb_props, 'r'))

        session_cache_path = os.path.join(cache_dir, LIMSReader.session_cache_fn)
        if overwrite or not os.path.exists(session_cache_path):
            logger.info('Generating cache: {}'.format(session_cache_path))
            query = VIEW_SES_METADATA + " WHERE os.stimulus_name = '{}'".format(stimulus_name)
            results_df = LIMSReader.query_lims(query, conn_props)
            results_df.to_csv(session_cache_path, index=False)

        else:
            logger.info('{} already exists, skipping.')

        experiment_cache_path = os.path.join(cache_dir, LIMSReader.experiment_cache_fn)
        if overwrite or not os.path.exists(experiment_cache_path):
            logger.info('Generating cache: {}'.format(experiment_cache_path))
            query = VIEW_EXP_METADATA + " WHERE os.stimulus_name = '{}'".format(stimulus_name)
            results_df = LIMSReader.query_lims(query, conn_props)
            results_df.to_csv(experiment_cache_path, index=False)

        else:
            logger.info('{} already exists, skipping.')

        wkf_cache_path = os.path.join(cache_dir, LIMSReader.wkf_cache_fn)
        if overwrite or not os.path.exists(wkf_cache_path):
            logger.info('Generating cache: {}'.format(wkf_cache_path))
            query = VIEW_WKF_SEGMENTATIONS + " WHERE os.stimulus_name = '{}'".format(stimulus_name)
            query += ' UNION ' + VIEW_WKF_TRACES + " WHERE os.stimulus_name = '{}'".format(stimulus_name)
            results_df = LIMSReader.query_lims(query, conn_props)
            results_df.to_csv(wkf_cache_path, index=False)

        else:
            logger.info('{} already exists, skipping.')


class LIMSReaderAtHome(LIMSReader):
    """So I can work on this without being logged into the AIBS network, I can replace /allen/programs/braintv with
    a path that exists on my LAN. Should be removed at some point
    """

    def __init__(self, old_base_dir, new_base_dir, **params):
        super(LIMSReaderAtHome, self).__init__(**params)
        self.old_base_dir = old_base_dir
        self.new_base_dir = new_base_dir

    def get_file(self, session_id=None, file_type=None, experiment_id=None, stimulus_name=None):
        file_path = super(LIMSReaderAtHome, self).get_file(session_id=session_id, file_type=file_type,
                                                           experiment_id=experiment_id, stimulus_name=stimulus_name)

        return file_path if file_path is None else file_path.replace(self.old_base_dir, self.new_base_dir)
