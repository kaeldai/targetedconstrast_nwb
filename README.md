# targetedconstrast_nwb

A few simple scripts for generating NWB (2.0) files from the ophys Target Contrast experiments, (stimulus_name = 'VisCodingTargetedContrast').

## Requirements
* Python 3+
* You should install the "master" version of AllenSDK (as of 05/14/2020). All packages used by these scripts are also meet
  by the AllenSDK requirements.
* The various data files required to package all data in a ophys session/experiment are found on the aibs shared drive 
_/allen/production/neuralcoding/\<prod>/\<specimen>/\<session>/_. Some of the metadata is also found in the lims db, 
although I've cached the relevant tables under _lims_cache/_.

### Generating NWB files
One nwb will be generated for each session with file name _<session-id>.nwb_. To generate nwb files using a set of session ids:

```bash
$ python build_nwb.py --output-dir /path/to/nwb/folders <session-id> <session-id> ...
```

If you have a manifest file with columns **ophys_session_id** and **stimulus_name**
```bash
$ python build_nwb.py --output-dir /path/to/nwb/folders -manifest target_manifest.csv
``` 

See ```$python build_nwb.py -h``` for more options


### Reading NWB files

I've also created a class **brain_observatory_nwb2_data.BrainObservatoryNwb2DataSet**, it is similar to **allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet** 
as much as possible, expect it reads nwb files using NWB2/pynwb. There is also an accompanying jupyter notebook that shows
example of how to use this class.

### Relevant files
* build_nwb.py - main script for generating nwb (2.0) files
* ophys_session.py - Containers Helper object for pulling and formating various session data and metadata.
* lims_reader.py - Helper classes for fetching metadata and file-paths from the lims database. 
* lims_cache/*.csv - Contains relevant data taken from lims ophys_session/ophys_experiments/well_known_files tables (as of the
morning of 05/14/20). Makes the process faster and allows working outside the network.
* Check NWB.ipynb - A reproduction of the ophys tutorial notebook but using the NWB 2.0 files.
* limsdb2_props.json - properties for connecting to the lims database

### TO-DO:

* Missing metadata values: __device__, __fov__, __pipeline_version__, __session_type__, specimen_name__.
* Need to better parse stimulus table:
  * Figure out appropiate __stimulus_name__ values?
  * Probably need to differentiate different types of stimulus when __sweep_number__ == -1
* Find __roi_ids__
* Find/calculate __neuropil_r__
* To implement __get_corrected_fluorescence_traces()__ need to figure out __neuropil_r__ and __pipeline_version__
* Implement eye-tracking (__get_pupil_location()__ and __get_pupil_size()__)