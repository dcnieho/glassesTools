[![Downloads](https://static.pepy.tech/badge/glassestools)](https://pepy.tech/project/glassestools)
[![PyPI Latest Release](https://img.shields.io/pypi/v/glassesTools.svg)](https://pypi.org/project/glassesTools/)
[![image](https://img.shields.io/pypi/pyversions/glassesTools.svg)](https://pypi.org/project/glassesTools/)

# GlassesTools v1.11.7
Tools for processing wearable eye tracker recordings.

If you use this package or any of the code in this repository, please cite:<br>
[Niehorster, D.C., Hessels, R.S., Benjamins, J.S., Nyström, M. and Hooge, I.T.C. (2023). GlassesValidator:
A data quality tool for eye tracking glasses. Behavior Research Methods. doi: 10.3758/s13428-023-02105-5](https://doi.org/10.3758/s13428-023-02105-5)


# API

## Eye tracker support
glassesTools supports the following eye trackers:
|Name|`glassesTools.eyetracker.EyeTracker` `Enum` value|
| --- | --- |
|AdHawk MindLink|EyeTracker.AdHawk_MindLink|
|Pupil Core|EyeTracker.Pupil_Core|
|Pupil Invisible|EyeTracker.Pupil_Invisible|
|Pupil Neon|EyeTracker.Pupil_Neon|
|SeeTrue STONE|EyeTracker.SeeTrue_STONE|
|SMI ETG 1 and ETG 2|EyeTracker.SMI_ETG|
|Tobii Pro Glasses 2|EyeTracker.Tobii_Glasses_2|
|Tobii Pro Glasses 3|EyeTracker.Tobii_Glasses_3|

Pull requests or partial help implementing support for further wearable eye trackers are gladly received. To support a new eye tracker,
device support in [`glassesTools.importing`](#glassestoolsimporting) should be implemented and the new eye tracker added to
the `glassesTools.eyetracker.EyeTracker` `Enum`.

## Converting recordings to a common data format
glassesTools includes functionality to import data from the supported eye trackers to a [common data format](#commondataformat). Before data from some of these eye trackers can be imported, the recordings may have to be prepared. These steps are described here first, after which the [`glassesTools.importing`](#glassestoolsimporting) API will be outlined.

### Required preprocessing outside glassesTools
For some eye trackers, the recording delivered by the eye tracker's recording unit or software can be directly imported using
glassesTools. Recordings from some other eye trackers however require some steps to be performed in the manufacturer's
software before they can be imported using glassesTools. These are:
- *Pupil Labs eye trackers*: Recordings should either be preprocessed using Pupil Player (*Pupil Core* and *Pupil Invisible*),
  Neon Player (*Pupil Neon*) or exported from Pupil Cloud (*Pupil Invisible* and *Pupil Neon*).
  - Using Pupil Player (*Pupil Core* and *Pupil Invisible*) or Neon player (*Pupil Neon*): Each recording should 1) be opened
    in Pupil/Neon Player, and 2) an export of the recording (`e` hotkey) should be run from pupil player. Make sure to disable the
    `World Video Exporter` in the `Plugin Manager` before exporting, as the exported video is not used by glassesValidator and takes a long time to create. Note that importing a Pupil/Neon Player export of a Pupil Invisibl/Neone recording may require an internet connection. This is used to retrieve the scene camera calibration from Pupil Lab's servers in case the recording does not have
    a `calibration.bin` file.
  - Using Pupil Cloud (*Pupil Invisible* and *Pupil Neon*): Export the recordings using the `Timeseries data + Scene video` action.
  - For the *Pupil Core*, for best results you may wish to do a scene camera calibration yourself, see https://docs.pupil-labs.com/core/software/pupil-capture/#camera-intrinsics-estimation.
    If you do not do so, a generic calibration will be used by Pupil Capture during data recording, by Pupil Player during data
    analysis and by glassesValidator, which may result in incorrect accuracy values.
- *SMI ETG*: For SMI ETG recordings, access to BeGaze is required and the following steps should be performed:
  - Export gaze data: `Export` -> `Legacy: Export Raw Data to File`.
    - In the `General` tab, make sure you select the following:
      - `Channel`: enable both eyes
      - `Points of Regard (POR)`: enable `Gaze position`, `Eye position`, `Gaze vector`
      - `Binocular`: enable `Gaze position`
      - `Misc Data`: enable `Frame counter`
      - disable everything else
    - In the Details tab, set:
      - `Decimal places` to 4
      - `Decimal separator` to `point`
      - `Separator` to `Tab`
      - enable `Single file output`
    - This will create a text file with a name like `<experiment name>_<participant name>_<number> Samples.txt`
      (e.g. `005-[5b82a133-6901-4e46-90bc-2a5e6f6c6ea9]_005_001 Samples.txt`). Move this file/these files to the
      recordings folder and rename them. If, for instance, the folder contains the files `005-2-recording.avi`,
      `005-2-recording.idf` and `005-2-recording.wav`, amongst others, for the recording you want to process,
      rename the exported samples text file to `005-2-recording.txt`.
  - Export the scene video:
    - On the Dashboard, double click the scene video of the recording you want to export to open it in the scanpath tool.
    - Right click on the video and select settings. Make the following settings in the `Cursor` tab:
      - set `Gaze cursor` to `translucent dot`
      - set `Line width` to 1
      - set `Size` to 1
    - Then export the video, `Export` -> `Export Scan Path Video`. In the options dialogue, make the following settings:
      - set `Video Size` to the maximum (e.g. `(1280,960)` in my case)
      - set `Frames per second` to the framerate of the scene camera (24 in my case)
      - set `Encoder` to `Performance [FFmpeg]`
      - set `Quality` to `High`
      - set `Playback speed` to `100%`
      - disable `Apply watermark`
      - enable `Export stimulus audio`
      - finally, click `Save as`, navigate to the folder containing the recording, and name it in the same format as the
        gaze data export file we created above but replacing `recording` with `export`, e.g. `005-2-export.avi`.

### `glassesTools.importing`
|function|inputs|description|
| --- | --- | --- |
|`get_recording_info()`|<ol><li>[`source_dir`](#common-input-arguments)</li><li>`device`: `glassesTools.eyetracker.EyeTracker`</li></ol>|Determine if provided path contains a recording/recordings made with the specified eye tracker (`device`) and if so, get info about these recordings.|
|`do_import()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`device`: `glassesTools.eyetracker.EyeTracker`</li><li>`rec_info`</li></ol>|Import the specified recording to `output_dir`. Either `device` or `rec_info` must be specified. Does nothing if directory does not contain a recording made with the specified eye tracker.|
|  |  |  |
|`adhawk_mindlink()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li><li>`cam_cal_file`: OpenCV XML file containing a camera calibration to be used when processing this recording. Optional. If not provided a default calibration provided by AdHawk is used.</li></ol>|Import an AdHawk MindLink recording to a subdirectory of `output_dir`. Does nothing if directory does not contain an AdHawk MindLink recording. `rec_info` is optional.|
|`pupil_core()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li></ol>|Import a Pupil Core recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Pupil Core recording. `rec_info` is optional.|
|`pupil_invisible()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li></ol>|Import a Pupil Invisible recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Pupil Invisible recording. `rec_info` is optional.|
|`pupil_neon()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li></ol>|Import a Pupil Neon recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Pupil Neon recording. `rec_info` is optional.|
|`SeeTrue_STONE()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li><li>`cam_cal_file`: OpenCV XML file containing a camera calibration to be used when processing this recording. Optional. If not provided a default calibration provided by SeeTrue is used.</li></ol>|Import a SeeTrue recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a SeeTrue recording. `rec_info` is optional.|
|`SMI_ETG()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li></ol>|Import a SMI ETG recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a SMI ETG 1 or 2 recording. `rec_info` is optional.|
|`tobii_G2()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li></ol>|Import a Tobii Pro Glasses 2 recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Tobii Pro Glasses 2 recording. `rec_info` is optional.|
|`tobii_G3()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`rec_info`</li></ol>|Import a Tobii Pro Glasses 3 recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Tobii Pro Glasses 3 recording. `rec_info` is optional.|

#### Common input arguments
|argument|description|
| --- | --- |
|`source_dir`|Path to directory containing one (or for some eye trackers potentially multiple) eye tracker recording(s) as stored by the eye tracker's recording hardware or software.|
|`working_dir`, or `output_dir`|Path to a glassesValidator recording directory. In the case of output_dir, it is the directory the functions in `glassesTools.importing` will import the recording to.|

### `glassesTools.video_utils`
Besides the gaze data, the common data format also contains a file with timestamps for each frame in the eye tracker's scene camera video. This file is created automatically when importing a recording using the functions in [`glassesTools.importing`](#glassestoolsimporting), but can also be created manually using the function `glassesTools.video_utils.get_frame_timestamps_from_video`. The duration of a video file can furthermore be determined using `glassesTools.video_utils.get_video_duration` and gaze data can be associated with frames in the scene camera video by means of the gaze and video timestamp using `glassesTools.video_utils.timestamps_to_frame_number`.

## Common data format
The common data format of glassesTools contains the following files:
|file|description|
| --- | --- |
|`worldCamera.mp4`|[`glassesValidator.preprocess`](#glassesvalidatorpreprocess) functions|(Optional) copy of the scene camera video.|
|`recording_info.json`|Information about the recording.|
|`gazeData.tsv`|Gaze data cast into a common format understood by glassesValidator.|
|`frameTimestamps.tsv`|Timestamps for each frame in the scene camera video.|
|`calibration.xml`|Camera calibration parameters for the scene camera.|

### Recording info
`glassesTools.recording`

### Gaze data
#### Head-referenced gaze data
`glassesTools.gaze_headref`
#### World-referenced gaze data
`glassesTools.gaze_worldref`

## Processing
`glassesTools.aruco`, `glassesTools.marker`, `glassesTools.ocv`, `glassesTools.plane`