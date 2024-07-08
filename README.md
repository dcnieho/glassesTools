[![Downloads](https://static.pepy.tech/badge/glassestools)](https://pepy.tech/project/glassestools)
[![PyPI Latest Release](https://img.shields.io/pypi/v/glassesTools.svg)](https://pypi.org/project/glassesTools/)
[![image](https://img.shields.io/pypi/pyversions/glassesTools.svg)](https://pypi.org/project/glassesTools/)

# GlassesTools v1.5.5
Tools for processing wearable eye tracker recordings.

If you use this package or any of the code in this repository, please cite:<br>
[Niehorster, D.C., Hessels, R.S., Benjamins, J.S., Nyström, M. and Hooge, I.T.C. (2023). GlassesValidator:
A data quality tool for eye tracking glasses. Behavior Research Methods. doi: 10.3758/s13428-023-02105-5](https://doi.org/10.3758/s13428-023-02105-5)


# API

### glassesTools.importing
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


### Common input arguments
|argument|description|
| --- | --- |
|`source_dir`|Path to directory containing one (or for some eye trackers potentially multiple) eye tracker recording(s) as stored by the eye tracker's recording hardware or software.|
|`working_dir`, or `output_dir`|Path to a glassesValidator recording directory. In the case of output_dir, it is the directory the functions in `glassesTools.importing` will import the recording to.|
