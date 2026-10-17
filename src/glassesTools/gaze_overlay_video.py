import numpy as np
import cv2
import pathlib
import shutil
import os
import subprocess
import typing
from enum import Enum, auto

from . import gaze_headref, naming, ocv, timestamps, video_utils, _has_GUI
if _has_GUI:
    from .gui import video_player
else:
    # stub out video_player as a class so type annotations below do not fail
    class video_player:
        @property
        def GUI(self) -> typing.Any: ...

class Status(Enum):
    Ok = auto()
    Skip = auto()
    Finished = auto()

class VideoMaker:
    def __init__(self, output_path: str|pathlib.Path, video_file: str|pathlib.Path, frame_timestamp_file: str|pathlib.Path|timestamps.VideoTimestamps, camera_calibration_file: str|pathlib.Path|ocv.CameraParams, gaze_data_file: str|pathlib.Path|dict[int,list[gaze_headref.Gaze]], pixel_format: str = 'yuv420p'):
        self.src_video   = pathlib.Path(video_file)
        self.output_path = pathlib.Path(output_path)
        if self.output_path.is_dir():
            self.output_path /= naming.gaze_overlay_video_file
        self.video_ts   = frame_timestamp_file if isinstance(frame_timestamp_file,timestamps.VideoTimestamps) else timestamps.VideoTimestamps(frame_timestamp_file)
        self.video      = ocv.CV2VideoReader(self.src_video, self.video_ts.timestamps)
        self.cam_params = camera_calibration_file if isinstance(camera_calibration_file,ocv.CameraParams) else ocv.CameraParams.read_from_file(camera_calibration_file)
        self.gaze       = gaze_data_file if isinstance(gaze_data_file,dict) else gaze_headref.read_dict_from_file(gaze_data_file, ts_column_suffixes=['VOR',''])[0]

        self._cache: tuple[Status, tuple[np.ndarray, int, float]] = None  # self._cache[1][1] is frame number

        self.gui                    : video_player.GUI  = None
        self.has_gui                                    = False
        self.progress_updater       : typing.Callable[[], None] = None

        self.do_visualize                               = False
        self.sub_pixel_fac                              = 8

        self.gaze_styles            : gaze_headref.OverlayGazeStyles = gaze_headref.DEFAULT_OVERLAY_GAZE_STYLES

        self._first_frame                               = True

        # set up output video
        self._fps= 1000/self.video_ts.get_IFI()
        self._res= (int(self.video.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)),int(self.video.get_prop(cv2.CAP_PROP_FRAME_WIDTH)))
        self._vid_writer = video_utils.create_video_writer(self.output_path, pixel_format, self._res[1], self._res[0], self._fps)

    def __del__(self):
        if self.has_gui:
            self.gui.stop()

    def attach_gui(self, gui: video_player.GUI|None, window_id: int = None):
        self.gui                    = gui
        self.has_gui                = self.gui is not None
        self.do_visualize           = self.has_gui

        if self.has_gui:
            self.gui.set_show_timeline(True, self.video_ts, None, window_id)
            self.gui.set_frame_size(self._res)

    def set_visualize_on_frame(self, do_visualize: bool):
        self.do_visualize           = do_visualize

    def set_progress_updater(self, progress_updater: typing.Callable[[], None]):
        self.progress_updater = progress_updater

    def set_gaze_styles(self, styles: gaze_headref.OverlayGazeStyles | None = None):
        self.gaze_styles = styles or gaze_headref.DEFAULT_OVERLAY_GAZE_STYLES

    def _read_frame(self, wanted_frame_idx:int|None = None) -> tuple[Status, tuple[np.ndarray, int, float]]:
        if self._first_frame and self.has_gui:
            self.gui.set_playing(True)
            self._first_frame = False

        if wanted_frame_idx is not None and self._cache is not None and self._cache[1][1]==wanted_frame_idx:
            return self._cache

        should_exit, frame, frame_idx, frame_ts, frame_info = self.video.read_frame(report_gap=True, wanted_frame_idx=wanted_frame_idx)

        if should_exit:
            self._cache = Status.Finished, (None, None, None)
            return self._cache

        if self.has_gui:
            requests = self.gui.get_requests()
            for r,_ in requests:
                if r=='exit':   # only requests we need to handle
                    self._cache = Status.Finished, (None, None, None)
                    return self._cache
                if r=='close':
                    self.has_gui = False
                    self.gui.stop()

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.set(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        if frame is None:
            # we don't have a valid frame or nothing to do, continue to next
            self._cache = Status.Skip, (frame, frame_idx, frame_ts, frame_info)
            return self._cache

        self._cache = Status.Ok, (frame, frame_idx, frame_ts, frame_info)
        return self._cache

    def process_one_frame(self) -> Status:
        status, (frame, frame_idx, frame_ts, frame_info) = self._read_frame()
        if status==Status.Finished:
            return status
        if frame is None:
            # get black frame
            frame = np.zeros(self._res,'uint8')

        # draw gaze
        if 'offset_x' in frame_info and 'offset_y' in frame_info:
            ROI_offset = (frame_info['offset_x'], frame_info['offset_y'])
        else:
            ROI_offset = (0,0)
        if frame_idx in self.gaze:
            for g in self.gaze[frame_idx]:
                  g.draw(frame, self.cam_params, sub_pixel_fac=self.sub_pixel_fac, styles=self.gaze_styles, ROI_offset=ROI_offset)

        # now that all processing is done, handle gui
        if self.has_gui:
            self.gui.update_image(frame, frame_ts/1000., frame_idx)

        # submit frame to be encoded
        video_utils.write_video_frame(self._vid_writer, frame, frame_idx/self._fps)

        if self.progress_updater:
            self.progress_updater()

    def finish_video(self):
        # nothing more to show, so clean up GUI if any
        if self.has_gui:
            self.gui.stop()
        self._vid_writer.close()
        # if ffmpeg is on path, add audio to scene video
        if (shutil.which('ffmpeg') is not None) and (shutil.which('ffprobe') is not None):
            # check if source file has audio
            command = ['ffprobe',
               '-loglevel', 'error',
               '-select_streams', 'a',
               '-show_entries', 'stream=codec_type',
               '-of', 'csv=p=0',
               f'{self.src_video}']
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            if not err and out.decode().strip()=='audio':
                # file has audio, lets go
                # move file to temp name
                tempName = self.output_path.with_stem(self.output_path.stem + '_temp')
                shutil.move(self.output_path, tempName)

                # add audio
                cmds = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', f'"{tempName}"', '-i', f'"{self.src_video}"', '-vcodec', 'copy']
                # determine if can copy audio or need to transcode
                # we transcode if source file is not mp4-style, to be safe. Some audio formats that are allowed in
                # avi and mkv are not allowed in mp4
                if video_utils.is_isobmmf(self.src_video):
                    # copy. else, specifying nothing, it will be transcoded
                    cmds.extend(['-acodec', 'copy'])
                cmds.extend(['-map', '0:v:0', '-map', '1:a:0?', '-shortest', f'"{self.output_path}"'])
                os.system(' '.join(cmds))

                # clean up
                if self.output_path.exists():
                    tempName.unlink(missing_ok=True)
                else:
                    # something failed. Put file without audio back under output name
                    shutil.move(tempName, self.output_path)

    def process_video(self):
        while True:
            status = self.process_one_frame()
            if status==Status.Finished:
                break
            if status==Status.Skip:
                continue

        self.finish_video()