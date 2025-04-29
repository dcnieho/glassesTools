# -*- coding: utf-8 -*-
from psychopy import visual, monitors, core, event, tools
import numpy as np
import pandas as pd
import json
import traceback
import random
from collections import defaultdict
import cv2

class ABCFixPoint:
    def __init__(self, win, outer_diameter=0.7, inner_diameter=0.1,
                 outer_color='black', inner_color='white', units='degFlatPos'):
        self.outer_dot      = visual.Circle(win, fillColor = outer_color,
                                            radius = outer_diameter/2,
                                            units = units)
        self.inner_dot      = visual.Circle(win, fillColor = outer_color,
                                            radius = inner_diameter/2,
                                            units = units)
        self.line_vertical  = visual.Rect(win, fillColor=inner_color,
                                          width=inner_diameter, height=outer_diameter,
                                          units = units)
        self.line_horizontal= visual.Rect(win, fillColor=inner_color,
                                          width=outer_diameter, height=inner_diameter,
                                          units = units)

    def set_size(self, size: float):
        self.outer_dot.radius       = size / 2
        self.line_vertical.size     = (self.line_vertical.width, size)
        self.line_horizontal.size   = (size, self.line_horizontal.height)

    def set_pos(self, pos):
        self.outer_dot.pos          = pos
        self.inner_dot.pos          = pos
        self.line_vertical.pos      = pos
        self.line_horizontal.pos    = pos

    def get_pos(self):
        return self.outer_dot.pos

    def get_size(self):
        return self.outer_dot.size

    def draw(self):
        self.outer_dot.draw()
        self.line_vertical.draw()
        self.line_horizontal.draw()
        self.inner_dot.draw()

class SegmentationMarker:
    def __init__(self, win: visual.Window, refresh_rate: int, duration: float,
                 marker_size: float, units: str, margin: float, background_color: str):
        self.win = win
        self.refresh_rate = refresh_rate
        self.duration = duration
        self.size = marker_size
        self.units = units
        self.margin = margin
        self.background_color = background_color

        self._marker    = visual.ImageStim(win, image=None, size=self.size, units=self.units)
        self._marker_bg = visual.Rect(win,
                                      units =self.units,
                                      width =self.size*(1+self.margin*2),
                                      height=self.size*(1+self.margin*2),
                                      fillColor=self.background_color)

    def draw(self, m_id: int):
        check_escape(self.win)
        self._marker.image = get_aruco_marker(m_id, self.size, self.units, self.win)
        n_segment_marker_frames = int(self.duration*self.refresh_rate)
        for _ in range(n_segment_marker_frames):
            self._marker_bg.draw()
            self._marker.draw()
            self.win.flip()

def read_coord_file(file):
    return pd.read_csv(file, dtype=defaultdict(lambda: np.float32, ID='int32', color='str')).dropna(axis=0, how='all').set_index('ID')

_aruco_dict: cv2.aruco.Dictionary = None
_aruco_border_bits: int = None
def load_aruco_dict(aruco_dict_name: str, border_bits: int):
    global _aruco_dict
    global _aruco_border_bits
    str_to_dict: dict[str, int] = {k: getattr(cv2.aruco,k) for k in ['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000', 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000', 'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16H5', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36H11', 'DICT_ARUCO_MIP_36H12']}

    if aruco_dict_name not in str_to_dict:
        possible = '"'+'", "'.join(str_to_dict.keys())+'"'
        raise ValueError(f'ArUco dictionary with name "{aruco_dict_name}" is not known. Possible values are: {possible}.')
    if border_bits<1:
        raise ValueError('The number of border bits for ArUco markers must be 1 or higher.')

    _aruco_dict = cv2.aruco.getPredefinedDictionary(str_to_dict[aruco_dict_name])
    _aruco_border_bits = border_bits

def get_aruco_marker(m_id: int, size: float, units: str, win: visual.Window):
    if _aruco_dict is None or _aruco_border_bits is None:
        RuntimeError('You must call load_aruco_dict() before you try to load an ArUco marker')
    size = tools.monitorunittools.convertToPix(np.array([-.5*size,.5*size]),np.array([0.,0.]),units,win)
    size = int(size[1]-size[0])
    # NB: flipud because PsychoPy draws images loaded from memory upside down
    return np.flipud(cv2.aruco.generateImageMarker(_aruco_dict, m_id, size, borderBits=_aruco_border_bits).astype(np.float32)/255.*2-1)

def check_escape(win: visual.Window):
    if 'escape' in event.getKeys():
        win.close()
        core.quit()


def prepare_validation(win: visual.Window, config: dict, screen_config: dict):
    # get markers and target positions
    target_positions   = read_coord_file(config["targets"]["file"])
    fiducial_positions = read_coord_file(config["markers"]["file"])

    # check there are no overlapping ArUco markers
    indicator_IDs = []
    for ri,_ in enumerate(config["markers"]["replace_IDs"]):
        off = config["markers"]["replace_ID_start"]+ri*config["markers"]["replace_ID_offset"]
        indicator_IDs += [i+off for i in target_positions.index]
    if duplicates:=set(indicator_IDs) & set(fiducial_positions.index):
        raise ValueError(f"For the current setup, marker IDs specified in marker file {config['markers']['file']} would also appear as target indicator markers. Adapt the marker positions file or config->markers->replace_ID_start, config->markers->replace_ID_offset and replace_IDs. Duplicates: {sorted(list(duplicates))}.")

    # Create a background with ArUco markers and circle placeholders where target will appear
    stimList = []
    # draw all ArUco markers
    # NB ArUco markers are drawn using the 'deg' and not the 'degFlatPos' coordinate system so
    # that these markers are evenly spaced and not more eccentric than necessary
    for m_id, marker in fiducial_positions.iterrows():
        aruco_im = visual.ImageStim(win, image=None, size=config["markers"]["size"], units=config["markers"]["units"])
        aruco_bg = visual.Rect(win, units=config["markers"]["units"],
                               width =config["markers"]["size"]*(1+config["markers"]["margin"]*2),
                               height=config["markers"]["size"]*(1+config["markers"]["margin"]*2),
                               fillColor=config["markers"]["background_color"])

        aruco_bg.pos = [marker.x, marker.y]
        aruco_im.image = get_aruco_marker(m_id, config["markers"]["size"], config["markers"]["units"], win)
        aruco_im.pos = [marker.x, marker.y]

        stimList.append(aruco_bg)
        stimList.append(aruco_im)

        aruco_bg.draw()
        aruco_im.draw()

    # draw placeholder circles at all target positions
    if (use_file_color:='color' in target_positions.columns) or config["targets"]["placeholder"]["color"]:
        # if color specified in the the target_positions file, use that. Else, fall back on placeholder color
        # specified in config. If neither is specified, do not draw placeholders
        for _, target in target_positions.iterrows():
            circle = visual.Circle(win, radius=config["targets"]["placeholder"]["diameter"]/2, units=config["targets"]["units"], fillColor=target.color if use_file_color else config["targets"]["placeholder"]["color"])
            circle.pos = [target.x, target.y]
            stimList.append(circle)
            circle.draw()

    # Get screenshot of background, so that we can draw unchanging things at once
    background = visual.ImageStim(win, visual.BufferImageStim(win, stim=stimList).image)    # because https://github.com/psychopy/psychopy/issues/840
    return {'background': background, 'target_positions': target_positions, 'fiducial_positions': fiducial_positions}

def show_validation(win: visual.Window, config: dict, refresh_rate: int, task_vars: dict):
    # prepare visual objects
    fixation_target = ABCFixPoint(win, outer_color=config["targets"]["look"]["outer_color"], inner_color=config["targets"]["look"]["inner_color"], units=config["targets"]["units"])
    aruco_ims = [visual.ImageStim(win, image=None, size=config["markers"]["size"], units=config["markers"]["units"]) for _ in config["markers"]["replace_IDs"]]
    if (have_cue:=config["targets"]["cue"]["color"] is not None):
        cue = visual.Circle(win, radius=config["targets"]["cue"]["diameter"]/2, units=config["targets"]["units"], fillColor=config["targets"]["cue"]["color"])

    # prepare target order
    if config["targets"]["first_ID"] not in task_vars['target_positions'].index:
        raise ValueError(f'Target ID set in setup.json->targets->first_ID ({config["targets"]["first_ID"]}) is not in the targets file {config["targets"]["file"]}')
    ts = [t for t in task_vars['target_positions'].index.to_list() if t!=config["targets"]["first_ID"]]
    random.shuffle(ts)
    targets = [config["targets"]["first_ID"]] + ts

    # prepare task parameters
    n_shrink_frames = int(config["targets"]["shrink"]["duration"]*refresh_rate)
    shrink_sizes    = np.linspace(config["targets"]["look"]["diameter_max"], config["targets"]["look"]["diameter_min"], n_shrink_frames)
    n_capture_frames= int(config["targets"]["duration"]*refresh_rate)

    # show fixation target sequence
    old_pos = task_vars['target_positions'].loc[targets[0],['x','y']].to_numpy()
    for i in targets:
        check_escape(win)

        # Move target to new position
        pos   = task_vars['target_positions'].loc[i,['x','y']].to_numpy()
        d     = np.hypot(old_pos[0]-pos[0], old_pos[1]-pos[1])
        old_pos_pix = tools.monitorunittools.convertToPix(np.array([0.,0.]),old_pos,config["targets"]["units"],win)
        pos_pix     = tools.monitorunittools.convertToPix(np.array([0.,0.]),    pos,config["targets"]["units"],win)
        d_pix = np.hypot(old_pos_pix[0]-pos_pix[0], old_pos_pix[1]-pos_pix[1])
        # Target should move at constant speed regardless of distance to cover, duration contains time to move
        # over width of whole screen. Adjust time to proportion of screen width covered by current move
        move_duration = max(config["targets"]["move"]["min_duration"], config["targets"]["move"]["duration"]*d_pix/win.size[0])
        n_move_frames = int(move_duration*refresh_rate)
        if config["targets"]["move"]["move_with_acceleration"]:
            accel   = 0 if not d else d/(move_duration/2)**2    # solve x=.5*a*t^2 for a, use d/2 for x
            moveVec = [0.,0.] if not d else [(x-y)/d for x,y in zip(pos,old_pos)]
            def calc_pos(frac):
                if frac<.5:
                    return [p+m*.5*accel*(frac*move_duration)**2 for p,m in zip(old_pos,moveVec)]
                else:
                    # implement deceleration by accelerating from the other side in backward time
                    return [p-m*.5*accel*((1-frac)*move_duration)**2 for p,m in zip(pos,moveVec)]
            tar_pos = [calc_pos(x) for x in np.linspace(0., 1., n_move_frames)]
        else:
            x_tar_pos = np.linspace(old_pos[0], pos[0], n_move_frames)
            y_tar_pos = np.linspace(old_pos[1], pos[1], n_move_frames)
            tar_pos = [[x,y] for x,y in zip(x_tar_pos,y_tar_pos)]

        # two markers indicate which target is shown (two for redundancy)
        for ri,r in enumerate(config["markers"]["replace_IDs"]):
            m_id = config["markers"]["replace_ID_start"]+ri*config["markers"]["replace_ID_offset"]+i
            aruco_ims[ri].image = get_aruco_marker(m_id, config["markers"]["size"], config["markers"]["units"], win)
            aruco_ims[ri].pos   = task_vars['fiducial_positions'].loc[r,['x','y']].tolist()

        # move to next target
        fixation_target.set_size(config["targets"]["move"]["diameter"])
        cue.pos = pos
        for p in tar_pos:
            task_vars['background'].draw()
            if have_cue:
                cue.draw()
            fixation_target.set_pos(p)
            fixation_target.draw()
            win.flip()

        # shrink
        for s in shrink_sizes:
            task_vars['background'].draw()
            fixation_target.set_size(s)
            fixation_target.draw()
            win.flip()

        # fixation period
        for _ in range(n_capture_frames):
            task_vars['background'].draw()
            [a.draw() for a in aruco_ims]
            fixation_target.draw()
            win.flip()

        old_pos = pos

def run_validation(win: visual.Window, config: dict):
    # prepare fiducials
    load_aruco_dict(config["aruco"]["dict"], config["aruco"]["border_bits"])

    # prepare trial segmentation
    segmenter = SegmentationMarker(win, config["screen"]["refresh_rate"], config["segment_marker"]["duration"],
                                   config["segment_marker"]["size"], config["segment_marker"]["units"], config["segment_marker"]["margin"], config["segment_marker"]["background_color"])
    # prepare instruction
    textstim = visual.TextStim(win, text="", height=config["instruction_text"]["height"], color=config["instruction_text"]["color"], wrapWidth=9999.)


    # validation instruction
    textstim.text = 'Follow the fixation target around, and keep fixating the center of it\n\n(Press the spacebar to start)'
    textstim.draw()
    win.flip()
    event.waitKeys()

    # prepare validation
    task_vars = prepare_validation(win, config["validation"], config["screen"])

    # run validation
    for r in range(config["validation"]["n_repetitions"]):
        # signal start
        for m_id in config["validation"]["segment_marker"]["start_IDs"]:
            segmenter.draw(m_id)

        # show targets
        show_validation(win, config["validation"], config["screen"]["refresh_rate"], task_vars)

        # signal end
        for m_id in config["validation"]["segment_marker"]["end_IDs"]:
            segmenter.draw(m_id)

        # blank screen to separate validation intervals
        if r!=config["validation"]["n_repetitions"]-1:
            win.flip()
            core.wait(1.)

def main():
    # read protocol setup
    with open("setup.json") as fp:
        config = json.load(fp)

    try:
        # Open window, check
        mon = monitors.Monitor('temp')
        mon.setWidth(config["screen"]["width"])
        mon.setDistance(config["screen"]["viewing_distance"])
        mon.setSizePix(config["screen"]["resolution"])
        win = visual.Window(monitor=mon,
                            fullscr=True,
                            color=config["screen"]["background_color"],
                            screen=config["screen"]["which_monitor"] or 0,
                            units=config["screen"]["units"],
                            allowGUI=False,
                            multiSample=True,
                            numSamples=4,
                            infoMsg='')
        win.mouseVisible = False
        if not all((x==y for x,y in zip(win.size,config["screen"]["resolution"]))):
            raise RuntimeError(f'expected resolution of {config["screen"]["resolution"]}, but got {win.size}')
        if 1/win.monitorFramePeriod<config["screen"]["refresh_rate"]-2 or 1/win.monitorFramePeriod>config["screen"]["refresh_rate"]+2:  # check within 2 Hz
            raise RuntimeError(f'expected framerate of {config["screen"]["refresh_rate"]}, but got {1/win.monitorFramePeriod}')

        run_validation(win, config)

    except Exception as e:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        print("".join(tb_lines))
    finally:
        if 'win' in locals():
            win.close()

if __name__=="__main__":
    main()