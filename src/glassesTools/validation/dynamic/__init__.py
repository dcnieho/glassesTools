import pathlib
import importlib.resources
import shutil
import json
import math

from ..config import get_markers, get_targets

def deploy_setup_and_script(output_dir: str|pathlib.Path, overwrite=False) -> list[str]:
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f'The requested directory "{output_dir}" does not exist')

    # copy over all files
    not_copied: list[str] = []
    for r in ['markerPositions.csv', 'targetPositions.csv', 'setup.json', 'stim.py']:
        out_file = output_dir/r
        if out_file.exists() and not overwrite:
            not_copied.append(r)
            continue
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, out_file)
    return not_copied

def _get_extent(extent: float, distance: float, psychopy_unit: str) -> float:
    match psychopy_unit:
        case 'cm':
            return extent
        case 'deg' | 'degFlatPos':
            # sizes are not corrected for flat screen when unit is degFlatPos, so same as deg
            # for factor, see comment in _get_position
            return extent * distance*0.017455
        case _:
            raise ValueError(f'PsychoPy unit {psychopy_unit} is not understood')

def _get_position(position: tuple[float,float], distance: float, psychopy_unit: str) -> tuple[float,float]:
    match psychopy_unit:
        case 'cm':
            return position
        case 'deg':
            # 1 deg (centered) at 1 cm is approximately pi/180 (2*tand(.5)), seems to be the logic for PsychoPy
            # this is then apparently hardcoded as 0.017455, which is rounded off a bit wrong...
            fac = distance * 0.017455
            return tuple((p*fac for p in position))
        case 'degFlatPos':
            # positions corrected for flat screen, sizes not
            x,y = (math.radians(x) for x in position)
            return (math.hypot(distance, math.tan(y) * distance) * math.tan(x),
                    math.hypot(distance, math.tan(x) * distance) * math.tan(y))
        case _:
            raise ValueError(f'PsychoPy unit {psychopy_unit} is not understood')

def setup_to_automatic_coding(config_dir: str|pathlib.Path|None, file_name: str='setup.json') -> dict[str, int|list[int]]:
    # load config used to run validation
    # if no config dir specified, load default
    if config_dir is not None:
        with open(pathlib.Path(config_dir)/file_name, 'r') as f:
            setup = json.load(f)
    else:
        with importlib.resources.open_text(__package__, file_name) as f:
            setup = json.load(f)

    out = setup["validation"]["segment_marker"]
    out["border_bits"] = setup["segment_marker"]["border_bits"]
    return out

def setup_to_plane_config(output_dir: str|pathlib.Path, config_dir: str|pathlib.Path|None, file_name: str='setup.json'):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f'The requested directory "{output_dir}" does not exist')

    # load config used to run validation
    # if no config dir specified, load default
    if config_dir is not None:
        with open(pathlib.Path(config_dir)/file_name, 'r') as f:
            setup = json.load(f)
    else:
        with importlib.resources.open_text(__package__, file_name) as f:
            setup = json.load(f)

    # get units in which positions are expressed, and viewing distance which may be needed for converting them
    marker_units = setup["validation"]["markers"]["units"] if "units" in setup["validation"]["markers"] else setup["screen"]["units"]
    target_units = setup["validation"]["targets"]["units"] if "units" in setup["validation"]["targets"] else setup["screen"]["units"]
    dist = setup["screen"]["viewing_distance"]
    # load target and marker positions (default ones from package if config dir is not specified)
    target_positions = get_targets(config_dir, setup["validation"]["targets"]["file"], __package__)
    marker_positions = get_markers(config_dir, setup["validation"]["markers"]["file"], __package__)
    # convert positions to cm
    target_positions[['x','y']] = target_positions.apply(lambda r: _get_position((r['x'],r['y']), dist, target_units), axis=1, result_type='expand')
    marker_positions[['x','y']] = marker_positions.apply(lambda r: _get_position((r['x'],r['y']), dist, marker_units), axis=1, result_type='expand')
    # negate y as positive y is upward for PsychoPy script but downward for our logic
    target_positions['y'] = -target_positions['y']
    marker_positions['y'] = -marker_positions['y']
    # Marker and target positions are centers, and relative to center of screen.
    # Our positions should have (0,0) in the top-left. Shift (taking sizes into account so who markers and targets fit on the reference image)
    t_size= _get_extent(setup["validation"]["targets"]["look"]["diameter_max"], dist, target_units)
    m_size= _get_extent(setup["validation"]["markers"]["size"]                , dist, marker_units)
    min_t = target_positions[['x','y']].min()-t_size/2
    min_m = marker_positions[['x','y']].min()-m_size/2
    x_min = min(min_t['x'],min_m['x'])
    y_min = min(min_t['y'],min_m['y'])
    target_positions['x']-=x_min
    marker_positions['x']-=x_min
    target_positions['y']-=y_min
    marker_positions['y']-=y_min

    # get plane size
    max_t = target_positions[['x','y']].max()+t_size/2
    max_m = marker_positions[['x','y']].max()+m_size/2
    x_max = max(max_t['x'],max_m['x'])
    y_max = max(max_t['y'],max_m['y'])

    # write appropriate validationSetup.txt file
    valSetup = {
        'distance': dist,
        'mode': 'cm',
        'markerBorderBits': setup["validation"]["markers"]["border_bits"],
        'markerSide': m_size,
        'markerPosFile': 'markerPositions.csv',
        'targetPosFile': 'targetPositions.csv',
        'targetType': 'Thaler',
        'targetDiameter': t_size,
        'showGrid': 0,
        'gridCols': x_max,
        'gridRows': y_max,
        'minNumMarkers': 3,
        'centerTarget': setup["validation"]["targets"]["center_target"],
        'referencePosterSize': 1920
    }

    # add marker indicator(s) as columns to target file
    for i,_ in enumerate(setup["validation"]["markers"]["replace_IDs"]):
        IDs = target_positions.index.to_numpy()+setup["validation"]["markers"]["replace_ID_start"]+i*setup["validation"]["markers"]["replace_ID_offset"]
        target_positions[f'marker_{i}'] = IDs

    # store everything to files
    with open(output_dir/'validationSetup.txt','w') as f:
        for key,value in valSetup.items():
            f.write(f'{key} = {value}\n')
    target_positions.to_csv(output_dir/valSetup['targetPosFile'], float_format='%.8f')
    marker_positions.to_csv(output_dir/valSetup['markerPosFile'], float_format='%.8f')
