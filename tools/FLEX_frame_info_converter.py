# this script requires the following python packages:
# pip install pandas

# It searches the entire directory tree at the provided for video files with the indicated extension, and for each video file found, it looks for a .json file as provided by SWAG with the same name (but _info.json extension) containing metadata about the recording.
# The script transforms the info in this JSON file into a tsv file with a format compatible with the VideoInfo class in glassesTools.

import pathlib
import pandas as pd
import re

root        = r"\\et-nas.humlab.lu.se\FLEX\2026_listing\take 2 20260326\recordings_20260327"
extension   = '.mp4'



def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

# find all video files in the directory tree
video_files = list(pathlib.Path(root).rglob(f'*{extension}'))
for video_file in video_files:
    # construct the path to the corresponding JSON file
    json_file = video_file.parent / (video_file.stem + '_info.json')

    # skip if the JSON file does not exist
    if not json_file.exists():
        continue

    # read the JSON file into a pandas DataFrame
    df = pd.read_json(json_file)
    df.index.name = 'frame_idx'
    # for the other columns, turn camelcase names into snake_case
    df.columns = [camel_to_snake(col) for col in df.columns]

    # save the DataFrame as a TSV file with the same name as the video file but with .tsv extension
    tsv_file = video_file.parent / (video_file.stem + '_frame_info.tsv')
    df.to_csv(tsv_file, sep='\t', na_rep='nan', float_format="%.16f")
    print(f'Converted {json_file} to {tsv_file}')