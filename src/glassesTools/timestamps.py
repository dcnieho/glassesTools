import numpy as np
import pandas as pd
import datetime
import warnings
import bisect

from . import utils

class Timestamp:
    def __init__(self, unix_time: int | float, format="%Y-%m-%d %H:%M:%S"):
        self.format = format
        self.display = ""
        self.value = 0
        self.update(unix_time)

    def update(self, unix_time: int | float):
        self.value = int(unix_time)
        if self.value == 0:
            self.display = ""
        else:
            self.display = datetime.datetime.fromtimestamp(unix_time).strftime(self.format)

utils.register_type(utils.CustomTypeEntry(Timestamp,'__Timestamp__',lambda x: x.value, lambda x: Timestamp(x)))


# for reading video timestamp files
class VideoTimestamps:
    def __init__(self, fileName):
        self.timestamp_dict : dict[int,float] = {}
        self.indices        : list[int] = []
        self.timestamps     : list[float] = []
        self._ifi           : float = None

        df = pd.read_csv(fileName, delimiter='\t', index_col='frame_idx')
        self.timestamp_dict = df.to_dict()['timestamp']

        df = df.reset_index()
        df = df[df['frame_idx']!=-1]
        self.indices    = df['frame_idx'].to_list()
        self.timestamps = df['timestamp'].to_list()

    def get_timestamp(self, idx) -> float:
        idx = int(idx)
        if idx in self.timestamp_dict:
            return self.timestamp_dict[idx]
        else:
            warnings.warn('frame_idx %d is not in set\n' % ( idx ), RuntimeWarning )
            return -1.

    def find_frame(self, ts) -> int:
        idx = bisect.bisect(self.timestamps, ts)
        # return nearest
        if idx>=len(self.timestamps):
            return self.indices[-1]
        elif idx>0 and abs(self.timestamps[idx-1]-ts)<abs(self.timestamps[idx]-ts):
            return self.indices[idx-1]
        else:
            return self.indices[idx]

    def get_last(self) -> tuple[int,float]:
        return self.indices[-1], self.timestamps[-1]

    def get_IFI(self) -> float:
        if self._ifi is None:
            self._ifi = np.mean(np.diff(self.timestamps))
        return self._ifi