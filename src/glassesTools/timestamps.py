import numpy as np
import pandas as pd
import datetime
import bisect
import pathlib
import enum

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
class Type(enum.Enum):
    Normal      = enum.auto()
    Stretched   = enum.auto()


class VideoTimestamps:
    def __init__(self, fileName: str|pathlib.Path):
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

        self.timestamp_stretched_dict   : dict[int,float] = None
        self.timestamps_stretched       : list[float] = None
        self._ifi_stretched             : float = None
        self.has_stretched = 'timestamp_stretched' in df.columns
        if self.has_stretched:
            self.timestamp_stretched_dict = df.to_dict()['timestamp_stretched']
            self.timestamps_stretched = df['timestamp_stretched'].to_list()


    def get_timestamp(self, idx: int, which: Type=Type.Normal) -> float:
        idx = int(idx)
        match which:
            case Type.Normal:
                d = self.timestamp_dict
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError('stretched timestamps are not available for this video')
                d = self.timestamp_stretched_dict
        if idx in d:
            return d[idx]
        else:
            return -1.

    def find_frame(self, ts: float, which: Type=Type.Normal) -> int:
        match which:
            case Type.Normal:
                timestamps = self.timestamps
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError('stretched timestamps are not available for this video')
                timestamps = self.timestamps_stretched

        idx = bisect.bisect(timestamps, ts)
        # return nearest
        if idx>=len(timestamps):
            return self.indices[-1]
        elif idx>0 and abs(timestamps[idx-1]-ts)<abs(timestamps[idx]-ts):
            return self.indices[idx-1]
        else:
            return self.indices[idx]

    def get_last(self, which: Type=Type.Normal) -> tuple[int,float]:
        match which:
            case Type.Normal:
                timestamps = self.timestamps
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError('stretched timestamps are not available for this video')
                timestamps = self.timestamps_stretched
        return self.indices[-1], timestamps[-1]

    def get_IFI(self, which: Type=Type.Normal) -> float:
        match which:
            case Type.Normal:
                if self._ifi is None:
                    self._ifi = np.mean(np.diff(self.timestamps))
                return self._ifi
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError('stretched timestamps are not available for this video')
                if self._ifi_stretched is None:
                    self._ifi_stretched = np.mean(np.diff(self.timestamps_stretched))
                return self._ifi_stretched