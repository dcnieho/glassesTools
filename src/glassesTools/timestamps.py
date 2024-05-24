import numpy as np
import datetime
import csv
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


# for reading frame timestamp files
def from_file(file) -> np.ndarray:
    return np.genfromtxt(file, dtype=None, delimiter='\t', skip_header=1, usecols=1)

class Idx2Timestamp:
    def __init__(self, fileName):
        self.timestamps = {}
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(float(entry['frame_idx']))
                if frame_idx!=-1:
                    self.timestamps[frame_idx] = float(entry['timestamp'])

    def get(self, idx):
        if idx in self.timestamps:
            return self.timestamps[int(idx)]
        else:
            warnings.warn('frame_idx %d is not in set\n' % ( idx ), RuntimeWarning )
            return -1.

class Timestamp2Index:
    def __init__(self, fileName):
        self.indices = []
        self.timestamps = []
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(float(entry['frame_idx']))
                if frame_idx!=-1:
                    self.indices   .append(int(float(entry['frame_idx'])))
                    self.timestamps.append(    float(entry['timestamp']))

    def find(self, ts):
        idx = bisect.bisect(self.timestamps, ts)
        # return nearest
        if idx>=len(self.timestamps):
            return self.indices[-1]
        elif idx>0 and abs(self.timestamps[idx-1]-ts)<abs(self.timestamps[idx]-ts):
            return self.indices[idx-1]
        else:
            return self.indices[idx]

    def getLast(self):
        return self.indices[-1], self.timestamps[-1]

    def getIFI(self):
        return np.mean(np.diff(self.timestamps))