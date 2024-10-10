# intervals are either lists of lists ([1,2], [100,200]), or dicts containing such lists as values
from . import annotation

def _prep_ival_dict(intervals, add_incomplete_intervals=False):
    intervals = intervals.copy()    # so any manipulation doesn't propagate back out
    for k in intervals:
        if not intervals[k]:
            continue
        if not isinstance(intervals[k][0], list):
            if not isinstance(k,annotation.Event) or annotation.type_map[k]==annotation.Type.Interval:
                temp = []
                for m in range(0,len(intervals[k])-1,2): # read in batches of two, and run until -1 to make sure we don't pick up incomplete intervals
                    temp.append(intervals[k][m:m+2])
                if add_incomplete_intervals and len(intervals[k])%2==1:
                    temp.append(intervals[k][-1:])
                intervals[k] = temp
            else:
                intervals[k] = [[tp] for tp in intervals[k]]
    return intervals

def is_in_interval(frame_idx, intervals):
    if intervals is None:
        return True # no interval defined, that means all frames should be processed

    # if its a dict, flatten it
    if isinstance(intervals, dict):
        intervals = _prep_ival_dict(intervals)
        intervals = [iv for k in intervals for iv in intervals[k]]

    # return True if we're in a current interval
    for iv in intervals:
        if len(iv)==1:
            if frame_idx==iv[0]:
                # exactly on the frame of a time point coding
                return True
        elif frame_idx>=iv[0] and frame_idx<=iv[1]:
            return True
    return False

def which_interval(frame_idx, intervals):
    if not isinstance(intervals, dict):
        return None, None
    # prep input, if needed
    intervals = _prep_ival_dict(intervals, add_incomplete_intervals=True)

    # get output
    keys = []
    ivals = []
    for k in intervals:
        if not intervals[k]:
            continue
        for iv in intervals[k]:
            if len(iv)==1:
                if frame_idx==iv[0]:
                    keys.append(k)
                    ivals.append(iv)
            elif frame_idx>=iv[0] and frame_idx<=iv[1]:
                keys.append(k)
                ivals.append(iv)

    return keys, ivals

def beyond_last_interval(frame_idx, intervals):
    if not intervals:
        return False
    elif isinstance(intervals, dict):
        for k in intervals:
            if not intervals[k]:
                return False
            if isinstance(intervals[k][-1], list):
                if frame_idx <= intervals[k][-1][-1]:
                    return False
            elif frame_idx <= intervals[k][-1]:
                return False
        return True
    else:
        return frame_idx > intervals[-1][-1]