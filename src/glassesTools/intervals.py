# intervals are either lists of lists ([1,2], [100,200]), or dicts containing such lists as values

def is_in_interval(frame_idx, intervals):
    if intervals is None:
        return True # no interval defined, that means all frames should be processed

    # if its a dict, flatten it
    if isinstance(intervals, dict):
        intervals = [iv for k in intervals for iv in intervals[k]]

    # return True if we're in a current interval
    for iv in intervals:
        if frame_idx>=iv[0] and frame_idx<=iv[1]:
            return True
    return False

def beyond_last_interval(frame_idx, intervals):
    if not intervals:
        return False
    elif isinstance(intervals, dict):
        for k in intervals:
            if intervals[k] and frame_idx <= intervals[k][-1][-1]:
                return False
        return True
    else:
        return frame_idx > intervals[-1][-1]