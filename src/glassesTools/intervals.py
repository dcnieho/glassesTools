def is_in_interval(frame_idx, intervals):
    if intervals is None:
        return True # no interval defined, that means all frames

    # return True if we're in a current interval
    for f in range(0,len(intervals),2):
        if frame_idx>=intervals[f] and frame_idx<=intervals[f+1]:
            return True
    return False

def beyond_last_interval(frame_idx, intervals):
    if intervals is not None and frame_idx > intervals[-1]:
        return True
    return False