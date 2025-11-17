# intervals are either lists of lists ([1,2], [100,200]), or dicts containing such lists as values
from . import annotation

def is_in_interval(frame_idx: int, intervals: tuple[annotation.EventType, list[int]|list[list[int]]]) -> bool:
    if intervals is None:
        return True # no interval defined, that means all frames should be processed

    # return True if we're in a current interval
    for iv in intervals[1]:
        if len(iv)==1:
            if frame_idx==iv[0]:
                # exactly on the frame of a time point coding
                return True
        elif frame_idx>=iv[0] and frame_idx<=iv[1]:
            return True
    return False

def which_interval(frame_idx, intervals: dict[str, tuple[annotation.EventType, list[int]|list[list[int]]]]) -> tuple[list[str]|None, list[list[int]]|None]:
    if not isinstance(intervals, dict) or not intervals:
        return None, None
    # prep input
    if any(not isinstance(intervals[k][1][0],list) for k in intervals if intervals[k][1]):
        intervals = annotation.unflatten_annotation_dict(intervals, add_incomplete_intervals=True)

    # get output
    keys = []
    ivals = []
    for k in intervals:
        for iv in intervals[k][1]:
            if len(iv)==1:
                if frame_idx==iv[0]:
                    keys.append(k)
                    ivals.append(iv)
            elif frame_idx>=iv[0] and frame_idx<=iv[1]:
                keys.append(k)
                ivals.append(iv)

    return keys, ivals

def beyond_last_interval(frame_idx, intervals: dict[str, tuple[annotation.EventType, list[int]|list[list[int]]]]):
    if not intervals:
        return False
    if isinstance(intervals, dict):
        for k in intervals:
            if intervals[k] is None:
                # None indicates all frames should be processed
                return False
            if not intervals[k][1]:
                return False
            if isinstance(intervals[k][1][-1], list):
                if frame_idx <= intervals[k][1][-1][-1]:
                    return False
            elif frame_idx <= intervals[k][1][-1]:
                return False
        return True