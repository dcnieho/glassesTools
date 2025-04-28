import enum
import multiprocessing.managers
import pebble
import multiprocessing
import typing
import threading
import dataclasses
import time

from . import json, utils

ProcessFuture: typing.TypeAlias = pebble.ProcessFuture
_UserDataT = typing.TypeVar("_UserDataT")


class CounterContext:
    _count = -1     # so that first number is 0
    def __enter__(self):
        self._count += 1
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def get_count(self):
        return self._count


class State(enum.IntEnum):
    # states for signalling task status, to be stored in files
    Not_Run     = enum.auto()
    Pending     = enum.auto()
    Running     = enum.auto()
    Completed   = enum.auto()
    # two more states needed by process_pool/GUI
    Canceled    = enum.auto()
    Failed      = enum.auto()
    @property
    def displayable_name(self):
        return self.name.replace("_", " ")
json.register_type(json.TypeEntry(State,'__enum.process.State__', utils.enum_val_2_str, lambda x: getattr(State, x.split('.')[1])))


class ProcessWaiter(object):
    """Routes completion through to user callback."""
    def __init__(self, job_id: int, user_data: _UserDataT, done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None]):
        self.done_callback  = done_callback
        self.job_id         = job_id
        self.user_data      = user_data

    def add_result(self, future: ProcessFuture):
        self._notify(future, State.Completed)

    def add_exception(self, future: ProcessFuture):
        self._notify(future, State.Failed)

    def add_cancelled(self, future: ProcessFuture):
        self._notify(future, State.Canceled)

    def _notify(self, future: ProcessFuture, state: State):
        self.done_callback(future, self.job_id, self.user_data, state)

class PoolJob(typing.NamedTuple):
    future   : ProcessFuture
    user_data: _UserDataT
class ProcessPool:
    def __init__(self, num_workers = 2):
        self.num_workers            = num_workers
        self.auto_cleanup_if_no_work= False

        # NB: pool is only started in run() once needed
        self._pool              : pebble.pool.ProcessPool   = None
        self._jobs              : dict[int,PoolJob]         = None
        self._job_id_provider   : CounterContext            = CounterContext()
        self._lock              : threading.Lock            = threading.Lock()

    def _cleanup(self):
        # cancel all pending and running jobs
        self.cancel_all_jobs()

        # stop pool
        if self._pool and self._pool.active:
            self._pool.stop()
            self._pool.join()
        self._pool = None
        self._jobs = None

    def cleanup(self):
        with self._lock:
            self._cleanup()

    def cleanup_if_no_jobs(self):
        with self._lock:
            self._cleanup_if_no_jobs()

    def _cleanup_if_no_jobs(self):
        # NB: lock must be acquired when calling this
        if self._pool and not self._jobs:
            self._cleanup()

    def set_num_workers(self, num_workers: int):
        # NB: doesn't change number of workers on an active pool, only takes effect when pool is restarted
        self.num_workers = num_workers

    def run(self, fn: typing.Callable, user_data: _UserDataT=None, done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None]=None, *args, **kwargs) -> tuple[int, ProcessFuture]:
        with self._lock:
            if self._pool is None or not self._pool.active:
                context = multiprocessing.get_context("spawn")  # ensure consistent behavior on Windows (where this is default) and Unix (where fork is default, but that may bring complications)
                self._pool = pebble.ProcessPool(max_workers=self.num_workers, context=context)

            if self._jobs is None:
                self._jobs = {}

            with self._job_id_provider:
                job_id = self._job_id_provider.get_count()
                self._jobs[job_id] = PoolJob(self._pool.schedule(fn, args=args, kwargs=kwargs), user_data)
                if done_callback:
                    self._jobs[job_id].future._waiters.append(ProcessWaiter(job_id, user_data, done_callback))
                # Finally, register our internal cleanup to run last
                self._jobs[job_id].future._waiters.append(ProcessWaiter(job_id, user_data, self._job_done_callback))
                return job_id, self._jobs[job_id].future

    def _job_done_callback(self, future: ProcessFuture, job_id: int, user_data: _UserDataT, state: State):
        with self._lock:
            if self._jobs is not None and job_id in self._jobs:
                # clean up the work item since we're done with it
                del self._jobs[job_id]

            if self.auto_cleanup_if_no_work:
                # close pool if no work left
                self._cleanup_if_no_jobs()

    def get_job_state(self, job_id: int) -> State:
        if not self._jobs:
            return None
        job = self._jobs.get(job_id, None)
        if job is None:
            return None
        else:
            return _get_status_from_future(job.future)

    def get_job_user_data(self, job_id: int) -> _UserDataT:
        if not self._jobs:
            return None
        job = self._jobs.get(job_id, None)
        if job is None:
            return None
        else:
            return job.user_data

    def cancel_job(self, job_id: int) -> bool:
        if not self._jobs:
            return False
        if (job := self._jobs.get(job_id, None)) is None:
            return False
        return job.future.cancel()

    def cancel_all_jobs(self):
        if not self._jobs:
            return
        for job_id in reversed(self._jobs): # reversed so that later pending jobs don't start executing when earlier gets cancelled, only to be canceled directly after
            if not self._jobs[job_id].future.done():
                self._jobs[job_id].future.cancel()


class JobPayload(typing.NamedTuple):
    fn:     typing.Callable[..., None]
    args:   tuple
    kwargs: dict

class _EMA(object):
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.
    N.B.: copied from tqdm

    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Ranges from 0 (yields old value) to 1 (yields new value).
    """
    def __init__(self, smoothing=0.3):
        self.alpha = smoothing
        self.last = 0
        self.calls = 0

    def __call__(self, x=None):
        beta = 1 - self.alpha
        if x is not None:
            self.last = self.alpha * x + beta * self.last
            self.calls += 1
        return self.last / (1 - beta ** self.calls) if self.calls else self.last

def _format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'

class JobProgress:
    # based on tqdm, much reduced functionality
    def __init__(self, initial: int=0, total: int=999999, unit: str="it", update_interval: int=1, smoothing: float=.3, printer: typing.Callable[[str], None]=None, print_interval: int = 100):
        self.n              = initial
        self.total          = total
        self.unit           = unit
        self.update_interval= update_interval
        self.smoothing      = 0.3

        self._printer       = printer
        self.print_interval = print_interval

        self._ema_dn        = _EMA(smoothing)
        self._ema_dt        = _EMA(smoothing)
        self._time          = time.time

        self.percentage     = 0.
        self.progress_str   = ''

        self.last_update_n  = initial
        self.last_update_t  = self._time()
        self.start_t        = self.last_update_t

    def set_total(self, total: int):
        self.total = total
    def set_unit(self, unit: str):
        self.unit = unit
    def set_intervals(self, update_interval: int, print_interval: int):
        self.update_interval= max(1,update_interval)
        self.print_interval = max(1,print_interval)

    def update(self, n=1):
        self.n += n
        should_print = self._printer is not None and self.n%self.print_interval==0
        if not self.progress_str or self.n-self.last_update_n>=self.update_interval or should_print or self.n==self.total:
            cur_t = self._time()
            dt = cur_t - self.last_update_t
            dn = self.n - self.last_update_n
            if self.smoothing and dt and dn:
                self._ema_dn(dn)
                self._ema_dt(dt)
                rate = self._ema_dn()/dts if (dts:=self._ema_dt()) else None
            else:
                rate = dn/dt if dt else None

            inv_rate = 1 / rate if rate else None
            rate_fmt     = (f'{rate:5.2f}'     if     rate else '?') + ' ' + self.unit + '/s'
            rate_inv_fmt = (f'{inv_rate:5.2f}' if inv_rate else '?') + ' s/' + self.unit
            rate_str = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_fmt

            elapsed = cur_t - self.start_t
            elapsed_str = _format_interval(elapsed)

            remaining = (self.total - self.n) / rate if rate and self.total else 0
            remaining_str = _format_interval(remaining) if rate else '?'

            self.percentage = (self.n / self.total) * 100
            percentage_str = f'{self.percentage:3.0f}%'

            self.progress_str = f'{self.n}/{self.total} ({percentage_str}) [{elapsed_str}<{remaining_str}, {rate_str}]'

            self.last_update_n = self.n
            self.last_update_t = cur_t
        if should_print:
            self._printer(self.progress_str)

    def get_progress(self):
        return (self.percentage, self.progress_str)

    def set_start_time_to_now(self):
        self.last_update_t  = self._time()
        self.start_t        = self.last_update_t

    def set_finished(self):
        self.update(self.total-self.n)
multiprocessing.managers.BaseManager.register('JobProgress', JobProgress)

@dataclasses.dataclass
class JobDescription(typing.Generic[_UserDataT]):
    user_data:          _UserDataT
    payload:            JobPayload
    progress:           JobProgress
    done_callback:      typing.Callable[[ProcessFuture, _UserDataT, int, State], None]

    exclusive_id:       typing.Optional[int]      = None # if set, only one task with a given id can be run at a time, rest are kept in waiting. E.g. to ensure only one task needing a gui is run at a time
    priority:           int                       = 999  # jobs with a higher priority are scheduled first, unless they cannot be because they're exclusive (exclusive_id is set) and task of that exclusivity is already running, or because their dependencies are not met yet
    depends_on:         typing.Optional[set[int]] = None # set of job ids that need to be completed before this one can be launched

    _pool_job_id:       typing.Optional[int]            = None
    _future:            typing.Optional[ProcessFuture]  = dataclasses.field(init=False, default=None)
    _final_state:       typing.Optional[State]          = dataclasses.field(init=False, default=None)
    error:              typing.Optional[str]            = None

    def get_state(self) -> State:
        if self._final_state is not None:
            return self._final_state
        elif self._future is not None:
            job_state = _get_status_from_future(self._future)
            if job_state not in [State.Pending, State.Running]:
                # finished, cache result
                self._final_state = job_state
                # can also dump the future
                self._future = None
            return job_state
        return State.Pending

    def is_scheduled(self):
        # True if job is scheduled to the pool
        return self._pool_job_id is not None and self.get_state() in [State.Pending, State.Running]

    def is_finished(self):
        return self._final_state is not None

class JobScheduler(typing.Generic[_UserDataT]):
    def __init__(self, pool: ProcessPool, job_is_valid_checker : typing.Callable[[_UserDataT], bool]|None = None):
        self.jobs               : dict[int, JobDescription[_UserDataT]] = {}
        self._job_id_provider   : CounterContext                        = CounterContext()
        self._pending_jobs      : list[int]                             = []    # jobs not scheduled or finished

        self._job_is_valid_checker  = job_is_valid_checker
        self._pool                  = pool

        self._manager               = multiprocessing.managers.BaseManager(ctx=multiprocessing.get_context("spawn"))
        self._manager.start()

    def add_job(self,
                user_data: _UserDataT, payload: JobPayload, done_callback: typing.Callable[[ProcessFuture, _UserDataT, int, State], None],
                progress_indicator: JobProgress=None,
                exclusive_id: typing.Optional[int] = None, priority: int = None, depends_on: typing.Optional[set[int]] = None) -> int:
        with self._job_id_provider:
            job_id = self._job_id_provider.get_count()
        self.jobs[job_id] = JobDescription(user_data, payload, progress_indicator, done_callback, exclusive_id, priority, depends_on)
        self._pending_jobs.append(job_id)
        return job_id

    def get_progress_indicator(self, **kwargs):
        return self._manager.JobProgress(**kwargs)

    def cancel_job(self, job_id: int):
        if job_id not in self.jobs:
            return

        if self.jobs[job_id]._pool_job_id is not None:
            self._pool.cancel_job(self.jobs[job_id]._pool_job_id)
        else:
            if job_id in self._pending_jobs:
                self._pending_jobs.remove(job_id)
            self.jobs[job_id]._final_state = State.Canceled
            if self.jobs[job_id].done_callback:
                self.jobs[job_id].done_callback(None, None, self.jobs[job_id].user_data, State.Canceled)
        # TODO: also cancel all jobs that depend on this job

    def cancel_all_jobs(self):
        # cancel any jobs that may still be running
        for job_id in self.jobs:
            if self.jobs[job_id]._final_state not in [State.Completed, State.Canceled, State.Failed]:
                self.cancel_job(job_id)
        # make double sure they're cancelled
        self._pool.cancel_all_jobs()
        # ensure pool is no longer running
        self._pool.cleanup()

    def clear(self):
        # cancel any jobs that may still be running
        self.cancel_all_jobs()
        # clean up
        self._pending_jobs.clear()
        self.jobs.clear()
        # reset job counter
        self._job_id_provider = CounterContext()

    def update(self):
        # first count how many are scheduled to the pool and whether all tasks are still valid
        num_scheduled_to_pool = 0
        exclusive_ids: set[int] = set()
        for job_id in self.jobs:
            job = self.jobs[job_id]
            # check job still valid or should be canceled
            if self._job_is_valid_checker is not None:
                if not self._job_is_valid_checker(job.user_data):
                    self.cancel_job(job_id)
            # remove finished job from pending jobs if its still there
            job.get_state() # update state
            if job.is_finished() and job_id in self._pending_jobs:
                self._pending_jobs.remove(job_id)
            # check how many scheduled jobs we have
            if job.is_scheduled():
                num_scheduled_to_pool += 1
                if job.exclusive_id is not None:
                    exclusive_ids.add(job.exclusive_id)

        # if we have less than max number of tasks scheduled to the pool, see if anything new to schedule to the pool
        while num_scheduled_to_pool < self._pool.num_workers:
            # find suitable next task to schedule
            # order tasks by priority, filtering out those who have a colliding exclusive_id
            job_ids = [i for i in sorted(self._pending_jobs, key=lambda ii: 999 if self.jobs[ii].priority is None else self.jobs[ii].priority) if self.jobs[i].exclusive_id not in exclusive_ids]
            if not job_ids:
                break
            job_id = job_ids[0]
            to_schedule = self.jobs[job_id]
            extra_kwargs = {}
            if to_schedule.progress is not None:
                extra_kwargs['progress_indicator'] = to_schedule.progress

            to_schedule._pool_job_id, to_schedule._future = \
                self._pool.run(to_schedule.payload.fn, to_schedule.user_data, to_schedule.done_callback, *to_schedule.payload.args, **extra_kwargs, **to_schedule.payload.kwargs)
            self._pending_jobs.remove(job_id)
            if to_schedule.exclusive_id is not None:
                exclusive_ids.add(to_schedule.exclusive_id)
            num_scheduled_to_pool += 1


def _get_status_from_future(fut: ProcessFuture) -> State:
    if fut.running():
        return State.Running
    elif fut.done():
        if fut.cancelled():
            return State.Canceled
        elif fut.exception() is not None:
            return State.Failed
        else:
            return State.Completed
    else:
        return State.Pending