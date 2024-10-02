# from https://stackoverflow.com/a/31614591/3103767
# added cleanup fun that may be needed to cancel work occurring in main thread so that join gets called
from threading import Thread

class PropagatingThread(Thread):
    def __init__(self, cleanup_fun=None, *args, **kwargs):
        super(PropagatingThread, self).__init__(*args, **kwargs)
        self.cleanup_fun = cleanup_fun
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e
            if self.cleanup_fun:
                self.cleanup_fun()

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret