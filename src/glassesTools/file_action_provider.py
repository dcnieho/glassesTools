import concurrent
import pathlib
from typing import Callable

from . import async_thread, file_actions, platform

class FileActionProvider:
    def __init__(self, listing_callback: Callable[[str, str|pathlib.Path, list[file_actions.DirEntry]|Exception], None]=None, action_callback: Callable[[str, pathlib.Path, str, pathlib.Path|Exception], None] = None):
        self.waiters: set[concurrent.futures.Future] = set()

        self.listing_callbacks: list[Callable[[list[file_actions.DirEntry]|Exception, bool], None]] = []
        if listing_callback:
            self.listing_callbacks.append(listing_callback)
        self.action_callbacks:  list[Callable[[pathlib.Path|Exception, str], None]] = []
        if action_callback:
            self.action_callbacks.append(action_callback)

    def __del__(self):
        for w in self.waiters:
            if not w.done():
                w.cancel()

    local_name = 'This PC' if platform.os==platform.Os.Windows else 'Root'

    def get_listing(self, path: str|pathlib.Path) -> list[file_actions.DirEntry]|concurrent.futures.Future:
        fut = None
        if platform.os==platform.Os.Windows:
            if path=='root':
                try:
                    result = file_actions.get_drives()
                    result.extend(file_actions.get_thispc_listing())
                except Exception as exc:
                    result = exc
                self._listing_done(result, 'root')
            else:
                # check whether this is a path to a network computer (e.g. \\SERVER)
                net_comp = file_actions.get_net_computer(path)
                try:
                    if net_comp:
                        # network computer name, get its shares
                        result = file_actions.get_visible_shares(net_comp,'Guest','')
                    else:
                        # normal directory or share on a network computer, no special handling needed
                        result = file_actions.get_dir_list_sync(path)
                except Exception as exc:
                    result = exc
                self._listing_done(result, path)
        else:
            try:
                l_path = path
                if path=='root':
                    l_path = '/'
                result = file_actions.get_dir_list_sync(l_path)
            except Exception as exc:
                result = exc
            self._listing_done(result, path)
        if fut:
            self.waiters.add(fut)
        return fut

    def _listing_done(self, fut: concurrent.futures.Future|list[file_actions.DirEntry], path: str|pathlib.Path):
        result = self._get_result_from_future(fut)
        if result=='cancelled':
            return  # nothing more to do

        if not self.listing_callbacks:
            return
        if result is None:
            return

        # call callback
        for c in self.listing_callbacks:
            c(path, result)

    def make_dir(self, path: pathlib.Path):
        action = 'make_dir'
        fut = async_thread.run(file_actions.make_dir(path), lambda f: self._action_done(f, path, action))
        self.waiters.add(fut)
        return fut

    def rename_path(self, old_path: pathlib.Path, new_path: pathlib.Path):
        action = 'rename_path'
        fut = async_thread.run(file_actions.rename_path(old_path, new_path), lambda f: self._action_done(f, old_path, action))
        self.waiters.add(fut)
        return fut

    def _action_done(self, fut: concurrent.futures.Future, path: pathlib.Path, action: str):
        result = self._get_result_from_future(fut)
        if result=='cancelled':
            return  # nothing more to do

        if not self.action_callbacks:
            return

        # call callback
        for c in self.action_callbacks:
            c(path, action, result)

    def _get_result_from_future(self, fut: concurrent.futures.Future|list[file_actions.DirEntry]) -> list[file_actions.DirEntry]:
        if isinstance(fut, concurrent.futures.Future):
            self.waiters.discard(fut)
            try:
                return fut.result()
            except concurrent.futures.CancelledError:
                return 'cancelled'
            except Exception as exc:
                return exc
        else:
            return fut