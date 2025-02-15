import psutil
import subprocess
import time
import sys
import json
import threading

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class ProcessContainer:
    def __init__(self, pid):
        self.pid = pid
        self.max_vms_memory = 0
        self.max_rss_memory = 0
        self.active = True

    def stop(self):
        self.active = False

    def poll(self):
        pp = psutil.Process(self.pid)

        # obtain a list of the subprocess and all its descendants
        descendants = list(pp.children(recursive=True))
        descendants = descendants + [pp]

        rss_memory = 0
        vms_memory = 0
        # calculate and sum up the memory of the subprocess
        # and all its descendants
        for descendant in descendants:
            mem_info = descendant.memory_info()
            rss_memory += mem_info[0]
            vms_memory += mem_info[1]
        self.max_vms_memory = max(self.max_vms_memory, vms_memory)
        self.max_rss_memory = max(self.max_rss_memory, rss_memory)

    @threaded
    def profile(self):
        while self.active:
            self.poll()
            time.sleep(1.0) 
