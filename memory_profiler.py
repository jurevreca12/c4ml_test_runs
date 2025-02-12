import psutil
import subprocess
import time
import sys
import json

class ProcessContainer:
  def __init__(self, command=None, subp=None):
    self.command = command
    self.p = subp
    self.max_vms_memory = 0
    self.max_rss_memory = 0
    if self.p is None:
        self.execution_state = False
    else:
        self.execution_state = True


  def execute(self):
    self.max_vms_memory = 0
    self.max_rss_memory = 0
    if self.p is None:
        self.p = subprocess.Popen(self.command,shell=False)
        self.execution_state = True

  def poll(self):
    if not self.check_execution_state():
      return False

    try:
      pp = psutil.Process(self.p.pid)

      #obtain a list of the subprocess and all its descendants
      descendants = list(pp.children(recursive=True))
      descendants = descendants + [pp]

      rss_memory = 0
      vms_memory = 0

      #calculate and sum up the memory of the subprocess and all its descendants 
      for descendant in descendants:
        try:
          mem_info = descendant.memory_info()

          rss_memory += mem_info[0]
          vms_memory += mem_info[1]
        except psutil.NoSuchProcess:
          #sometimes a subprocess descendant will have terminated between the time
          # we obtain a list of descendants, and the time we actually poll this
          # descendant's memory usage.
          pass
      self.max_vms_memory = max(self.max_vms_memory, vms_memory)
      self.max_rss_memory = max(self.max_rss_memory, rss_memory)

    except psutil.NoSuchProcess:
      return self.check_execution_state()


    return self.check_execution_state()


  def is_running(self):
    return psutil.pid_exists(self.p.pid) and self.p.poll() == None
  def check_execution_state(self):
    if not self.execution_state:
      return False
    if self.is_running():
      return True
    self.executation_state = False
    return False

  def close(self,kill=False):

    try:
      pp = psutil.Process(self.p.pid)
      if kill:
        pp.kill()
      else:
        pp.terminate()
    except psutil.NoSuchProcess:
      pass


if __name__ == "__main__":
    proc_cont = ProcessContainer(sys.argv[1:])
    try:
      proc_cont.execute()
      #poll as often as possible; otherwise the subprocess might 
      # "sneak" in some extra memory usage while you aren't looking
      while proc_cont.poll():
        time.sleep(.5)
    finally:
      #make sure that we don't leave the process dangling?
      proc_cont.close()
   
    print('return code:', proc_cont.p.returncode)
    print(f'max_vms_memory_mib: {proc_cont.max_vms_memory / (1024 * 1024)} MiB')
    print(f'max_rss_memory_mib: {proc_cont.max_rss_memory / (1024 * 1024)} MiB')
    print(f'max_vms_memory: {proc_cont.max_vms_memory}')
    print(f'max_rss_memory: {proc_cont.max_rss_memory}')
    result_dict = {
        'sys_args': sys.argv,
        'max_vms_memory': proc_cont.max_vms_memory, 
        'max_rss_memory': proc_cont.max_rss_memory
    }
    with open('memory_info.json', 'w') as mem_file:
        json.dump(result_dict, mem_file)
    sys.exit(0)
