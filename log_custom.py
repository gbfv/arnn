import os

log_active_flags = []
all_flags = set()
all_logs = True
no_log = False
main_dir = "default"


def add_log(flag:str,filename:str,log:str):
    global log_active_flags, all_logs,all_flags,no_log,main_dir
    all_flags = all_flags.union([flag])
    if no_log or (not all_logs and flag not in log_active_flags):
        return
    os.makedirs(f"logs/{main_dir}",exist_ok=True)
    
    file = open(f"logs/{main_dir}/{filename}","a")
    file.write(log + "\n")
    file.close()


def get_all_flags_used():
    global all_flags
    print(all_flags)
def activate_flag(flag:str):
    global log_active_flags
    log_active_flags.append(flag)