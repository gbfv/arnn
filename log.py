import os

log_active_flags = []
all_logs = True
def add_log(flag:str,filename:str,log:str):
    global log_active_flags, all_logs
    if not all_logs and flag not in log_active_flags:
        return
    os.makedirs("logs",exist_ok=True)
    
    file = open(f"logs/{filename}","a")
    file.write(log + "\n")
    file.close()

def activate_flag(flag:str):
    global log_active_flags
    log_active_flags.append(flag)