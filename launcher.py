import sys
import random
import torch
import queue
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time

gpu_queue = queue.Queue()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()): #ajoute les gpu à la queue
        gpu_queue.put(i)
else:
    for i in range(6): #remplace par des cpu
        gpu_queue.put(i)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python launcher.py <main|sub>")
    

