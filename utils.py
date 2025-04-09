import numpy as np
import re

class MemorySegment:
    def __init__(self):
        self.start = 0
        self.data = []
    
    
class Memory:
    def __init__(self): self.segments = []

    def seg_in_memory(self, address):
        for seg in self.segments:
            if address >= seg.start and (address - seg.start) < len(seg.data):
                return True
        return False

    def get_byte(self, address):
        for seg in self.segments:
            if address >= seg.start and (address - seg.start) < len(seg.data):
                return seg.data[ address - seg.start]
        return None

def parse_mem(lines):
    mem = Memory()
    for (s,d,e) in parse_by_keyword(lines, "segment"):
        segment = MemorySegment()
        m = re.search("addr=(.*)>", s)
        segment.start = int(m.group(1),base=16)
        segment.data = d
        m = re.search("addr=(.*)>", e)
        mem.segments.append(segment)
    return mem
    
def get_mem(filename):
    with open(filename) as f:
        lines = f.read().split("\n")
        return parse_mem(lines)

def parse_by_keyword(lines, keyword):
    array = []
    for i in range(len(lines)):
        if keyword in lines[i] and "/" not in lines[i]:
            line = lines[i+1].strip(" \n")
            data = tuple(int(a,base=16) for a in line.split(" ") if a)
            array.append((lines[i], data, lines[i+2]))
    return array

def parse_funs(lines):
    return set(parse_by_keyword(lines, "function")[0][1])

def parse_heads(lines):
    return set(parse_by_keyword(lines, "head")[0][1])

def parse_log_file(filename):
    res = dict()
    with open(filename) as f:
        lines = f.readlines()
        res['head'] = parse_heads(lines)
        res['function'] = parse_funs(lines)
        res['mem'] = parse_mem(lines)
    return res
