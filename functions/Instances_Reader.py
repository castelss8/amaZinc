from z3 import *

import numpy as np
from itertools import combinations
from utils import *
import math
import time

def inst_read(instance_n):
    '''
    input: instance_n (int) - Instance Number
    output: n (int) - number of items
            m (int) - number of couriers
            s (list) - list of items' sizes
            l (list) - list of couriers' capacities
            D (list of lists) - Distance matrix
    '''
    instance_n+=1

    if instance_n<10:
        file_name='inst0'+str(instance_n)+'.dat'
    else:
        file_name='inst'+str(instance_n)+'.dat'
    file = open('./Instances/'+file_name, 'r')

    splitted_file = file.read().split('\n')

    m = int(splitted_file[0])
    n = int(splitted_file[1])
    l = list(map(int, splitted_file[2].split(' ')))
    tmp_s = splitted_file[3].split(' ')
    if '' in tmp_s:
        s = list(map(int, [tmp_s[i] for i in range(len(tmp_s)) if tmp_s[i]!='']))
    else:
        s = list(map(int, splitted_file[3].split(' ')))
    D = [list(map(int, line.strip().split(' '))) for line in splitted_file[4:(n+5)]]

    return n, m, s, l, D