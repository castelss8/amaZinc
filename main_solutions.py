from z3 import *

import numpy as np
from itertools import combinations
from utils import *
import math
import time
from importlib import reload

from SAT import SAT_Model as SM
from functions import Instances_Reader as IR
reload(SM)

for i in range(21):
    SM.SAT_MCP(IR.inst_read(i))