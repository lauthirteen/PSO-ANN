
#!/bin/env python3

from functools import total_ordering
from hashlib import md5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import linecache
import os
import re
import sys
import time
import random
import glob
from threading import Thread
from multiprocessing import  Process
from scipy.integrate import simps

a = np.array([[1,2,3],[2,4,8]])
b = np.array([2,1,1])
c = a/b
print(a)
print(b)
print(c)
