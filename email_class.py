# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy import stats
import sqlite3
import time

def get_unique_words(doc):
    # define splitter with non-alphabetic chars
    regex_str = r'[\W]+'
    #dddd
    splitter = re.compile(regex_str)
    # split documents with splitter as separator
    words = [s.lower() for s in re.split(splitter, doc) if len(s)>2 and len(s)<20]
    return {word:1 for word in words}