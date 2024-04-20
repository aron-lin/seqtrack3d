""" 
__init__.py
Created by zenn at 2021/7/15 21:40
"""

from models import seqtrack3d

def get_model(name):
    model = globals()[name.lower()].__getattribute__(name.upper())
    return model
