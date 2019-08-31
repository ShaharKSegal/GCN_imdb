import os
import logging

import argparse

from ds import CastGraph

log: logging.Logger
args: argparse.Namespace
cast_graph: CastGraph

data_path = os.path.join('.', 'data')
batch_size = 100
train_log_interval = 2
