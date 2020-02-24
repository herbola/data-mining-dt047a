
# Enables import from other directories
import os, sys 
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# other imports
from shared.config import *