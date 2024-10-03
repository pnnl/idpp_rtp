"""
    unit testing subpackage, mirror structure of parent package
"""


import os


# set the path to the _include dir in this test subpackage
TEST_INCLUDE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "_include/"))
