from Point.Point import Point

# declare constants

# field parameters
MIN_SIZE = 2
DEFAULT_SEED = 0
DEFAULT_START = Point(0.0, 0.0)
DEFAULT_END = Point(10.0, 10.0)
FILE_PATH = "../database/field_data.txt"
FIELD_SIZE = 100  # 30

# icebergs parameters
if FIELD_SIZE >= 100:
    MIN_ICEBERGS = 3  # 2
    MAX_ICEBERGS = 10  # 5  # maximum number of icebergs should be 70
    MIN_DOTS = 4  # min dots in each icebergs
    MAX_DOTS = 7  # max dots in each icebergs
    MIN_RADIUS = 45  # MIN radius size
    MAX_RADIUS = 45  # MAX radius size - should be 45
else:  # FIELD_SIZE=30
    MIN_ICEBERGS = 2  # 2
    MAX_ICEBERGS = 4  # 5  # maximum number of icebergs should be 70
    MIN_DOTS = 4  # min dots in each icebergs
    MAX_DOTS = 5  # max dots in each icebergs
    MIN_RADIUS = 45  # MIN radius size
    MAX_RADIUS = 45  # MAX radius size - should be 45

# RRT parameters
ITERATION = 500
STEP_SIZE = 4

# Dubins parameters
DUBINS_VEL = 13  # constant velocity
DUBINS_PHI = 45  # maximum allowable roll angle
POLY_TOLERANCE = 10
