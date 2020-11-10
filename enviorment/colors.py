class Color:
    
    WHITE  = (255, 255, 255)
    BLACK  = (0,   0,   0  )
    GRAY   = (100, 100, 100)
    RED    = (220, 20,  60 )
    GREEN  = (50,  205, 50 )
    YELLOW = (255, 255, 0  )
    PURPLE = (218, 112, 214)

    ALL = [WHITE, BLACK, GRAY, RED, GREEN]
    
# just for printing colors in terminal
class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKCYAN    = '\033[96m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    
def green(msg):
    return bcolors.OKGREEN+msg+bcolors.ENDC

def header(msg):
    return bcolors.HEADER+msg+bcolors.ENDC

def fail(msg):
    return bcolors.FAIL+msg+bcolors.ENDC

def cyan(msg):
    return bcolors.OKCYAN+msg+bcolors.ENDC

def warning(msg):
    return bcolors.WARNING+msg+bcolors.ENDC