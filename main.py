from src.util import *
from src.loop import TestLoop

if __name__ == "__main__":
    opts = opts()
    
    loop = TestLoop(opts)
    
    print("Building Test Loop")
    loop.build()
    
    print("Strolling Test Loop")
    loop.stroll()

    print("Done")