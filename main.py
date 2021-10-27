from src.util import *
from src.loop import TestLoop

if __name__ == "__main__":
    args = opts().parse()

    loop = TestLoop(args)
    
    print("Building Test Loop")
    loop.build()
    
    print("Strolling Test Loop")
    loop.stroll()

    print("Done")