from src.util import *
from src.loop import TestLoop

if __name__ == "__main__":
    args = opts().parse()

    loop = TestLoop(args)
    
    print("Building Test Loop")
    loop.build()
    
    if args.spec == "all":
        if args.loop == "stroll":
            print("Strolling Test Loop")
            loop.stroll()
        elif args.loop == "test":
            print("Test Only")
            loop.test_stroll()
        elif args.loop == "evaluate":
            print("Evaluate with PCK metric")
            loop.evaluations()
            
    else:
        print(f"Training Designated Spec : {args.spec}")
        loop.train_spec()

    print("Done")