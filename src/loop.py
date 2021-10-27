import os

from util import *
from train import *

class TestLoop(object):
    def __init__(self, opts):
      self.opts = opts

    def build(self):
        cwd = os.getcwd()
        datasets_dir = os.path.join(cwd, "datasets")
        self.datasets_dir = datasets_dir

        test_report_dir = os.path.join(cwd, "test_results")
        self.test_report_dir = test_report_dir
        if not os.path.exists(test_report_dir):
            os.makedirs(test_report_dir)

        test_design_list = os.listdir(datasets_dir)
        self.test_design_list = test_design_list
        if len(test_design_list) > 0:
            for idx, design in enumerate(test_design_list):
                report_dir = os.path.join(test_report_dir, design)
                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
    
    def stroll(self):
        for idx_d, design in enumerate(self.test_design_list):
            setups_dir = os.path.join(self.datasets_dir, design)
            reports_dir = 

            setups_list = os.listdir(setups_dir)
            for idx_s, setups_list in enumerate(setups_list):
                # Set arguments for training
                self.opts.set_mode(mode="train")
                self.opts.set_continue(train_continue="off")
                base_epoch = self.opts.parse(args="--base_epoch").base_epoch
                epoch_d = self.opts.parse(args="--base_epoch").epoch_d
                epoch_steps = self.opts.parse(args="--base_epoch").epoch_steps