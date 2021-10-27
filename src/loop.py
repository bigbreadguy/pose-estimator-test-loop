import os
import json
import tqdm

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
            for idx, design in enumerate(tqdm.tqdm(test_design_list)):
                report_dir = os.path.join(test_report_dir, design)
                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
    
    def stroll(self):
        for idx_d, design in enumerate(tqdm.tqdm(self.test_design_list)):
            setups_dir = os.path.join(self.datasets_dir, design)
            reports_dir = os.path.join(self.test_report_dir, design)

            setups_list = os.listdir(setups_dir)
            for idx_s, setup in enumerate(setups_list):
                # Set arguments for training
                self.opts.set_mode(mode="train")
                self.opts.set_continue(train_continue="off")

                train_data_dir = os.path.join(setups_dir, setup, "train")
                if not os.path.exists(train_data_dir):
                    os.makedirs(train_data_dir)
                self.opts.set_data_dir(dir=train_data_dir)

                with open(os.path.join(train_data_dir, "labels", "mpii_style.json"), "r", encoding="utf-8") as fread:
                    labels_dict = json.load(fread)
                    num_mark = len(labels_dict[0]["joints_vis"])
                self.opts.set_num_mark(num=num_mark)

                base_epoch = self.opts.parse(args="--base_epoch").base_epoch
                epoch_d = self.opts.parse(args="--epoch_d").epoch_d
                epoch_steps = self.opts.parse(args="--epoch_steps").epoch_steps
                report_dir_list = []
                for idx_e, num_epoch in enumerate(range(base_epoch, base_epoch + epoch_d * epoch_steps + 1, epoch_d)):
                    self.opts.set_epoch(epoch=num_epoch)
                    if idx_e > 0:
                        self.opts.set_continue(train_continue="on")
                    
                    train_report_dir = os.path.join(reports_dir, setup+"epochs_"+str(num_epoch))
                    if not os.path.exists(train_report_dir):
                        os.makedirs(train_report_dir)
                    if not train_report_dir in report_dir_list:
                        report_dir_list += train_report_dir
                    self.opts.set_report_dir(base_dir=train_report_dir)

                    args = self.opts.setup()
                    
                    # Initiate train loop
                    train(args)
                
                self.opts.set_mode(mode="test")
                test_data_dir = os.path.join(setups_dir, setup, "test")
                if not os.path.exists(test_data_dir):
                    os.makedirs(test_data_dir)
                self.opts.set_data_dir(dir=test_data_dir)
                for idx_t, test_report_dir in enumerate(report_dir_list):
                    self.opts.set_report_dir(base_dir=test_report_dir)

                    args = self.opts.setup()

                    # Initiate test loop
                    test(args)
