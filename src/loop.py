import os
import json
import tqdm

from src.util import *
from src.train import *

class TestLoop(object):
    def __init__(self, args):
      self.args = args
      self.vars = vars(args)

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
                self.vars["mode"] = "train"

                train_data_dir = os.path.join(setups_dir, setup)
                if not os.path.exists(train_data_dir):
                    os.makedirs(train_data_dir)
                self.vars["data_dir"] = train_data_dir

                with open(os.path.join(train_data_dir, "train", "labels", "mpii_style.json"), "r", encoding="utf-8") as fread:
                    labels_dict = json.load(fread)
                    num_mark = len(labels_dict[0]["joints_vis"])
                self.vars["num_mark"] = num_mark

                base_epoch = self.args.base_epoch
                epoch_d = self.args.epoch_d
                epoch_steps = self.args.epoch_steps
                report_dir_list = []
                for idx_e, num_epoch in enumerate(range(base_epoch, base_epoch + epoch_d * epoch_steps + 1, epoch_d)):
                    self.args.num_epoch = num_epoch
                    if idx_e > 0:
                        self.args.train_continue = "on"
                    
                    train_report_dir = os.path.join(reports_dir, setup+"epochs_"+str(num_epoch))
                    if not os.path.exists(train_report_dir):
                        os.makedirs(train_report_dir)
                    if not train_report_dir in report_dir_list:
                        report_dir_list += train_report_dir
                    self.args.ckpt_dir = os.path.join(train_report_dir, "checkpoint")
                    self.args.log_dir = os.path.join(train_report_dir, "log")
                    self.args.result_dir = os.path.join(train_report_dir, "result")
                    
                    # Initiate train loop
                    train(args=self.args)
                
                self.vars["mode"] = "test"
                for idx_t, test_report_dir in enumerate(report_dir_list):
                    self.args.ckpt_dir = os.path.join(test_report_dir, "checkpoint")
                    self.args.log_dir = os.path.join(test_report_dir, "log")
                    self.args.result_dir = os.path.join(test_report_dir, "result")

                    # Initiate test loop
                    test(args=self.args)
