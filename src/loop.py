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
        
        self.execute_log_dir = os.path.join(cwd, "log")
        if not os.path.exists(self.execute_log_dir):
            os.makedirs(self.execute_log_dir)
        
        if len(test_design_list) > 0:
            for idx, design in enumerate(tqdm.tqdm(test_design_list)):
                report_dir = os.path.join(test_report_dir, design)
                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
                
                log_dir = os.path.join(self.execute_log_dir, design)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    
    def stroll(self):
        for idx_d, design in enumerate(tqdm.tqdm(self.test_design_list)):
            setups_dir = os.path.join(self.datasets_dir, design)
            reports_dir = os.path.join(self.test_report_dir, design)

            setups_list = os.listdir(setups_dir)
            for idx_s, setup in enumerate(setups_list):
                # Set arguments for training
                self.vars["mode"] = "train"

                self.vars["log_prefix"] = os.path.join(self.execute_log_dir, design, setup)

                train_data_dir = os.path.join(setups_dir, setup)
                if not os.path.exists(train_data_dir):
                    os.makedirs(train_data_dir)
                self.vars["data_dir"] = train_data_dir

                with open(os.path.join(train_data_dir, "train", "labels", "mpii_style.json"), "r", encoding="utf-8") as fread:
                    labels_dict = json.load(fread)
                    num_mark = len(labels_dict[0]["joints_vis"])
                self.vars["num_mark"] = num_mark

                num_epoch = self.args.num_epoch
                
                train_report_dir = os.path.join(reports_dir, setup)
                if not os.path.exists(train_report_dir):
                    os.makedirs(train_report_dir)
                
                self.args.ckpt_dir = os.path.join(train_report_dir, "checkpoint")
                self.args.log_dir = os.path.join(train_report_dir, "log")
                self.args.result_dir = os.path.join(train_report_dir, "result")
                
                # Initiate train loop
                train(args=self.args)
                
                self.vars["mode"] = "test"

                self.args.ckpt_dir = os.path.join(train_report_dir, "checkpoint")
                self.args.log_dir = os.path.join(train_report_dir, "log")
                self.args.result_dir = os.path.join(train_report_dir, "result")

                # Initiate test loop
                test(args=self.args)

    def test_stroll(self):
        for idx_d, design in enumerate(tqdm.tqdm(self.test_design_list)):
            setups_dir = os.path.join(self.datasets_dir, design)
            reports_dir = os.path.join(self.test_report_dir, design)

            setups_list = os.listdir(setups_dir)
            for idx_s, setup in enumerate(setups_list):
                # Set arguments for training
                self.vars["mode"] = "test"

                self.vars["log_prefix"] = os.path.join(self.execute_log_dir, design, setup)

                train_data_dir = os.path.join(setups_dir, setup)
                if not os.path.exists(train_data_dir):
                    os.makedirs(train_data_dir)
                self.vars["data_dir"] = train_data_dir

                with open(os.path.join(train_data_dir, "train", "labels", "mpii_style.json"), "r", encoding="utf-8") as fread:
                    labels_dict = json.load(fread)
                    num_mark = len(labels_dict[0]["joints_vis"])
                self.vars["num_mark"] = num_mark
                
                test_report_dir = os.path.join(reports_dir, setup)
                if not os.path.exists(test_report_dir):
                    os.makedirs(test_report_dir)
                
                self.args.ckpt_dir = os.path.join(test_report_dir, "checkpoint")
                self.args.log_dir = os.path.join(test_report_dir, "log")
                self.args.result_dir = os.path.join(test_report_dir, "result")

                # Initiate test loop
                test(args=self.args)
    
    def evaluations(self):
        eval_results = {}
        for idx_d, design in enumerate(tqdm.tqdm(self.test_design_list)):
            setups_dir = os.path.join(self.datasets_dir, design)
            reports_dir = os.path.join(self.test_report_dir, design)

            setups_list = os.listdir(setups_dir)
            for idx_s, setup in enumerate(setups_list):
                # Set arguments for evaluation
                self.vars["mode"] = "test"

                test_data_dir = os.path.join(setups_dir, setup)
                self.vars["data_dir"] = test_data_dir

                with open(os.path.join(test_data_dir, "test", "labels", "mpii_style.json"), "r", encoding="utf-8") as fread:
                    labels_dict = json.load(fread)
                    num_mark = len(labels_dict[0]["joints_vis"])
                self.vars["num_mark"] = num_mark

                self.args.ckpt_dir = os.path.join(reports_dir, setup, "checkpoint")

                # Evaluate
                evals = evaluate(args=self.args)
                eval_results["%s" % design+"-"+setup] = evals

        save_dir = os.path.join(self.test_report_dir, f"evaluation_{self.args.mode}_dataset.json")
        with open(save_dir, "w", encoding = "UTF-8-SIG") as file:
            json.dump(eval_results, file, ensure_ascii=False)
        
        avg_acc_array = np.zeros((4, len(evals)))

        for i, (key, result) in enumerate(tqdm.tqdm(eval_results.items())):
            for j, snip in enumerate(result):
                avg_acc = snip["avg_acc"]
                avg_acc_array[i, j] = avg_acc
        
        for idx_d, design in enumerate(tqdm.tqdm(self.test_design_list)):
            x = np.array(list(range(len(evals))))

            unity_mean = np.mean(avg_acc_array[2*idx_d+1,:])
            gan_mean = np.mean(avg_acc_array[2*idx_d,:])

            plt.figure(figsize=(15, 5))
            plt.title(f"{design}, {self.args.mode} data")
            plt.plot(x, avg_acc_array[1,:], color="tab:blue", alpha=0.6, label="Unity")
            plt.plot(x, avg_acc_array[0,:], color="tab:orange", alpha=0.6, label="GAN")
            plt.axhline(y=unity_mean, color="tab:blue", alpha=1, linestyle="dotted", label=f"mean={unity_mean}")
            plt.axhline(y=gan_mean, color="tab:orange", alpha=1, linestyle="dotted", label=f"mean={gan_mean}")
            plt.xlabel('Data index')
            plt.ylabel('Average accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.test_report_dir, f"{design}_{self.args.mode}.png"), dpi=300)

            difference = avg_acc_array[2*idx_d+1,:] - avg_acc_array[2*idx_d,:]
            mean_diff = np.mean(difference)

            plt.figure(figsize=(15, 5))
            plt.title(f"{self.args.mode}, {self.args.mode} data")
            plt.plot(x, difference, color="tab:green", alpha=0.6, label="Unity - GAN")
            plt.axhline(y=mean_diff, color="tab:green", alpha=1, linestyle="dotted", label=f"mean={mean_diff}")
            plt.xlabel('Data index')
            plt.ylabel('Mean accuracy difference')
            plt.legend()
            plt.savefig(os.path.join(self.test_report_dir, f"{design}_{self.args.mode}_diff.png"), dpi=300)

    def train_spec(self):
        design = self.args.spec
        setups_dir = os.path.join(self.datasets_dir, design)
        if os.path.exists(setups_dir):
            reports_dir = os.path.join(self.test_report_dir, design)
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)

            setups_list = os.listdir(setups_dir)
            for idx_s, setup in enumerate(setups_list):
                # Set arguments for training
                self.vars["mode"] = "train"

                self.vars["log_prefix"] = os.path.join(self.execute_log_dir, design, setup)

                train_data_dir = os.path.join(setups_dir, setup)
                if not os.path.exists(train_data_dir):
                    os.makedirs(train_data_dir)
                self.vars["data_dir"] = train_data_dir

                with open(os.path.join(train_data_dir, "train", "labels", "mpii_style.json"), "r", encoding="utf-8") as fread:
                    labels_dict = json.load(fread)
                    num_mark = len(labels_dict[0]["joints_vis"])
                self.vars["num_mark"] = num_mark

                num_epoch = self.args.num_epoch
                
                train_report_dir = os.path.join(reports_dir, setup)
                if not os.path.exists(train_report_dir):
                    os.makedirs(train_report_dir)
                
                self.args.ckpt_dir = os.path.join(train_report_dir, "checkpoint")
                self.args.log_dir = os.path.join(train_report_dir, "log")
                self.args.result_dir = os.path.join(train_report_dir, "result")
                
                # Initiate train loop
                train(args=self.args)
        else:
            print(f"Designated Spec {design} Does not Exists!")