import numpy as np
import csv
import sys
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import json
from collections import namedtuple
import argparse
import yaml
from termcolor import colored

_fontsize_labels=25
_fontsize_ticks=20
_linewidth=8
_legent_fs=20


def plot_train_val_progress(csv_file,eval_only):

    with open(csv_file, newline='') as csvfile:
        first_line = csvfile.readline()
        headers = first_line.lstrip('#').strip().split(',')
        reader = csv.DictReader(csvfile, fieldnames=headers)

        line_count = sum(1 for _ in csv.DictReader(csvfile, fieldnames=headers))
        if line_count<3:
            msg=colored("not enough lines in the csv file yet", "red",attrs=["bold"])
            raise Exception(msg)
        csvfile.seek(0)
 
        data = defaultdict(list)

        row_counter=0
        for row in reader:
            if not row_counter:
                row_counter+=1
                continue #because the first log is always missing some info

            eval_val=row["eval/mean_reward"]
            eval_ep_len=row["eval/mean_ep_length"]
            rollout_val=row["rollout/ep_rew_mean"]
            rollout_ep_len=row["rollout/ep_len_mean"]
            total_timesteps=row["time/total_timesteps"]

            if eval_val!="":#it's an eval episode and there wont be any rollout values
                data["eval/mean_reward"].append(float(eval_val))
                data["eval/mean_ep_length"].append(float(eval_ep_len))
                data["eval_domain"].append(float(total_timesteps))
            else:
                data["rollout/ep_rew_mean"].append(float(rollout_val))
                data["rollout/mean_ep_length"].append(float(rollout_ep_len))
                data["rollout_domain"].append(float(total_timesteps))

            row_counter+=1

        progress_data=dict(data)
        #pdb.set_trace()

       
        #plot rewards
        if not eval_only:
            plt.plot(progress_data["rollout_domain"],progress_data["rollout/ep_rew_mean"],linewidth=_linewidth,label="train",color="blue")
        plt.plot(progress_data["eval_domain"],progress_data["eval/mean_reward"],linewidth=_linewidth,label="eval",color="darkorange")
        #plt.xlabel('Environment timesteps (millions)', fontsize=_fontsize_labels)
        plt.xlabel('Environment timesteps', fontsize=_fontsize_labels)
        plt.ylabel(f'Mean reward \n(average over last N environments)', fontsize=_fontsize_labels)
        plt.tick_params(axis='both', labelsize=_fontsize_ticks)
        
        #manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.grid("on")
        plt.legend(fontsize=_legent_fs,loc="upper left")
        plt.show()


        #plot ep_len
        if not eval_only:
            plt.plot(progress_data["rollout_domain"],progress_data["rollout/mean_ep_length"],linewidth=_linewidth,label="train",color="blue")
        plt.plot(progress_data["eval_domain"],progress_data["eval/mean_ep_length"],linewidth=_linewidth,label="eval",color="darkorange")
        #plt.xlabel('Environment timesteps (millions)', fontsize=_fontsize_labels)
        plt.xlabel('Environment timesteps', fontsize=_fontsize_labels)
        plt.ylabel(f'Average episode length \n(over last N environments)', fontsize=_fontsize_labels)
        plt.tick_params(axis='both', labelsize=_fontsize_ticks)
        
        #manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.grid("on")
        plt.legend(fontsize=_legent_fs,loc="upper left")
        plt.show()




def plot_loss_evolutions(csv_file, config_file):

    data = defaultdict(list)
    
    with open(csv_file, newline='') as csvfile:
        first_line = csvfile.readline()
        headers = first_line.lstrip('#').strip().split(',')
        reader = csv.DictReader(csvfile, fieldnames=headers)
       
        row_counter=0
        for row in reader:

            entropy_loss=row["train/entropy_loss"]
            value_loss=row["train/value_loss"]
            pg_loss=row["train/policy_gradient_loss"]

            if entropy_loss!="":
                data["train/entropy_loss"].append(float(entropy_loss))
            if value_loss!="":
                data["train/value_loss"].append(float(value_loss))
            if pg_loss!="":
                data["train/policy_gradient_loss"].append(float(pg_loss))

            row_counter+=1

    progress_data=dict(data)

    with open(config_file,"r") as fl:
        config = yaml.safe_load(fl)


    ent_coef=config["algo"]["ent_coef"]
    val_coef=config["algo"]["vf_coef"]

    #plt.plot([x*ent_coef for x in progress_data["train/entropy_loss"]],"r",label="entropy")
    plt.plot([x for x in progress_data["train/entropy_loss"]],"r",label="entropy")
    plt.legend(fontsize=18)
    plt.show()
    plt.plot(progress_data["train/policy_gradient_loss"],"g",label="pg loss")
    plt.legend(fontsize=18)
    plt.show()
    #plt.plot([x*val_coef for x in progress_data["train/value_loss"]],"b",label="value loss")
    plt.plot([x for x in progress_data["train/value_loss"]],"b",label="value loss")
    plt.legend(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()


    #plt.plot([x*ent_coef for x in progress_data["train/entropy_loss"]],"r",label="entropy")
    plt.plot([x for x in progress_data["train/entropy_loss"]],"r",label="entropy")
    plt.plot(progress_data["train/policy_gradient_loss"],"g",label="pg loss")
    #plt.plot([x*val_coef for x in progress_data["train/value_loss"]],"b",label="value loss")
    plt.plot([x for x in progress_data["train/value_loss"]],"b",label="value loss")
    plt.legend(fontsize=18)
    plt.show()

        
       
if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="Plotting from logs")
    _parser.add_argument('--csv', type=str,required=True,help="your csv file")
    _parser.add_argument('--config', type=str,required=True,help="your config.yaml")
    _parser.add_argument("--plot_train", action="store_true", help="also plots training stats if passed to script")

    _args = _parser.parse_args()


    plot_train_val_progress(_args.csv,eval_only=not _args.plot_train)

