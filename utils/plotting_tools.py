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

Experiment=namedtuple('Experiment', ["data_dict", "algo", "seed"])

class LogProc:

    def __init__(self, log_dirs):
        """
        log_dirs     each directoy should contain progress.csv and a info.txt file 
        """

        self.experiments=defaultdict(list)#algo:expermient_list

        self.n_envs=-1
        for dn in log_dirs:
            self._read(dn)


        self._fontsize_labels=25
        self._fontsize_ticks=20
        self._linewidth=8
        self._legent_fs=20





    def _read(self,dir_name):

        csv_file=dir_name+"/progress.csv"
        info_file=dir_name+"/info.txt"

        print(f"reading {csv_file} and {info_file}")

        data = defaultdict(list)
        
        with open(csv_file, newline='') as csvfile:
            first_line = csvfile.readline()
            headers = first_line.lstrip('#').strip().split(',')
            reader = csv.DictReader(csvfile, fieldnames=headers)
           
            row_counter=0
            for row in reader:
                if not row_counter:
                    row_counter+=1
                    continue #because the first log is always missing some info
                
                for key, value in row.items():
                    data[key].append(float(value))
                row_counter+=1

        progress_data=dict(data)

        with open(info_file,"r") as fl:
            info=json.load(fl)

        self.experiments[info["algo"]].append(Experiment(progress_data,info["algo"],info["seed"]))

        if self.n_envs==-1:
            self.n_envs=info["num_envs"]
        else:
            assert info["num_envs"]==self.n_envs, "experiments should have the same num_env for fair comparison"


    def plot_reward(self):

        for horizontal_axis in ["n_steps", "n_updates"]:
            self._plot_reward(horizontal_axis)

    def _plot_reward(self,horizontal_axis:str):
        """

        """

        for algo in self.experiments.keys():
      
            rew_avg_lst=[]
            nn=len(self.experiments[algo])
            for idx in range(nn):

                experiment=self.experiments[algo][idx]
                rew_avg_lst.append(experiment.data_dict["rollout/ep_rew_mean"])

            rew_avg_lst_sorted=sorted(enumerate(rew_avg_lst), key=lambda x: len(x[1]))
            
            max_len=len(rew_avg_lst_sorted[-1][-1])
            max_idx=rew_avg_lst_sorted[-1][0]
            rew_avg_mat=np.zeros([nn,max_len])

            for ii in range(nn):
                rew_avg_mat[ii,:len(rew_avg_lst[ii])]=rew_avg_lst[ii]

            mean_rew_avg_across_experiments=rew_avg_mat.mean(0)
            std_rew_avg_across_experiments=rew_avg_mat.std(0)

            

            if horizontal_axis=="n_steps":
                timesteps=self.experiments[algo][max_idx].data_dict["time/total_timesteps"]
                domain=[x/1e6 for x in timesteps]
                plt.plot(domain,mean_rew_avg_across_experiments,linewidth=self._linewidth,label=algo)
                plt.fill_between(domain,
                        mean_rew_avg_across_experiments - std_rew_avg_across_experiments,
                        mean_rew_avg_across_experiments + std_rew_avg_across_experiments,
                        alpha=0.3)
                plt.xlabel('Environment timesteps (millions)', fontsize=self._fontsize_labels)
                plt.ylabel(f'Mean reward \n(average over last {self.n_envs} environments)', fontsize=self._fontsize_labels)
                plt.tick_params(axis='both', labelsize=self._fontsize_ticks)
            elif horizontal_axis=="n_updates":
                
                n_updates=self.experiments[algo][max_idx].data_dict["train/n_updates"]
                plt.plot(n_updates,mean_rew_avg_across_experiments,linewidth=self._linewidth,label=algo)
                plt.fill_between(n_updates,
                        mean_rew_avg_across_experiments - std_rew_avg_across_experiments,
                        mean_rew_avg_across_experiments + std_rew_avg_across_experiments,
                        alpha=0.3)
                plt.xlabel('Number of updates', fontsize=self._fontsize_labels)
                plt.ylabel(f'Mean reward \n(average over last {self.n_envs} environments)', fontsize=self._fontsize_labels)
                plt.tick_params(axis='both', labelsize=self._fontsize_ticks)

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.grid("on")
        plt.legend(fontsize=self._legent_fs)
        plt.show()



        
       
if __name__=="__main__":

    _parser = argparse.ArgumentParser(description="Plotting from logs. Given a list of dirs with info.txt and progress.csv, this script will group each log by method and plot and avg+std shading plot")
    _parser.add_argument('--dirs', nargs=argparse.REMAINDER,required=True,help="A list of directories. Each one must contain info.txt and progress.csv")

    _args = _parser.parse_args()

    _lp=LogProc(_args.dirs)
    _lp.plot_reward()

