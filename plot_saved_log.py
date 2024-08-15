# %% [markdown]
# ### Start

# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re

# %%
global_historys = list()
saved_files = list()
plot_labels = list()

# %%
def float_range(mini, maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

# %%
parser = argparse.ArgumentParser(prog="plot_saved_log.py 'file1.npz' 'label1' 'file2.npz' 'label2' ... ",
                                 description="Plotting saved log from '.npz' format",
                                 epilog="Written by Ali Bozorgzad for comparing results")

parser.add_argument("--loss_title", "-o", dest="loss_title", type=str, default="Loss on data",
                    help="plot title for loss, put it between \"name\"")
parser.add_argument("--acc_title", "-a", dest="acc_title", type=str, default="Accuracy on data",
                    help="plot title for accuracy, put it between \"name\"")
parser.add_argument("--xlabel", "-x", dest="xlabel", type=str, default="Epochs",
                    help="set the xlabel for plot.")
parser.add_argument("--ylabel", "-y", dest="ylabel", type=str, default="Loss/Acc",
                    help="set the ylabel for plot.")
parser.add_argument("--root_path", "-p", dest="root_path", type=str, default="save_log",
                    help="root path of the '*.npz' files, put it between \"name\"")
parser.add_argument("--xlim_left", "-l", dest="xlim_left", type=float, default="None",
                    help="set x limit left for plot.")
parser.add_argument("--xlim_right", "-r", dest="xlim_right", type=float, default="None",
                    help="set x limit right for plot.")
parser.add_argument("--ylim_bottom", "-b", dest="ylim_bottom", type=float, default="None",
                    help="set y limit bottom for plot.")
parser.add_argument("--ylim_top", "-t", dest="ylim_top", type=float, default="None",
                    help="set y limit top for plot.")
parser.add_argument("--plot_save_name", "-n", dest="plot_save_name", type=str, default="NONE",
                    help="plots will be saved, if you set this, put it between \"name\"")
parser.add_argument("--show_plot", "-s", dest="show_plot", type=int, default="1",
                    help="set '0', if u want to not showing plot", choices=[0, 1])
parser.add_argument("--plot_dpi", "-d", dest="plot_dpi", type=float_range(10, 1000), default="100",
                    help="set dpi for plotting figures, between [10...1000]")
parser.add_argument("--colors", "-c", dest="colors", type=lambda y:re.split(' |, ', y), default="NONE",
                    help="set color for each plot in order, like: \"red, #00FF00, b, C1, C2, ...\"")


args, unknown = parser.parse_known_args()
loss_title = args.loss_title
acc_title = args.acc_title
xlabel = args.xlabel
ylabel = args.ylabel
root_path = args.root_path
xlim_left = args.xlim_left
xlim_right = args.xlim_right
ylim_bottom = args.ylim_bottom
ylim_top = args.ylim_top
plot_save_name = args.plot_save_name
show_plot = args.show_plot
plot_dpi = args.plot_dpi
colors = args.colors
matplotlib.rcParams["figure.dpi"] = plot_dpi


for i, arg in enumerate(unknown):
    if i % 2 == 0:
        saved_files.append(arg)
    else:
        plot_labels.append(arg)

# %%
# set parameters (Manual)
saved_files = list()
plot_labels = list()

saved_files.append("FSS_MNIST_MLP1_cka_linear_best_10c_32b_1.0cp_1.0sp_normal_1rs_0.001lr_1ce_1pes_5_3_step_1049.npz")
saved_files.append("FSS_MNIST_MLP1_cka_rbf_best_10c_32b_1.0cp_1.0sp_normal_1rs_0.001lr_1ce_1pes_5_3_step_1049.npz")
saved_files.append("FSS_MNIST_MLP1_dcka_best_10c_32b_1.0cp_1.0sp_normal_1rs_0.001lr_1ce_1pes_5_3_step_1049.npz")
saved_files.append("FSS_MNIST_MLP1_sum_diff_best_10c_32b_1.0cp_1.0sp_normal_1rs_0.001lr_1ce_1pes_5_3_step_1049.npz")
saved_files.append("FSS_MNIST_MLP1_cca_best_10c_32b_1.0cp_1.0sp_normal_1rs_0.001lr_1ce_1pes_5_3_step_1049.npz")
saved_files.append("FA_MNIST_MLP1_10c_32b_1.0cp_normal_1rs_0.001lr_1ce_step_1049.npz")
saved_files.append("FS_MNIST_MLP1_10c_32b_1.0cp_1.0sp_normal_1rs_0.001lr_1ce_1pes_5_3_step_1049.npz")

plot_labels.append("linear")
plot_labels.append("rbf")
plot_labels.append("dcka")
plot_labels.append("diff")
plot_labels.append("cca")
plot_labels.append("FA")
plot_labels.append("FS")

colors = ["NONE"]
loss_title = "Loss on data"
acc_title = "Accuracy on data"
xlabel = "Epochs"
ylabel = "Loss/Acc"
root_path = r"D:\SSD_Optimization\User\Desktop\save_log\MNIST_log"
xlim_left = None
xlim_right = None
ylim_bottom = None
ylim_top = None
plot_save_name = "NONE"
show_plot = 1
matplotlib.rcParams["figure.dpi"] = 100

# %%
for sf in saved_files:
    file_path = os.path.join(root_path, sf)
    file_name = sorted(glob.glob(file_path+"*"), key=os.path.getmtime)
    if len(file_name) == 0:
        raise Exception(f"File not found! '{file_path}'")
    
    print(file_name[0].split("\\")[-1])
    npzFile = np.load(file_name[0], allow_pickle=True)
    global_historys.append(npzFile["global_history"].item())
    npzFile.close()

# %%
if colors[0] != "NONE":
    if len(colors) != len(saved_files):
        raise Exception("The length of the colors and files must be the same.")

# %% [markdown]
# ### Loss

# %%
for i, gh in enumerate(global_historys):
    length = len(gh["loss"])
    if colors[0] == "NONE":
        # or can fill the list with 'colors = ["C1", "C2", ...]'
        plt.plot(range(length), gh["loss"], label=plot_labels[i])
    else:
        plt.plot(range(length), gh["loss"], label=plot_labels[i], color=colors[i])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(loss_title)
plt.xlim(left=xlim_left, right=xlim_right)
plt.ylim(bottom=ylim_bottom, top=ylim_top)
plt.legend()

if plot_save_name != 'NONE':
    file_path = os.path.join(root_path, plot_save_name)
    plt.savefig(f"{file_path}_loss.jpg")

if show_plot:
    plt.show()

# %% [markdown]
# ### Acc

# %%
plt.clf() # clear the figure

for i, gh in enumerate(global_historys):
    length = len(gh["accuracy"])
    if colors[0] == "NONE":
        # or can fill the list with 'colors = ["C1", "C2", ...]'
        plt.plot(range(length), gh["accuracy"], label=plot_labels[i])
    else:
        plt.plot(range(length), gh["accuracy"], label=plot_labels[i], color=colors[i])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(acc_title)
plt.xlim(left=xlim_left, right=xlim_right)
plt.ylim(bottom=ylim_bottom, top=ylim_top)
plt.legend()

if plot_save_name != 'NONE':
    file_path = os.path.join(root_path, plot_save_name)
    plt.savefig(f"{file_path}_accuracy.jpg")
    
if show_plot:
    plt.show()

# %%



