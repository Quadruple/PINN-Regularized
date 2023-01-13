import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_title(data_index, array_index):
    label = ""
    if data_index == 1:
        label += "L1 Regularized "
    else:
        label += "L2 Regularized "
    if array_index == 0:
        label += "0.1"
    elif array_index == 1:
        label += "0.01"
    else:
        label += "0.001"
    return label

non_regularized_data_path = "./non_regularized/"
l1_data_path = "./l1/"
l2_data_path = "./l2/"
data_file_name = "pinn_run_output.txt"
image_save_folder = "figures/"

begin_indicator = "----------BEGIN"
iteration_indicator = "iteration"
l1_avg_indicator = "L1 AVG."
l1_rel_indicator = "L1 REL."
l2_avg_indicator = "L2 AVG."
l2_rel_indicator = "L2 REL."
w11_avg_indicator = "W11 AVG."
w11_rel_indicator = "W11 REL."

paths = [non_regularized_data_path, l1_data_path, l2_data_path]

# 0: non-regularized
# 1: l1
# 2: l2
all_obtained_data = []
for path in paths:
    # 0-9999 Lambda 0.1
    # 10000-19999 Lambda 0.01
    # 20000-29999 Lambda 0.001
    data = {
        "loss": [],
        "l1_avg": [],
        "l1_rel": [],
        "l2_avg": [],
        "l2_rel": [],
        "w11_avg": [],
        "w11_rel": []
    }

    with open(path + data_file_name) as data_file:
        content = data_file.readlines()
    for line in content:
        if begin_indicator in line:
            continue
        if iteration_indicator in line:
            data["loss"].append(float(line.split(" ")[3]))
            continue
        if l1_avg_indicator in line:
            data["l1_avg"].append(float(line.split(" ")[3]))
            continue
        if l1_rel_indicator in line:
            data["l1_rel"].append(float(line.split(" ")[3].replace("tensor(", "").replace(")", "")))
            continue
        if l2_avg_indicator in line:
            data["l2_avg"].append(float(line.split(" ")[3]))
            continue
        if l2_rel_indicator in line:
            data["l2_rel"].append(float(line.split(" ")[3].replace("tensor(", "").replace(")", "")))
            continue
        if w11_avg_indicator in line:
            data["w11_avg"].append(float(line.split(" ")[3]))
            continue
        if w11_rel_indicator in line:
            data["w11_rel"].append(float(line.split(" ")[3].replace("tensor(", "").replace(",", "")))
            continue
    all_obtained_data.append(data)

iteration_points = [x for x in range(5, 10005, 5)]
for data_index, data in enumerate(all_obtained_data):
    for key in data.keys():
        data_key_to_plot = key
        if len(data[data_key_to_plot]) == 2000:
            plt.title("Non-Regularized")
            plt.scatter(iteration_points, data[data_key_to_plot])
            plt.savefig(image_save_folder + key + "/non-regularized.jpg")
            plt.close()
        else:
            for array_index, split in enumerate(np.array_split(data[data_key_to_plot], 3)):
                plt.title(get_title(data_index, array_index))
                plt.scatter(iteration_points, split)
                plt.savefig(image_save_folder + key + "/" + get_title(data_index, array_index).replace(" ", "_").lower() + ".jpg")
                plt.close()

non_regularized_losses = all_obtained_data[0]
non_regularized_row = [ 
    non_regularized_losses["l1_avg"][-1],
    non_regularized_losses["l2_avg"][-1],
    non_regularized_losses["loss"][-1],
]

l1_regularization_results = all_obtained_data[1]
l1_avg_losses_for_each_lambda = [x[-1] for x in np.array_split(l1_regularization_results["l1_avg"], 3)]
l2_avg_losses_for_each_lambda = [x[-1] for x in np.array_split(l1_regularization_results["l2_avg"], 3)]
l_inf_losses_for_each_lambda = [x[-1] for x in np.array_split(l1_regularization_results["loss"], 3)]

l1_regularized_table = pd.DataFrame(list(zip(
    l1_avg_losses_for_each_lambda,
    l2_avg_losses_for_each_lambda,
    l_inf_losses_for_each_lambda
)), columns = ["L1", "L2", "Linf"], index=["0.1", "0.01", "0.001"])

l1_regularized_table.loc["N/A"] = non_regularized_row

print(l1_regularized_table)

l2_regularization_results = all_obtained_data[2]
l1_avg_losses_for_each_lambda = [x[-1] for x in np.array_split(l2_regularization_results["l1_avg"], 3)]
l2_avg_losses_for_each_lambda = [x[-1] for x in np.array_split(l2_regularization_results["l2_avg"], 3)]
l_inf_losses_for_each_lambda = [x[-1] for x in np.array_split(l2_regularization_results["loss"], 3)]

l2_regularized_table = pd.DataFrame(list(zip(
    l1_avg_losses_for_each_lambda,
    l2_avg_losses_for_each_lambda,
    l_inf_losses_for_each_lambda
)), columns = ["L1", "L2", "Linf"], index=["0.1", "0.01", "0.001"])

l2_regularized_table.loc["N/A"] = non_regularized_row

print(l2_regularized_table)
