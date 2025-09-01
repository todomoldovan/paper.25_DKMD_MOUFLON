import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pickle
import networkx as nx
import gc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import re
import numpy as np

# Globals for paths
obj_path="../data/obj"
log_path="../logs"
path_5e06="../logs/5e-06"
path_1e03="../logs/1e-03"
path_10000="../logs/10000"
path_1e03x10="../logs/1e-03x10"
path_10000x10="../logs/10000x10"
plot_path="../plots/"


def plot_multiple_alpha_real(networks, draw_error=True, filename="figureB1_combined"):
	import matplotlib.pyplot as plt
	import seaborn as sns
	import matplotlib.lines as mlines
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	from matplotlib.ticker import MaxNLocator
	import pandas as pd
	import numpy as np
	import string

	sns.set_style("whitegrid")

	def plot_block(subset, fig_index):
		num_networks = len(subset)
		nrows = 4
		ncols = 2 * num_networks  # Each dataset uses 2 columns (FairMod + MOUFLON)
		fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 12), sharey=True)
		axs = axs if isinstance(axs, np.ndarray) else np.array([[axs]])
		axs = axs.reshape(nrows, ncols)

		base_strategies = ["base", "fexp", "diversity", "fmody"]
		step2_strategies = ["step2", "hybrid", "step2div", "step2fmody"]
		strategy_info = {
			"base":        {"color": "tab:red",    "marker": "o", "label": "FairMod"},
			"fexp":        {"color": "tab:green",  "marker": "^", "label": "Fexp"},
			"diversity":   {"color": "tab:orange", "marker": "D", "label": "Diversity"},
			"fmody":       {"color": "tab:cyan",   "marker": "s", "label": "ModF"},
			"step2":       {"color": "tab:red",    "marker": "o", "label": "Step2"},
			"hybrid":      {"color": "tab:green",  "marker": "^", "label": "Step2Fexp"},
			"step2div":    {"color": "tab:orange", "marker": "D", "label": "Step2Diversity"},
			"step2fmody":  {"color": "tab:cyan",   "marker": "s", "label": "Step2ModF"},
		}

		for dataset_idx, network in enumerate(subset):
			df = pd.read_csv(f"{log_path}/{network}.csv", header=0)
			left_col = dataset_idx * 2
			right_col = left_col + 1

			# Plot base strategies
			for i, strategy in enumerate(base_strategies):
				ax = axs[i, left_col]
				config = strategy_info[strategy]
				color = config["color"]
				marker = config["marker"]

				df_strat = df[df["strategy"] == strategy]
				fairness_col = (
					"fair_div" if "div" in strategy else
					"fair_modf" if "fmod" in strategy else
					"fair_bal" if strategy == "base" else
					"fair_exp"
				)
				fairness_std_col = fairness_col + "_std"

				# Fairness line
				ax.plot(df_strat["alpha"], df_strat[fairness_col],
						color=color, linestyle="--", marker=marker,
						markerfacecolor=color, markeredgecolor=color)
				if draw_error:
					ax.errorbar(df_strat["alpha"], df_strat[fairness_col],
								yerr=df_strat[fairness_std_col],
								fmt='none', ecolor=color, capsize=1)

				# Modularity
				ax.plot(df_strat["alpha"], df_strat["modularity"],
						color="tab:blue", linestyle="-", marker="x")
				if draw_error:
					ax.errorbar(df_strat["alpha"], df_strat["modularity"],
								yerr=df_strat["modularity_std"],
								fmt='none', ecolor="tab:blue", capsize=1)

				ax.set_xticks(sorted(df_strat["alpha"].unique()))
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))

				# Inset: number of communities
				inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
				inset.plot(df_strat["alpha"], df_strat["ncomms"],
						   color="tab:purple", linestyle="-", marker="X")
				if draw_error:
					inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
								   yerr=df_strat["ncomms_std"],
								   fmt='none', ecolor="tab:purple", capsize=1)
				inset.tick_params(axis='both', labelsize=7)
				inset.set_xticklabels([])

			# Plot step2 strategies
			for i, strategy in enumerate(step2_strategies):
				ax = axs[i, right_col]
				config = strategy_info[strategy]
				color = config["color"]
				marker = config["marker"]

				df_strat = df[df["strategy"] == strategy]
				fairness_col = (
					"fair_div" if "div" in strategy else
					"fair_modf" if "fmod" in strategy else
					"fair_bal" if strategy == "step2" else
					"fair_exp"
				)
				fairness_std_col = fairness_col + "_std"

				ax.plot(df_strat["alpha"], df_strat[fairness_col],
						color=color, linestyle="--", marker=marker,
						markerfacecolor=color, markeredgecolor=color)
				if draw_error:
					ax.errorbar(df_strat["alpha"], df_strat[fairness_col],
								yerr=df_strat[fairness_std_col],
								fmt='none', ecolor=color, capsize=1)

				ax.plot(df_strat["alpha"], df_strat["modularity"],
						color="tab:blue", linestyle="-", marker="x")
				if draw_error:
					ax.errorbar(df_strat["alpha"], df_strat["modularity"],
								yerr=df_strat["modularity_std"],
								fmt='none', ecolor="tab:blue", capsize=1)

				ax.set_xticks(sorted(df_strat["alpha"].unique()))
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))

				inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
				inset.plot(df_strat["alpha"], df_strat["ncomms"],
						   color="tab:purple", linestyle="-", marker="X")
				if draw_error:
					inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
								   yerr=df_strat["ncomms_std"],
								   fmt='none', ecolor="tab:purple", capsize=1)
				inset.tick_params(axis='both', labelsize=7)
				inset.set_xticklabels([])

			# Subfigure label (A, B, …)
			label_ax = axs[0, left_col]
			label_ax.text(-0.25, 1.05, string.ascii_uppercase[dataset_idx],
						  transform=label_ax.transAxes,
						  fontsize=16, fontweight="bold",
						  va="top", ha="center")

			# Titles for Fair-mod and MOUFLON columns
			axs[0, left_col].set_title("Fair-mod", fontsize=12)
			axs[0, right_col].set_title("MOUFLON", fontsize=12)

		# Global labels
		fig.supxlabel("alpha")
		fig.supylabel("Score")

		# Shared legend (modularity, fairness variants, #comms)
		handles = [
			mlines.Line2D([], [], color="tab:blue", marker="x", linestyle="-", label="Modularity"),
			mlines.Line2D([], [], color="tab:red", marker="o", linestyle="--", label="balance"),
			mlines.Line2D([], [], color="tab:green", marker="^", linestyle="--", label="prop_balance"),
			mlines.Line2D([], [], color="tab:orange", marker="D", linestyle="--", label="diversity"),
			mlines.Line2D([], [], color="tab:cyan", marker="s", linestyle="--", label="mod_fairness"),
			mlines.Line2D([], [], color="tab:purple", marker="X", linestyle="-", label="Number of communities")
		]
		fig.legend(handles=handles, loc="upper center", ncol=3)
		fig.tight_layout(rect=[0, 0, 1, 0.93])
		fig.savefig(f"../plots/{filename}_{fig_index}.png", dpi=300)
		plt.close(fig)

	# Split into two plots
	plot_block(networks[:3], fig_index=1)
	plot_block(networks[3:], fig_index=2)


	
# Function to plot quality graphs from CSV files. Plots 6 subplots for different alpha values.
def plot_quality_graph(draw_error=True):
	full_path="quality/full/color-full_1000_r01_K2_c0"
	node_path="quality/node/color-node_1000_r01_K2_c0"
	
	full_df = pd.DataFrame()
	node_df = pd.DataFrame()
	
	for prob in range(1, 6):
		df1 = pd.read_csv(f"{log_path}/{full_path}{prob}.csv", header=0)
		df1["red_prob"] = prob * 0.1
		full_df = pd.concat([full_df, df1], ignore_index=True)
		df2 = pd.read_csv(f"{log_path}/{node_path}{prob}.csv", header=0)
		df2["red_prob"] = prob * 0.1
		node_df = pd.concat([node_df, df2], ignore_index=True)

	fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 10), sharey=True)
	sns.set_style("whitegrid")

	base_strategies = ["base", "fexp", "diversity", "fmody"]
	step2_strategies = ["step2", "hybrid", "step2div", "step2fmody"]
	alpha_values = [0.25, 0.5, 0.75]
	axes_map = {
		0.25: (ax1, ax2),
		0.5: (ax3, ax4),
		0.75: (ax5, ax6),
	}
	

	strategy_info = {
		"base":        {"color": "tab:red",    "marker": "o", "label": "FairMod"},
		"fexp":        {"color": "tab:green",  "marker": "^", "label": "Fexp"},
		"diversity":   {"color": "tab:orange", "marker": "D", "label": "Diversity"},
		"fmody":       {"color": "tab:cyan",   "marker": "s", "label": "ModF"},
		"step2":       {"color": "tab:red",    "marker": "o", "label": "Step2"},
		"hybrid":      {"color": "tab:green",  "marker": "^", "label": "Step2Fexp"},
		"step2div":    {"color": "tab:orange", "marker": "D", "label": "Step2Diversity"},
		"step2fmody":  {"color": "tab:cyan",   "marker": "s", "label": "Step2ModF"},
	}
	
	def plot_alpha_strategy(df, name):
		for alpha in alpha_values:
			alpha_df = df[df["alpha"] == alpha]
			ax_base, ax_step2 = axes_map[alpha]
			
			for strategy in df["strategy"].unique():
				if strategy not in strategy_info:
					continue
				
				is_base = strategy in base_strategies
				ax = ax_base if is_base else ax_step2
				color = strategy_info[strategy]["color"]
				marker = strategy_info[strategy]["marker"]
				label = strategy_info[strategy]["label"]
				fill_style = 'none' if is_base else color

				df_strat = alpha_df[alpha_df["strategy"] == strategy]
				fairness_col = (
					"fair_div_paper" if "div" in strategy else
					"fair_modf" if "fmod" in strategy else
					"fair_bal" if strategy in ["base", "step2"] else
					"fair_exp"
				)
				fairness_std_col = fairness_col + "_std"
				
				# print(f"Plotting {strategy} with {fairness_col}")

				ax.plot(
					df_strat["red_prob"],
					df_strat[fairness_col],
					marker=marker,
					linestyle="-",
					label=label,
					color=color,
					markerfacecolor=fill_style,
					markeredgecolor=color,
				)

				if draw_error:
					ax.errorbar(
						df_strat["red_prob"],
						df_strat[fairness_col],
						yerr=df_strat[fairness_std_col],
						fmt='none',
						ecolor=color,
						capsize=1,
						elinewidth=1,
					)

		# Set titles and labels
		ax1.set_title("Base Strategies (α = 0.25)")
		ax2.set_title("Step2 Variants (α = 0.25)")
		ax3.set_title("Base Strategies (α = 0.50)")
		ax4.set_title("Step2 Variants (α = 0.50)")
		ax5.set_title("Base Strategies (α = 0.75)")
		ax6.set_title("Step2 Variants (α = 0.75)")
		
		for ax in [ax5, ax6]:
			ax.set_xlabel("Red Node Probability")
		for ax in [ax1, ax3, ax5]:
			ax.set_ylabel("Fairness Score")
		for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
			ax.set_xticks(sorted(df["red_prob"].unique()))
			ax.set_ylim(0.0, 1.1)

		# Combine legend
		handles, labels = [], []
		for strat in base_strategies + step2_strategies:
			info = strategy_info[strat]
			fill = 'none' if strat in base_strategies else info["color"]
			line = mlines.Line2D([], [], color=info["color"], marker=info["marker"],
								markerfacecolor=fill, markeredgecolor=info["color"],
								label=info["label"], linestyle='-')
			handles.append(line)

		fig.legend(handles=handles, loc="upper center", ncol=4)
		fig.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for legend
		fig.savefig(f"../plots/quality_{name}.png", dpi=300)
		plt.clf()
		plt.close(fig)
		
	plot_alpha_strategy(full_df, "full")
	# plot_alpha_strategy(node_df, "node")

#######################
### Scalability 
#######################

## Figure 2
# Function to create a DataFrame from a CSV file with additional columns for nodes and density probability for performance plotting.
def create_time_df(filename, nodes, density_prob, path):
	df = pd.read_csv(f"{path}/{filename}.csv", header=0)
	df["nodes"] = nodes
	df["density"] = density_prob
	df['nodes'] = df['nodes'].astype(str)
	# df['density'] = df['density'].astype(str)
	# print(df.head())
	return df    

def plot_scalability(df1, prop1, df2, prop2, filename="figure1"):
	# Axis titles per property
	xtitles = {
		"nodes": "Number of nodes",
		"density": "Edge probability"
	}

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharey=True)

	base_strategies = ["base", "fexp", "diversity", "fmody"]
	step2_strategies = ["step2", "hybrid", "step2div", "step2fmody"]

	strategy_info = {
		"base":        {"color": "tab:red",    "marker": "o", "label": "Fair-mod (balance)"},
		"fexp":        {"color": "tab:green",  "marker": "^", "label": "Fair-mod (prop_balance)"},
		"diversity":   {"color": "tab:orange", "marker": "D", "label": "Fair-mod (diversity)"},
		"fmody":       {"color": "tab:cyan",   "marker": "s", "label": "Fair-mod (mod_fairness)"},
		"step2":       {"color": "tab:red",    "marker": "o", "label": "MOUFLON (balance)"},
		"hybrid":      {"color": "tab:green",  "marker": "^", "label": "MOUFLON (prop_balance)"},
		"step2div":    {"color": "tab:orange", "marker": "D", "label": "MOUFLON (diversity)"},
		"step2fmody":  {"color": "tab:cyan",   "marker": "s", "label": "MOUFLON (mod_fairness)"},
	}

	panel_map = {
		(0, 0): (df1, prop1, base_strategies),
		(0, 1): (df1, prop1, step2_strategies),
		(1, 0): (df2, prop2, base_strategies),
		(1, 1): (df2, prop2, step2_strategies),
	}

	subfig_labels = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}

	for (i, j), (df, prop, strategies) in panel_map.items():
		ax = axs[i][j]
		for strategy in df["strategy"].unique():
			if strategy not in strategies:
				continue

			info = strategy_info[strategy]
			df_strat = df[df["strategy"] == strategy]
			fill_style = 'none' if strategy in base_strategies else info["color"]

			ax.plot(
				df_strat[prop],
				df_strat["time"],
				marker=info["marker"],
				linestyle="-",
				label=info["label"],
				color=info["color"],
				markerfacecolor=fill_style,
				markeredgecolor=info["color"],
			)

			if "time_std" in df_strat.columns:
				ax.errorbar(
					df_strat[prop],
					df_strat["time"],
					yerr=df_strat["time_std"],
					fmt='none',
					ecolor=info["color"],
					capsize=1,
					elinewidth=1,
				)

		ax.set_xlabel(xtitles.get(prop, prop))
		# if j == 0:
		# 	ax.set_ylabel("Execution time (s)")
		ax.set_xticks(sorted(df[prop].unique()))
		ax.set_yscale("log")
		ax.set_ylim(bottom=0)
		ax.set_title("")

		# Subfigure label
		ax.text(
			0.01, 0.95, subfig_labels[(i, j)],
			transform=ax.transAxes,
			fontsize=14,
			fontweight="bold",
			va="top",
			ha="left"
		)

	# Unified legend
	handles = []
	for strat in base_strategies + step2_strategies:
		info = strategy_info[strat]
		fill = 'none' if strat in base_strategies else info["color"]
		line = mlines.Line2D([], [], color=info["color"], marker=info["marker"],
							 markerfacecolor=fill, markeredgecolor=info["color"],
							 label=info["label"], linestyle='-')
		handles.append(line)

	fig.legend(handles=handles, loc="upper center", ncol=4)
	fig.supylabel("Execution time (s)")

	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)



# Function to get stats from a network object and a log file.
def get_stats(network, log_file):
	net=None
	# Load file
	with open(f"{obj_path}/{network}.nx","rb") as g_open:
		net=pickle.load(g_open)

	print(f"Network object {network} loaded.")
	print(f"{network}: N={net.number_of_nodes()}, M={net.number_of_edges()}")
	del net
	gc.collect()
	
	df = pd.read_csv(f"{log_path}/{log_file}.csv", header=0)
	avg_time = df.groupby("strategy").agg({"time": ["mean"], "time_std": ["mean"]})
	print(avg_time)
	
	no_comms = df[df["alpha"].isin([0.0, 0.5, 1.0])][["strategy", "alpha", "ncomms", "ncomms_std"]]
	print(no_comms)




#######################
### Quality 
#######################

# Figure 3: MOUFLON. All fairness scores vs alpha. Left panel: node, right panel: clique coloring
def plot_mouflon_alpha(net_node,net_full,filename="figure3"):
	df1=pd.read_csv(f"{log_path}/quality_new/nodex10/{net_node}.csv", header=0)
	df2=pd.read_csv(f"{log_path}/quality_new/fullx10/{net_full}.csv", header=0)


	sns.set_style("whitegrid")
	df1 = df1[df1["strategy"] == "hybrid"]
	df2 = df2[df2["strategy"] == "hybrid"]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

	metric_styles = {
		"modularity":{"label":"Modularity","color":"tab:blue","marker":"P","linestyle":"-"},
		"fair_exp":{"label":"prop_balance","color":"tab:green","marker":"^","linestyle":"--"},
		"fair_bal":{"label":"balance","color":"tab:red","marker":"o","linestyle":"--"},
		"fair_modf":{"label":"mod_fairness","color":"tab:cyan","marker":"s","linestyle":"-."},
		"fair_div_paper":{"label":"diversity","color":"tab:orange","marker":"D","linestyle":"-."},
	}

	def plot_panel(ax, df, panel_title):
		alpha_vals = sorted(df["alpha"].unique())
		for metric, style in metric_styles.items():
			ax.plot(df["alpha"], df[metric],
					label=style["label"],
					color=style["color"],
					linestyle=style["linestyle"],
					marker=style["marker"])
			ax.errorbar(df["alpha"], df[metric],
						yerr=df[f"{metric}_std"],
						fmt='none', ecolor=style["color"],
						capsize=1, elinewidth=1)

		ax.set_title(panel_title)
		ax.set_xlabel("")
		ax.set_xticks(alpha_vals)
		ax.set_ylabel("")

		# Inset: number of communities (bottom right)
		inset_ax = inset_axes(ax, width="40%", height="35%", loc='lower right')
		inset_ax.plot(df["alpha"], df["ncomms"],
					  color="tab:purple", linestyle="-", marker="X", label="No. of Communities")
		inset_ax.errorbar(df["alpha"], df["ncomms"],
						  yerr=df["ncomms_std"],
						  fmt='none', ecolor="tab:purple",
						  capsize=1, elinewidth=1)
		inset_ax.set_title("", fontsize=9)
		inset_ax.tick_params(axis='y',labelsize=8)
		inset_ax.set_xticklabels([])

	plot_panel(ax1, df1, "")
	plot_panel(ax2, df2, "")

	

	subfig_labels = ["A", "B"]
	for i, ax in enumerate([ax1, ax2]):
		ax.text(0.01, 0.95, subfig_labels[i],
				transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

	fig.supxlabel("alpha")
	fig.supylabel("Score")

	# Legend from the first panel only (same styles used)
	handles, labels = ax1.get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper center", ncol=5)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)


# Figure 4: MOUFLON a=0.5, all scores vs. p_sensitive. Left panel: node, right panel: clique coloring
def extract_p_sensitive(filename):
	"""Extract p_sensitive from filename suffix like _p01.csv → 0.1"""
	match = re.search(r'_c(\d+)\.csv$', filename)
	return int(match.group(1)) / 10.0 if match else None

def load_and_prepare(files):
	"""Load and combine hybrid rows with alpha=0.5 from list of CSVs"""
	rows = []
	for file in files:
		df = pd.read_csv(file)
		p_sens = extract_p_sensitive(file)
		df = df[(df["strategy"] == "hybrid") & (df["alpha"] == 0.5)].copy()
		df["p_sensitive"] = p_sens
		rows.append(df)
	return pd.concat(rows, ignore_index=True).sort_values("p_sensitive")

def plot_mouflon_psensitive(file_list1, file_list2, filename="figure4"):
	sns.set_style("whitegrid")
	df1 = load_and_prepare(file_list1)
	df2 = load_and_prepare(file_list2)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

	metric_styles = {
		"modularity":{"label": "Modularity","color":"tab:blue","marker":"P","linestyle": "-"},
		"fair_exp":{"label": "prop_balance","color":"tab:green","marker":"^","linestyle": "--"},
		"fair_bal":{"label": "balance","color":"tab:red","marker": "o","linestyle": "--"},
		"fair_modf":{"label": "mod_fairness","color":"tab:cyan","marker":"s", "linestyle": "-."},
		"fair_div_paper":{"label": "diversity","color":"tab:orange","marker":"D", "linestyle": "-."},
	}

	def plot_panel(ax, df, title):
		for metric, style in metric_styles.items():
			ax.plot(df["p_sensitive"], df[metric],
					label=style["label"],
					color=style["color"],
					linestyle=style["linestyle"],
					marker=style["marker"])
			ax.errorbar(df["p_sensitive"], df[metric],
						yerr=df[f"{metric}_std"],
						fmt='none', ecolor=style["color"],
						capsize=1, elinewidth=1)

		ax.set_title(title)
		ax.set_xlabel("")
		ax.set_ylabel("")
		ax.set_xticks(sorted(df["p_sensitive"].unique()))

		# Inset for number of communities
		inset_ax = inset_axes(ax, width="40%", height="35%", loc='lower right')
		inset_ax.plot(df["p_sensitive"], df["ncomms"],
					  color="tab:purple", linestyle="-", marker="X", label="No. of Communities")
		inset_ax.errorbar(df["p_sensitive"], df["ncomms"],
						  yerr=df["ncomms_std"], fmt='none', ecolor="tab:purple",
						  capsize=1, elinewidth=1)
		inset_ax.set_title("", fontsize=9)
		inset_ax.tick_params(axis='y', labelsize=8)
		inset_ax.set_xticklabels([])

	plot_panel(ax1, df1, "")
	plot_panel(ax2, df2, "")

	handles, labels = ax1.get_legend_handles_labels()

	subfig_labels = ["A", "B"]
	for i, ax in enumerate([ax1, ax2]):
		ax.text(0.01, 0.95, subfig_labels[i],
				transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

	fig.supxlabel("p_sensitive")
	fig.supylabel("Score")

	fig.legend(handles, labels, loc="upper center", ncol=5)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)


# Figs. 5-6: MOUFLON x different fairness metric over alpha (Fig5) / p_sensitive (Fig6)
strategy_config = {
	"step2": {
		"label": "balance",
		"fairness": "fair_bal",
		"color": "tab:red",
		"marker": "o"
	},
	"hybrid": {
		"label": "prop_balance",
		"fairness": "fair_exp",
		"color": "tab:green",
		"marker": "^"
	},
	"step2fmody": {
		"label": "mod_fairness",
		"fairness": "fair_modf",
		"color": "tab:cyan",
		"marker": "s"
	},
	"step2div": {
		"label": "diversity",
		"fairness": "fair_div_paper",
		"color": "tab:orange",
		"marker": "D"
	}
}

# Figure 5
# def plot_strategies_alpha(net_node, filename="figure5"):
# 	df=pd.read_csv(f"{log_path}/quality/node/{net_node}.csv", header=0)

# 	sns.set_style("whitegrid")
# 	fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
# 	axs = axs.flatten()

# 	for idx, (strategy, config) in enumerate(strategy_config.items()):
# 		ax = axs[idx]
# 		df_strat = df[df["strategy"] == strategy]
# 		fair_col = config["fairness"]
# 		color = config["color"]
# 		marker = config["marker"]
# 		label = config["label"]

# 		# Modularity
# 		ax.plot(df_strat["alpha"], df_strat["modularity"],
# 				label="Modularity", color="tab:blue", linestyle="-", marker="o")
# 		ax.errorbar(df_strat["alpha"], df_strat["modularity"],
# 					yerr=df_strat["modularity_std"],
# 					fmt='none', ecolor="tab:blue", capsize=1)

# 		# Fairness
# 		ax.plot(df_strat["alpha"], df_strat[fair_col],
# 				label=config["label"], color=color, linestyle="--", marker=marker)
# 		ax.errorbar(df_strat["alpha"], df_strat[fair_col],
# 					yerr=df_strat[fair_col + "_std"],
# 					fmt='none', ecolor=color, capsize=1)

# 		ax.set_title("")
# 		ax.set_xlabel("")
# 		ax.set_ylabel("")
# 		ax.set_xticks(sorted(df_strat["alpha"].unique()))

# 		# Inset plot
# 		inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
# 		inset.plot(df_strat["alpha"], df_strat["ncomms"],
# 				   color="tab:purple", marker="X", linestyle="-")
# 		inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
# 					   yerr=df_strat["ncomms_std"],
# 					   fmt='none', ecolor="tab:purple", capsize=1)
# 		inset.set_title("# Comms", fontsize=9)
# 		inset.tick_params(axis='y', labelsize=8)

# 		ax.legend()

# 	subfig_labels = ["A", "B", "C", "D"]
# 	for i, ax in enumerate(axs):
# 		ax.text(0.01, 0.95, subfig_labels[i],
# 				transform=ax.transAxes,
# 				fontsize=14, fontweight="bold",
# 				va="top", ha="left")

# 	fig.supxlabel("alpha")
# 	fig.supylabel("Score")

# 	fig.tight_layout()
# 	fig.savefig(f"../plots/{filename}.png", dpi=300)
# 	plt.close(fig)

def plot_strategies_alpha(net_node, filename="figure5"):
	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	import matplotlib.lines as mlines

	df = pd.read_csv(f"{log_path}/quality_new/nodex10/{net_node}.csv", header=0)

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
	axs = axs.flatten()

	# Collect legend handles only once
	legend_handles = []

	for idx, (strategy, config) in enumerate(strategy_config.items()):
		ax = axs[idx]
		df_strat = df[df["strategy"] == strategy]
		fair_col = config["fairness"]
		color = config["color"]
		marker = config["marker"]
		label = config["label"]

		# Modularity
		ax.plot(df_strat["alpha"], df_strat["modularity"],
				label="Modularity", color="tab:blue", linestyle="-", marker="o")
		ax.errorbar(df_strat["alpha"], df_strat["modularity"],
					yerr=df_strat["modularity_std"],
					fmt='none', ecolor="tab:blue", capsize=1)

		# Fairness
		ax.plot(df_strat["alpha"], df_strat[fair_col],
				label=label, color=color, linestyle="--", marker=marker)
		ax.errorbar(df_strat["alpha"], df_strat[fair_col],
					yerr=df_strat[fair_col + "_std"],
					fmt='none', ecolor=color, capsize=1)

		# Optional titles and axis formatting
		ax.set_title("")
		ax.set_xlabel("")
		ax.set_ylabel("")
		ax.set_xticks(sorted(df_strat["alpha"].unique()))

		# Inset: Number of Communities
		inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
		inset.plot(df_strat["alpha"], df_strat["ncomms"],
				   color="tab:purple", marker="X", linestyle="-")
		inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
					   yerr=df_strat["ncomms_std"],
					   fmt='none', ecolor="tab:purple", capsize=1)
		inset.set_title("# Comms", fontsize=9)
		inset.tick_params(axis='y', labelsize=8)
		inset.set_xticklabels([])

		# Don't show per-panel legends
		# ax.legend()

		# Collect handles for legend
		modularity_handle = mlines.Line2D([], [], color="tab:blue", marker="o",
										  linestyle="-", label="Modularity")
		fairness_handle = mlines.Line2D([], [], color=color, marker=marker,
										linestyle="--", label=label)
		legend_handles.append((modularity_handle, fairness_handle))

	# Flatten and deduplicate handles
	flat_handles = []
	seen_labels = set()
	for h1, h2 in legend_handles:
		for h in [h1, h2]:
			if h.get_label() not in seen_labels:
				flat_handles.append(h)
				seen_labels.add(h.get_label())

	# Subfigure labels
	subfig_labels = ["A", "B", "C", "D"]
	for i, ax in enumerate(axs):
		ax.text(0.01, 0.95, subfig_labels[i],
				transform=ax.transAxes,
				fontsize=14, fontweight="bold",
				va="top", ha="left")

	# Shared axis labels
	fig.supxlabel("alpha")
	fig.supylabel("Score")

	# Add shared legend on top center
	fig.legend(handles=flat_handles, loc="upper center", ncol=4)
	fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for legend
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)



# Figure 6
def load_psensitive_dfs(files):
	dfs = []
	for f in files:
		df = pd.read_csv(f)
		p = extract_p_sensitive(f)
		df = df[df["alpha"] == 0.5].copy()
		df["p_sensitive"] = p
		dfs.append(df)
	return pd.concat(dfs, ignore_index=True).sort_values("p_sensitive")

# def plot_strategies_psensitive(files, filename="figure6"):
# 	import matplotlib.pyplot as plt
# 	import seaborn as sns
# 	from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 	df = load_psensitive_dfs(files)
# 	sns.set_style("whitegrid")
# 	fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
# 	axs = axs.flatten()

# 	for idx, (strategy, config) in enumerate(strategy_config.items()):
# 		ax = axs[idx]
# 		df_strat = df[df["strategy"] == strategy]
# 		fair_col = config["fairness"]
# 		color = config["color"]
# 		marker = config["marker"]
# 		label = config["label"]

# 		# Modularity
# 		ax.plot(df_strat["p_sensitive"], df_strat["modularity"],
# 				label="Modularity", color="tab:blue", linestyle="-", marker="o")
# 		ax.errorbar(df_strat["p_sensitive"], df_strat["modularity"],
# 					yerr=df_strat["modularity_std"],
# 					fmt='none', ecolor="tab:blue", capsize=1)

# 		# Fairness
# 		ax.plot(df_strat["p_sensitive"], df_strat[fair_col],
# 				label=config["label"], color=color, linestyle="--", marker=marker)
# 		ax.errorbar(df_strat["p_sensitive"], df_strat[fair_col],
# 					yerr=df_strat[fair_col + "_std"],
# 					fmt='none', ecolor=color, capsize=1)

# 		ax.set_title(f"{label} over p_sensitive")
# 		ax.set_xlabel("p_sensitive")
# 		ax.set_ylabel("Score")
# 		ax.set_xticks(sorted(df_strat["p_sensitive"].unique()))

# 		# Inset plot
# 		inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
# 		inset.plot(df_strat["p_sensitive"], df_strat["ncomms"],
# 				   color="tab:purple", marker="X", linestyle="-")
# 		inset.errorbar(df_strat["p_sensitive"], df_strat["ncomms"],
# 					   yerr=df_strat["ncomms_std"],
# 					   fmt='none', ecolor="tab:purple", capsize=1)
# 		inset.set_title("", fontsize=9)
# 		inset.tick_params(axis='y', labelsize=8)

# 		ax.legend()

# 	subfig_labels = ["A", "B", "C", "D"]
# 	for i, ax in enumerate(axs):
# 		ax.text(0.01, 0.95, subfig_labels[i],
# 				transform=ax.transAxes,
# 				fontsize=14, fontweight="bold",
# 				va="top", ha="left")

# 	fig.supxlabel("p_sensitive")
# 	fig.supylabel("Score")

# 	fig.tight_layout()
# 	fig.savefig(f"../plots/{filename}.png", dpi=300)
# 	plt.close(fig)


def plot_strategies_psensitive(files, filename="figure6"):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	import matplotlib.lines as mlines

	df = load_psensitive_dfs(files)
	sns.set_style("whitegrid")
	fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
	axs = axs.flatten()

	legend_handles = []

	for idx, (strategy, config) in enumerate(strategy_config.items()):
		ax = axs[idx]
		df_strat = df[df["strategy"] == strategy]
		fair_col = config["fairness"]
		color = config["color"]
		marker = config["marker"]
		label = config["label"]

		# Modularity
		ax.plot(df_strat["p_sensitive"], df_strat["modularity"],
				label="Modularity", color="tab:blue", linestyle="-", marker="o")
		ax.errorbar(df_strat["p_sensitive"], df_strat["modularity"],
					yerr=df_strat["modularity_std"],
					fmt='none', ecolor="tab:blue", capsize=1)

		# Fairness
		ax.plot(df_strat["p_sensitive"], df_strat[fair_col],
				label=label, color=color, linestyle="--", marker=marker)
		ax.errorbar(df_strat["p_sensitive"], df_strat[fair_col],
					yerr=df_strat[fair_col + "_std"],
					fmt='none', ecolor=color, capsize=1)

		ax.set_title("")
		ax.set_xlabel("")
		ax.set_ylabel("")
		ax.set_xticks(sorted(df_strat["p_sensitive"].unique()))

		# Inset plot: number of communities
		inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
		inset.plot(df_strat["p_sensitive"], df_strat["ncomms"],
				   color="tab:purple", marker="X", linestyle="-")
		inset.errorbar(df_strat["p_sensitive"], df_strat["ncomms"],
					   yerr=df_strat["ncomms_std"],
					   fmt='none', ecolor="tab:purple", capsize=1)
		inset.set_title("", fontsize=9)
		inset.tick_params(axis='y', labelsize=8)
		inset.set_xticklabels([])

		# No per-panel legend
		# ax.legend()

		# Collect handles
		mod_handle = mlines.Line2D([], [], color="tab:blue", marker="o",
								   linestyle="-", label="Modularity")
		fair_handle = mlines.Line2D([], [], color=color, marker=marker,
									linestyle="--", label=label)
		legend_handles.append((mod_handle, fair_handle))

	# Deduplicate handles
	flat_handles = []
	seen = set()
	for h1, h2 in legend_handles:
		for h in [h1, h2]:
			if h.get_label() not in seen:
				flat_handles.append(h)
				seen.add(h.get_label())

	# Subfigure labels
	subfig_labels = ["A", "B", "C", "D"]
	for i, ax in enumerate(axs):
		ax.text(0.01, 0.95, subfig_labels[i],
				transform=ax.transAxes,
				fontsize=14, fontweight="bold",
				va="top", ha="left")

	fig.supxlabel("p_sensitive")
	fig.supylabel("Score")

	# Shared top legend
	fig.legend(handles=flat_handles, loc="upper center", ncol=4)
	fig.tight_layout(rect=[0, 0, 1, 0.93])  # space for legend
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)



### ---------------------------------------------------

def main():

	# Plot Figure 2
	main_df = pd.DataFrame()
	for nodes in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
		df = create_time_df(f"ER_{nodes}_r0001_K2_c05",nodes,0.001,path_1e03x10)
		main_df = pd.concat([main_df, df], ignore_index=True)
	main_df1 = pd.DataFrame()
	for dp in range(1, 6):
		df1 = create_time_df(f"ER_10000_r0{dp}_K2_c05", 10000, float(dp/10),path_10000x10)
		main_df1 = pd.concat([main_df1, df1], ignore_index=True)
	plot_scalability(main_df,"nodes",main_df1,"density",filename="figure2")

	# Lists for all figures (p_sensitive)
	node_files=[f"{log_path}/quality_new/nodex10/color-node_1000_r01_K2_c{p_sens}.csv" for p_sens in ["01","02","03","04","05"]]
	full_files=[f"{log_path}/quality_new/fullx10/color-full_1000_r01_K2_c{p_sens}.csv" for p_sens in ["01","02","03","04","05"]]
	
	# Plot Figure 3
	plot_mouflon_alpha("color-node_1000_r01_K2_c05","color-full_1000_r01_K2_c05",filename="figure3")
	
	# Plot Figure 4
	plot_mouflon_psensitive(node_files,full_files,filename="figure4")

	# Plot Figure 5
	plot_strategies_alpha("color-node_1000_r01_K2_c05",filename="figure5")

	# Plot Figure 6
	plot_strategies_psensitive(node_files,filename="figure6")

	# Plot Figure B1
	plot_multiple_alpha_real(
		["facebook_final","deezer_final","twitch_8graphs","pokec-a_8graphs_thresh_chng","pokec-g"],
		draw_error=True,
		filename="figureB1"
	)

	# ********** Uncomment below to create different types of plots **********
	
	#plot_fairness_graph("twitch")
	
	# Plot performance of different strategies for different node counts with same density probability
	# main_df = pd.DataFrame()
	# for nodes in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000]:   # , 500000, 1000000, 2000000
	#     df = create_time_df(f"ER_{nodes}_r5e-06_K2_c05", nodes, 5e-06, path_5e06)
	#     # df = create_time_df(f"ER_{nodes}_r0001_K2_c05", nodes, 0.001, path_1e03)
	#     main_df = pd.concat([main_df, df], ignore_index=True)
	# # print(main_df.head(10))
	# plot_time_graph(main_df, property="nodes")
	
	# Plot performance of different strategies for different density probabilities with same node count
	# main_df1 = pd.DataFrame()
	# for dp in range(1, 6):
	#     df1 = create_time_df(f"ER_10000_r0{dp}_K2_c05", 10000, float(dp/10), path_10000)
	#     main_df1 = pd.concat([main_df1, df1], ignore_index=True)
	# # main_df1 = main_df1.sort_values(by=["density", "strategy"])
	# plot_time_graph(main_df1, property="density")
	
	# get_stats("twitch", "twitch")
	
	# plot_quality_graph()
	
	# plot_community_graph("color-node_1000_r01_K2_c05")
	
if __name__ == '__main__':
	main()