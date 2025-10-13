import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pickle
import networkx as nx
import gc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re
import numpy as np

# Globals for paths
obj_path = "../data/obj"
log_path = "../logs"
path_5e06 = "../logs/5e-06"
path_1e03 = "../logs/1e-03"
path_10000 = "../logs/10000"
path_1e03x10 = "../logs/1e-03x10"
path_10000x10 = "../logs/10000x10"
plot_path = "../plots/"

##############################################
# Shared style configuration (Fair-mod dotted, MOUFLON dashed)
##############################################

STYLE = {
	"modularity": {"color": "tab:blue", "linestyle": "-", "marker": "x", "label": "Modularity"},
	"balance_fm": {"color": "tab:red", "linestyle": ":", "marker": "o", "label": "Fair-mod (balance)"},
	"prop_fm": {"color": "tab:green", "linestyle": ":", "marker": "^", "label": "Fair-mod (prop_balance)"},
	"balance_mouflon": {"color": "tab:red", "linestyle": "--", "marker": "o", "label": "MOUFLON (balance)"},
	"prop_mouflon": {"color": "tab:green", "linestyle": "--", "marker": "^", "label": "MOUFLON (prop_balance)"},
	"ncomms": {"color": "tab:purple", "linestyle": "-", "marker": "X", "label": "Number of communities"},
}

# Which fairness column + style to use for each MOUFLON strategy
strategy_config = {
	"step2":  {"fairness": "fair_bal", "style": STYLE["balance_mouflon"]},
	"hybrid": {"fairness": "fair_exp",  "style": STYLE["prop_mouflon"]},
}


##############################################
# Helper for inset axis scaling (fallback if no hardcoded limits)
##############################################

def get_ncomms_limits(dfs, strategies=None):
	ymin, ymax = float("inf"), float("-inf")
	for df in dfs:
		sub = df if strategies is None else df[df["strategy"].isin(strategies)]
		if "ncomms" in sub:
			ymin = min(ymin, (sub["ncomms"] - sub.get("ncomms_std", 0)).min())
			ymax = max(ymax, (sub["ncomms"] + sub.get("ncomms_std", 0)).max())
	if ymin == float("inf"):
		return 0.9, 1.1  # harmless default
	mean_val = (ymax + ymin) / 2
	variation = (ymax - ymin) / max(mean_val, 1)
	if variation < 0.05:
		pad = max(mean_val * 0.1, 1)   # ±10% or at least ±1
		return max(mean_val - pad, 1e-6), mean_val + pad
	pad = max((ymax - ymin) * 0.05, 0.5)
	return max(ymin - pad, 1e-6), ymax + pad

# Hardcoded inset y-limits for ncomms (all start at 1)
INSET_LIMITS = {
	"facebook_final": (1, 2700),
	"deezer_final": (1, 20000),
	"twitch_8graphs": (1, 110200),
	"pokec-a_8graphs_thresh_chng": (1, 891000),
	"pokec-g": (1, 1070500),
}

def _with_padding(lo, hi, frac=0.05):
	lo_p = max(lo, 1)  # always start from at least 1
	hi_p = hi * (1 + frac)
	return lo_p, hi_p

# def plot_multiple_alpha_real(networks, draw_error=True, filename="figureB1_combined"):
# 	import string
# 	sns.set_style("whitegrid")

# 	def plot_block(subset, fig_index):
# 		num_networks = len(subset)
# 		nrows = 2
# 		ncols = 2 * num_networks
# 		fig, axs = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 6), sharey=True)
# 		axs = axs.reshape(nrows, ncols)

# 		all_data = [(net, pd.read_csv(f"{log_path}/{net}.csv", header=0)) for net in subset]

# 		for dataset_idx, (net_name, df) in enumerate(all_data):
# 			left_col = dataset_idx * 2
# 			right_col = left_col + 1

# 			# Use hardcoded limits if available
# 			if net_name in INSET_LIMITS:
# 				inset_lo, inset_hi = _with_padding(*INSET_LIMITS[net_name], frac=0.05)
# 				use_log = True
# 			else:
# 				inset_lo, inset_hi = get_ncomms_limits([df], strategies=["base", "fexp", "step2", "hybrid"])
# 				spread_ratio = inset_hi / max(inset_lo, 1)
# 				use_log = spread_ratio > 10 and (inset_hi - inset_lo) / max(inset_hi, 1) > 0.1

# 			# --- Base/Fexp (Fair-mod)
# 			for i, strategy in enumerate(["base", "fexp"]):
# 				ax = axs[i, left_col]
# 				df_strat = df[df["strategy"] == strategy]
# 				fairness_col = "fair_bal" if strategy == "base" else "fair_exp"
# 				fairness_std_col = fairness_col + "_std"
# 				style = STYLE["balance_fm"] if strategy == "base" else STYLE["prop_fm"]

# 				ax.plot(df_strat["alpha"], df_strat[fairness_col],
# 						color=style["color"], linestyle=style["linestyle"],
# 						marker=style["marker"], markerfacecolor="none")
# 				if draw_error:
# 					ax.errorbar(df_strat["alpha"], df_strat[fairness_col],
# 								yerr=df_strat[fairness_std_col], fmt='none',
# 								ecolor=style["color"], capsize=1)

# 				ax.plot(df_strat["alpha"], df_strat["modularity"], **STYLE["modularity"])
# 				if draw_error:
# 					ax.errorbar(df_strat["alpha"], df_strat["modularity"],
# 								yerr=df_strat["modularity_std"], fmt='none',
# 								ecolor=STYLE["modularity"]["color"], capsize=1)

# 				ax.set_xticks(sorted(df_strat["alpha"].unique()))
# 				ax.set_ylim(0, 1)

# 				# Inset
# 				inset = inset_axes(ax, width="40%", height="25%", loc="lower right")
# 				inset.plot(df_strat["alpha"], df_strat["ncomms"], **STYLE["ncomms"])
# 				if draw_error:
# 					inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
# 								   yerr=df_strat["ncomms_std"], fmt='none',
# 								   ecolor=STYLE["ncomms"]["color"], capsize=1)
# 				inset.set_ylim(inset_lo, inset_hi)
# 				if use_log:
# 					inset.set_yscale("log")
# 				inset.set_xticklabels([])
# 				inset.tick_params(axis='both', labelsize=7)
# 				inset.margins(x=0.02, y=0.1)

# 			# --- Step2/Hybrid (MOUFLON)
# 			for i, strategy in enumerate(["step2", "hybrid"]):
# 				ax = axs[i, right_col]
# 				df_strat = df[df["strategy"] == strategy]
# 				fairness_col = "fair_bal" if strategy == "step2" else "fair_exp"
# 				fairness_std_col = fairness_col + "_std"
# 				style = STYLE["balance_mouflon"] if strategy == "step2" else STYLE["prop_mouflon"]

# 				ax.plot(df_strat["alpha"], df_strat[fairness_col],
# 						color=style["color"], linestyle=style["linestyle"],
# 						marker=style["marker"])
# 				if draw_error:
# 					ax.errorbar(df_strat["alpha"], df_strat[fairness_col],
# 								yerr=df_strat[fairness_std_col], fmt='none',
# 								ecolor=style["color"], capsize=1)

# 				ax.plot(df_strat["alpha"], df_strat["modularity"], **STYLE["modularity"])
# 				if draw_error:
# 					ax.errorbar(df_strat["alpha"], df_strat["modularity"],
# 								yerr=df_strat["modularity_std"], fmt='none',
# 								ecolor=STYLE["modularity"]["color"], capsize=1)

# 				ax.set_xticks(sorted(df_strat["alpha"].unique()))
# 				ax.set_ylim(0, 1)

# 				# Inset
# 				inset = inset_axes(ax, width="40%", height="25%", loc="lower right")
# 				inset.plot(df_strat["alpha"], df_strat["ncomms"], **STYLE["ncomms"])
# 				if draw_error:
# 					inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
# 								   yerr=df_strat["ncomms_std"], fmt='none',
# 								   ecolor=STYLE["ncomms"]["color"], capsize=1)
# 				inset.set_ylim(inset_lo, inset_hi)
# 				if use_log:
# 					inset.set_yscale("log")
# 				inset.set_xticklabels([])
# 				inset.tick_params(axis='both', labelsize=7)
# 				inset.margins(x=0.02, y=0.1)

# 			# Dataset letter
# 			axs[0, left_col].text(-0.25, 1.05, string.ascii_uppercase[dataset_idx],
# 								  transform=axs[0, left_col].transAxes,
# 								  fontsize=16, fontweight="bold", va="top", ha="center")

# 		fig.supxlabel("alpha")
# 		fig.supylabel("Score")

# 		handles = [
# 			mlines.Line2D([], [], **STYLE["modularity"]),
# 			mlines.Line2D([], [], color=STYLE["balance_fm"]["color"], linestyle=STYLE["balance_fm"]["linestyle"],
# 						  marker=STYLE["balance_fm"]["marker"], markerfacecolor="none", label=STYLE["balance_fm"]["label"]),
# 			mlines.Line2D([], [], color=STYLE["prop_fm"]["color"], linestyle=STYLE["prop_fm"]["linestyle"],
# 						  marker=STYLE["prop_fm"]["marker"], markerfacecolor="none", label=STYLE["prop_fm"]["label"]),
# 			mlines.Line2D([], [], **STYLE["balance_mouflon"]),
# 			mlines.Line2D([], [], **STYLE["prop_mouflon"]),
# 			mlines.Line2D([], [], **STYLE["ncomms"]),
# 		]
# 		fig.legend(handles=handles, loc="upper center", ncol=6)
# 		fig.tight_layout(rect=[0, 0, 1, 0.93])
# 		fig.savefig(f"../plots/{filename}_{fig_index}.png", dpi=300)
# 		plt.close(fig)

# 	plot_block(networks[:3], fig_index=1)
# 	plot_block(networks[3:], fig_index=2)


def plot_multiple_alpha_real(networks, draw_error=True, filename="figureB_step2hybrid"):
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
        nrows = 2  # For step2 and hybrid
        ncols = num_networks  # One column per dataset
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 6), sharey=True)

        axs = np.array(axs).reshape(nrows, ncols)

        strategies = ["step2", "hybrid"]
        strategy_info = {
            "step2": {
                "color": "tab:red",
                "marker": "o",
                "label": "Step2",
                "fairness": "fair_bal"
            },
            "hybrid": {
                "color": "tab:green",
                "marker": "^",
                "label": "Step2Fexp",
                "fairness": "fair_exp"
            }
        }

        for dataset_idx, network in enumerate(subset):
            df = pd.read_csv(f"{log_path}/fexp_correction/{network}.csv", header=0)

            for row, strategy in enumerate(strategies):
                ax = axs[row, dataset_idx]
                cfg = strategy_info[strategy]
                df_strat = df[df["strategy"] == strategy]

                # Fairness plot
                ax.plot(df_strat["alpha"], df_strat[cfg["fairness"]],
                        label=cfg["label"],
                        color=cfg["color"],
                        linestyle="--",
                        marker=cfg["marker"],
                        markerfacecolor=cfg["color"],
                        markeredgecolor=cfg["color"])

                if draw_error:
                    ax.errorbar(df_strat["alpha"], df_strat[cfg["fairness"]],
                                yerr=df_strat[cfg["fairness"] + "_std"],
                                fmt='none', ecolor=cfg["color"], capsize=1)

                # Modularity plot
                ax.plot(df_strat["alpha"], df_strat["modularity"],
                        label="Modularity",
                        color="tab:blue",
                        linestyle="-",
                        marker="x")

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

            # Add subfigure label (A, B, ...) and title
            ax_top = axs[0, dataset_idx]
            ax_top.set_title("", fontsize=12)
            ax_top.text(-0.25, 1.05, string.ascii_uppercase[dataset_idx],
                        transform=ax_top.transAxes,
                        fontsize=16, fontweight="bold",
                        va="top", ha="center")

        # Global labels
        fig.supxlabel("alpha")
        fig.supylabel("Score")

        # Common legend
        handles = [
            mlines.Line2D([], [], color="tab:blue", marker="x", linestyle="-", label="Modularity"),
            mlines.Line2D([], [], color="tab:red", marker="o", linestyle="--", label="MOUFLON (balance)"),
            mlines.Line2D([], [], color="tab:green", marker="^", linestyle="--", label="MOUFLON (prop_balance)"),
            mlines.Line2D([], [], color="tab:purple", marker="X", linestyle="-", label="Number of communities"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"../plots/{filename}_{fig_index}.png", dpi=300)
        plt.close(fig)

    # # Split into two figures
    # plot_block(networks[:3], fig_index=1)
    # if len(networks) > 3:
    #     plot_block(networks[3:], fig_index=2)
    plot_block(networks,fig_index=0)



##############################################
# Quality Graph
##############################################

def plot_quality_graph(draw_error=True):
	full_path = "quality/full_c/color-full_1000_r01_K2_c0"
	node_path = "quality/node_c/color-node_1000_r01_K2_c0"

	full_df = pd.DataFrame()
	node_df = pd.DataFrame()

	for prob in range(1, 6):
		df1 = pd.read_csv(f"{log_path}/{full_path}{prob}.csv", header=0)
		df1["red_prob"] = prob * 0.1
		full_df = pd.concat([full_df, df1], ignore_index=True)

		df2 = pd.read_csv(f"{log_path}/{node_path}{prob}.csv", header=0)
		df2["red_prob"] = prob * 0.1
		node_df = pd.concat([node_df, df2], ignore_index=True)

	fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 8), sharey=True)
	sns.set_style("whitegrid")

	base_strategies = ["base", "fexp"]
	step2_strategies = ["step2", "hybrid"]
	alpha_values = [0.25, 0.5, 0.75]
	axes_map = {0.25: (ax1, ax2), 0.5: (ax3, ax4), 0.75: (ax5, ax6)}

	def plot_alpha_strategy(df, name):
		inset_ymin, inset_ymax = get_ncomms_limits([df], strategies=base_strategies + step2_strategies)

		for alpha in alpha_values:
			alpha_df = df[df["alpha"] == alpha]
			ax_base, ax_step2 = axes_map[alpha]

			for strategy in base_strategies + step2_strategies:
				if strategy not in alpha_df["strategy"].unique():
					continue

				if strategy == "base":
					style = STYLE["balance_fm"]
				elif strategy == "fexp":
					style = STYLE["prop_fm"]
				elif strategy == "step2":
					style = STYLE["balance_mouflon"]
				else:
					style = STYLE["prop_mouflon"]

				ax = ax_base if strategy in base_strategies else ax_step2
				fairness_col = "fair_bal" if strategy in ["base", "step2"] else "fair_exp"
				fairness_std_col = fairness_col + "_std"

				if strategy in base_strategies:
					ax.plot(alpha_df["red_prob"], alpha_df[fairness_col],
							color=style["color"], linestyle=style["linestyle"],
							marker=style["marker"], markerfacecolor="none")
				else:
					ax.plot(alpha_df["red_prob"], alpha_df[fairness_col], **style)

				if draw_error:
					ax.errorbar(alpha_df["red_prob"], alpha_df[fairness_col],
								yerr=alpha_df[fairness_std_col], fmt='none',
								ecolor=style["color"], capsize=1)

				ax.plot(alpha_df["red_prob"], alpha_df["modularity"], **STYLE["modularity"])
				if draw_error:
					ax.errorbar(alpha_df["red_prob"], alpha_df["modularity"],
								yerr=alpha_df["modularity_std"], fmt='none',
								ecolor=STYLE["modularity"]["color"], capsize=1)

				ax.set_xticks(sorted(alpha_df["red_prob"].unique()))
				ax.set_ylim(0, 1)
				ax.margins(x=0.02)

				inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
				inset.plot(alpha_df["red_prob"], alpha_df["ncomms"], **STYLE["ncomms"])
				if draw_error:
					inset.errorbar(alpha_df["red_prob"], alpha_df["ncomms"],
								   yerr=alpha_df["ncomms_std"], fmt='none',
								   ecolor=STYLE["ncomms"]["color"], capsize=1)
				inset.set_ylim(inset_ymin, inset_ymax)
				inset.set_xticklabels([])
				inset.tick_params(axis='both', labelsize=7)
				inset.margins(x=0.02, y=0.05)

		fig.supxlabel("Red Node Probability")
		fig.supylabel("Score")

		handles = [
			mlines.Line2D([], [], **STYLE["modularity"]),
			mlines.Line2D([], [], color=STYLE["balance_fm"]["color"], linestyle=STYLE["balance_fm"]["linestyle"],
						  marker=STYLE["balance_fm"]["marker"], markerfacecolor="none", label=STYLE["balance_fm"]["label"]),
			mlines.Line2D([], [], color=STYLE["prop_fm"]["color"], linestyle=STYLE["prop_fm"]["linestyle"],
						  marker=STYLE["prop_fm"]["marker"], markerfacecolor="none", label=STYLE["prop_fm"]["label"]),
			mlines.Line2D([], [], **STYLE["balance_mouflon"]),
			mlines.Line2D([], [], **STYLE["prop_mouflon"]),
			mlines.Line2D([], [], **STYLE["ncomms"]),
		]
		fig.legend(handles=handles, loc="upper center", ncol=6)
		fig.tight_layout(rect=[0, 0, 1, 0.93])
		fig.savefig(f"../plots/quality_{name}.png", dpi=300)
		plt.close(fig)

	plot_alpha_strategy(full_df, "full")
	# plot_alpha_strategy(node_df, "node")

# ##############################################
# # Scalability (Figure 2) — 1×2: size vs density
# ##############################################

def create_time_df(filename, nodes, density_prob, path):
	df = pd.read_csv(f"{path}/{filename}.csv", header=0)
	df["nodes"] = nodes
	df["density"] = density_prob
	df["nodes"] = df["nodes"].astype(int)
	return df

# def plot_scalability_1x2(df_size, df_density, filename="figure2"):
# 	sns.set_style("whitegrid")
# 	fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# 	strategies = ["base", "fexp", "step2", "hybrid"]

# 	# --- Left: size (nodes)
# 	for strategy in strategies:
# 		if strategy not in df_size["strategy"].unique():
# 			continue
# 		style = STYLE["balance_fm"] if strategy == "base" else \
# 				STYLE["prop_fm"] if strategy == "fexp" else \
# 				STYLE["balance_mouflon"] if strategy == "step2" else STYLE["prop_mouflon"]

# 		df_s = df_size[df_size["strategy"] == strategy].sort_values("nodes")
# 		if strategy in ["base", "fexp"]:
# 			axL.plot(df_s["nodes"], df_s["time"],
# 					 color=style["color"], linestyle=style["linestyle"],
# 					 marker=style["marker"], markerfacecolor="none", label=style["label"])
# 		else:
# 			axL.plot(df_s["nodes"], df_s["time"], **style)
# 		if "time_std" in df_s.columns:
# 			axL.errorbar(df_s["nodes"], df_s["time"], yerr=df_s["time_std"],
# 						 fmt='none', ecolor=style["color"], capsize=1)

# 	axL.set_xlabel("Number of nodes")
# 	axL.set_xscale("log")
# 	axL.set_yscale("log")
# 	axL.set_ylim(bottom=0)
# 	axL.set_xticks(sorted(df_size["nodes"].unique()))
# 	axL.margins(x=0.02)

# 	# --- Right: density (Edge probability) — only show the densities you have
# 	for strategy in strategies:
# 		if strategy not in df_density["strategy"].unique():
# 			continue
# 		style = STYLE["balance_fm"] if strategy == "base" else \
# 				STYLE["prop_fm"] if strategy == "fexp" else \
# 				STYLE["balance_mouflon"] if strategy == "step2" else STYLE["prop_mouflon"]

# 	# plot by strategy
# 		df_s = df_density[df_density["strategy"] == strategy].sort_values("density")
# 		if strategy in ["base", "fexp"]:
# 			axR.plot(df_s["density"], df_s["time"],
# 					 color=style["color"], linestyle=style["linestyle"],
# 					 marker=style["marker"], markerfacecolor="none", label=style["label"])
# 		else:
# 			axR.plot(df_s["density"], df_s["time"], **style)
# 		if "time_std" in df_s.columns:
# 			axR.errorbar(df_s["density"], df_s["time"], yerr=df_s["time_std"],
# 						 fmt='none', ecolor=style["color"], capsize=1)

# 	axR.set_xlabel("Edge probability")
# 	axR.set_yscale("log")
# 	axR.set_ylim(bottom=0)
# 	axR.set_xticks(sorted(df_density["density"].unique()))  # ← removes 0.15, 0.25, etc.
# 	axR.margins(x=0.02)

# 	fig.supylabel("Execution time (s)")
# 	handles = [
# 		mlines.Line2D([], [], **STYLE["modularity"]),
# 		mlines.Line2D([], [], color=STYLE["balance_fm"]["color"], linestyle=STYLE["balance_fm"]["linestyle"],
# 					  marker=STYLE["balance_fm"]["marker"], markerfacecolor="none", label=STYLE["balance_fm"]["label"]),
# 		mlines.Line2D([], [], color=STYLE["prop_fm"]["color"], linestyle=STYLE["prop_fm"]["linestyle"],
# 					  marker=STYLE["prop_fm"]["marker"], markerfacecolor="none", label=STYLE["prop_fm"]["label"]),
# 		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
# 		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
# 	]
# 	fig.legend(handles=handles, loc="upper center", ncol=5)
# 	fig.tight_layout(rect=[0, 0, 1, 0.93])
# 	fig.savefig(f"../plots/{filename}.png", dpi=300)
# 	plt.close(fig)

def plot_scalability_1x2(df_size, df_density, filename="figure2_mouflon"):
	import matplotlib.pyplot as plt
	import seaborn as sns
	import matplotlib.lines as mlines

	sns.set_style("whitegrid")
	fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

	strategies = ["step2", "hybrid"]

	# --- Left: scaling with number of nodes
	for strategy in strategies:
		if strategy not in df_size["strategy"].unique():
			continue
		style = STYLE["balance_mouflon"] if strategy == "step2" else STYLE["prop_mouflon"]
		df_s = df_size[df_size["strategy"] == strategy].sort_values("nodes")
		axL.plot(df_s["nodes"], df_s["time"], **style)
		if "time_std" in df_s.columns:
			axL.errorbar(df_s["nodes"], df_s["time"], yerr=df_s["time_std"],
						 fmt='none', ecolor=style["color"], capsize=1)

	axL.set_xlabel("Number of nodes")
	axL.set_xscale("log")
	axL.set_yscale("log")
	axL.set_ylim(bottom=0)
	axL.set_xticks(sorted(df_size["nodes"].unique()))
	axL.margins(x=0.02)

	# --- Right: scaling with edge probability
	for strategy in strategies:
		if strategy not in df_density["strategy"].unique():
			continue
		style = STYLE["balance_mouflon"] if strategy == "step2" else STYLE["prop_mouflon"]
		df_s = df_density[df_density["strategy"] == strategy].sort_values("density")
		axR.plot(df_s["density"], df_s["time"], **style)
		if "time_std" in df_s.columns:
			axR.errorbar(df_s["density"], df_s["time"], yerr=df_s["time_std"],
						 fmt='none', ecolor=style["color"], capsize=1)

	axR.set_xlabel("Edge probability")
	axR.set_yscale("log")
	axR.set_ylim(bottom=0)
	axR.set_xticks(sorted(df_density["density"].unique()))
	axR.margins(x=0.02)

	# Shared y-axis label
	fig.supylabel("Execution time (s)")

	# Legend for just MOUFLON
	handles = [
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
	]
	fig.legend(handles=handles, loc="upper center", ncol=2)

	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)




##############################################
# Figure 3: MOUFLON over alpha
##############################################

def plot_mouflon_alpha(net_node, net_full, filename="figure3"):
	df1 = pd.read_csv(f"{log_path}/fexp_correction/quality/node_c/{net_node}.csv", header=0)
	df2 = pd.read_csv(f"{log_path}/fexp_correction/quality/full_c/{net_full}.csv", header=0)

	sns.set_style("whitegrid")
	df1 = df1[df1["strategy"] == "hybrid"]
	df2 = df2[df2["strategy"] == "hybrid"]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
	inset_ymin, inset_ymax = get_ncomms_limits([df1, df2], strategies=["step2", "hybrid"])

	# Only modularity + prop_balance (no balance line)
	metric_styles = {
		"modularity": {**STYLE["modularity"]},
		"fair_exp":   {**STYLE["prop_mouflon"]},  # label remains "MOUFLON (prop_balance)"
	}

	def plot_panel(ax, df):
		alpha_vals = sorted(df["alpha"].unique())
		for metric, style in metric_styles.items():
			ax.plot(df["alpha"], df[metric], **style)
			ax.errorbar(df["alpha"], df[metric],
						yerr=df[f"{metric}_std"], fmt='none',
						ecolor=style["color"], capsize=1)
		ax.set_xticks(alpha_vals)
		ax.set_ylim(0, 1)
		ax.margins(x=0.02)

		inset_ax = inset_axes(ax, width="40%", height="35%", loc='lower center')
		inset_ax.plot(df["alpha"], df["ncomms"], **STYLE["ncomms"])
		inset_ax.errorbar(df["alpha"], df["ncomms"],
						  yerr=df["ncomms_std"], fmt='none',
						  ecolor=STYLE["ncomms"]["color"], capsize=1)
		inset_ax.set_ylim(inset_ymin, inset_ymax)
		inset_ax.set_xticklabels([])
		inset_ax.tick_params(axis='y', labelsize=8)
		inset_ax.margins(x=0.02, y=0.05)

	plot_panel(ax1, df1)
	plot_panel(ax2, df2)

	for i, ax in enumerate([ax1, ax2]):
		ax.text(0.01, 0.95, ["A","B"][i],
				transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

	fig.supxlabel("alpha")
	fig.supylabel("Score")

	# Legend without balance line
	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	fig.legend(handles=handles, loc="upper center", ncol=3)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)


##############################################
# Figure 4: MOUFLON over p_sensitive
##############################################

def extract_p_sensitive(filename):
	match = re.search(r'_c(\d+)\.csv$', filename)
	return int(match.group(1)) / 10.0 if match else None

def load_and_prepare(files):
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

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
	inset_ymin, inset_ymax = get_ncomms_limits([df1, df2], strategies=["step2", "hybrid"])

	# Labels changed to simple 'prop_balance' and 'balance'
	# Make balance line less opaque
	metric_styles = {
		"modularity": {**STYLE["modularity"]},                      # label: "Modularity"
		"fair_exp":   {**STYLE["prop_mouflon"], "label": "prop_balance"},
		"fair_bal":   {**STYLE["balance_mouflon"], "label": "balance", "alpha": 0.5},
	}

	def plot_panel(ax, df):
		for metric, style in metric_styles.items():
			s = style.copy()
			ax.plot(df["p_sensitive"], df[metric], **s)
			ax.errorbar(df["p_sensitive"], df[metric],
						yerr=df[f"{metric}_std"], fmt='none',
						ecolor=s.get("color", "black"), capsize=1, alpha=s.get("alpha", 1.0))
		ax.set_xticks(sorted(df["p_sensitive"].unique()))
		ax.set_ylim(0, 1)
		ax.margins(x=0.02)

		inset_ax = inset_axes(ax, width="40%", height="35%", loc='lower right')
		inset_ax.plot(df["p_sensitive"], df["ncomms"], **STYLE["ncomms"])
		inset_ax.errorbar(df["p_sensitive"], df["ncomms"],
						  yerr=df["ncomms_std"], fmt='none',
						  ecolor=STYLE["ncomms"]["color"], capsize=1)
		inset_ax.set_ylim(inset_ymin, inset_ymax)
		inset_ax.set_xticklabels([])
		inset_ax.tick_params(axis='y', labelsize=8)
		inset_ax.margins(x=0.02, y=0.05)

	plot_panel(ax1, df1)
	plot_panel(ax2, df2)

	for i, ax in enumerate([ax1, ax2]):
		ax.text(0.01, 0.95, ["A","B"][i],
				transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

	fig.supxlabel("p_sensitive")
	fig.supylabel("Score")

	# Legend with simplified labels
	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], color=STYLE["prop_mouflon"]["color"], linestyle=STYLE["prop_mouflon"]["linestyle"],
					  marker=STYLE["prop_mouflon"]["marker"], label="prop_balance"),
		mlines.Line2D([], [], color=STYLE["balance_mouflon"]["color"], linestyle=STYLE["balance_mouflon"]["linestyle"],
					  marker=STYLE["balance_mouflon"]["marker"], label="balance", alpha=0.5),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	fig.legend(handles=handles, loc="upper center", ncol=4)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)


##############################################
# Figure 5: Strategies over alpha (MOUFLON only)
##############################################

def plot_strategies_alpha(net_node, filename="figure5"):
	df = pd.read_csv(f"{log_path}/fexp_correction/quality/node_c/{net_node}.csv", header=0)
	sns.set_style("whitegrid")
	fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
	axs = axs.flatten()

	inset_ymin, inset_ymax = get_ncomms_limits([df], strategies=["step2", "hybrid"])

	for idx, (strategy, config) in enumerate(strategy_config.items()):
		ax = axs[idx]
		df_strat = df[df["strategy"] == strategy]
		fair_col = config["fairness"]
		style = config["style"]

		# --- Modularity
		ax.plot(df_strat["alpha"], df_strat["modularity"], **STYLE["modularity"])
		ax.errorbar(df_strat["alpha"], df_strat["modularity"],
					yerr=df_strat["modularity_std"], fmt='none',
					ecolor=STYLE["modularity"]["color"], capsize=1)

		# --- Fairness
		ax.plot(df_strat["alpha"], df_strat[fair_col], **style)
		ax.errorbar(df_strat["alpha"], df_strat[fair_col],
					yerr=df_strat[fair_col + "_std"], fmt='none',
					ecolor=style["color"], capsize=1)

		ax.set_xticks(sorted(df_strat["alpha"].unique()))
		ax.set_ylim(0, 1)
		ax.margins(x=0.02)

		# --- Inset (only step2/hybrid range)
		inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
		inset.plot(df_strat["alpha"], df_strat["ncomms"], **STYLE["ncomms"])
		inset.errorbar(df_strat["alpha"], df_strat["ncomms"],
					   yerr=df_strat["ncomms_std"], fmt='none',
					   ecolor=STYLE["ncomms"]["color"], capsize=1)
		inset.set_ylim(inset_ymin, inset_ymax)
		inset.set_xticklabels([])
		inset.tick_params(axis='y', labelsize=8)
		inset.margins(x=0.02, y=0.05)

		subfig_labels = ["A", "B"]
		ax.text(0.01, 0.95, subfig_labels[idx],
				transform=ax.transAxes, fontsize=14, fontweight="bold",
				va="top", ha="left")

	fig.supxlabel("alpha")
	fig.supylabel("Score")

	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	fig.legend(handles=handles, loc="upper center", ncol=4)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)

##############################################
# Figure 6: Strategies over p_sensitive (MOUFLON only)
##############################################

def load_psensitive_dfs(files):
	dfs = []
	for f in files:
		df = pd.read_csv(f)
		p = extract_p_sensitive(f)
		df = df[df["alpha"] == 0.5].copy()
		df["p_sensitive"] = p
		dfs.append(df)
	return pd.concat(dfs, ignore_index=True).sort_values("p_sensitive")

def plot_strategies_psensitive(files, filename="figure6"):
	df = load_psensitive_dfs(files)
	sns.set_style("whitegrid")
	fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
	axs = axs.flatten()

	inset_ymin, inset_ymax = get_ncomms_limits([df], strategies=["step2", "hybrid"])

	for idx, (strategy, config) in enumerate(strategy_config.items()):
		ax = axs[idx]
		df_strat = df[df["strategy"] == strategy]
		fair_col = config["fairness"]
		style = config["style"]

		# --- Modularity
		ax.plot(df_strat["p_sensitive"], df_strat["modularity"], **STYLE["modularity"])
		ax.errorbar(df_strat["p_sensitive"], df_strat["modularity"],
					yerr=df_strat["modularity_std"], fmt='none',
					ecolor=STYLE["modularity"]["color"], capsize=1)

		# --- Fairness
		ax.plot(df_strat["p_sensitive"], df_strat[fair_col], **style)
		ax.errorbar(df_strat["p_sensitive"], df_strat[fair_col],
					yerr=df_strat[fair_col + "_std"], fmt='none',
					ecolor=style["color"], capsize=1)

		ax.set_xticks(sorted(df_strat["p_sensitive"].unique()))
		ax.set_ylim(0, 1)
		ax.margins(x=0.02)

		# --- Inset (only step2/hybrid range)
		inset = inset_axes(ax, width="40%", height="35%", loc="lower right")
		inset.plot(df_strat["p_sensitive"], df_strat["ncomms"], **STYLE["ncomms"])
		inset.errorbar(df_strat["p_sensitive"], df_strat["ncomms"],
					   yerr=df_strat["ncomms_std"], fmt='none',
					   ecolor=STYLE["ncomms"]["color"], capsize=1)
		inset.set_ylim(inset_ymin, inset_ymax)
		inset.set_xticklabels([])
		inset.tick_params(axis='y', labelsize=8)
		inset.margins(x=0.02, y=0.05)

		subfig_labels = ["A", "B"]
		ax.text(0.01, 0.95, subfig_labels[idx],
				transform=ax.transAxes, fontsize=14, fontweight="bold",
				va="top", ha="left")

	fig.supxlabel("p_sensitive")
	fig.supylabel("Score")

	handles = [
		mlines.Line2D([], [], **STYLE["modularity"]),
		mlines.Line2D([], [], **STYLE["balance_mouflon"]),
		mlines.Line2D([], [], **STYLE["prop_mouflon"]),
		mlines.Line2D([], [], **STYLE["ncomms"]),
	]
	fig.legend(handles=handles, loc="upper center", ncol=4)
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.savefig(f"../plots/{filename}.png", dpi=300)
	plt.close(fig)

##############################################
# Helper: Stats from network/logs
##############################################

def get_stats(network, log_file, only_log=True):
	net = None
	if not only_log:
		with open(f"{obj_path}/{network}.nx", "rb") as g_open:
			net = pickle.load(g_open)

		print(f"Network object {network} loaded.")
		print(f"{network}: N={net.number_of_nodes()}, M={net.number_of_edges()}")
		del net
		gc.collect()

	df = pd.read_csv(f"{log_path}/fexp_correction/{log_file}.csv", header=0)
	avg_time = df.groupby("strategy").agg({"time": ["mean"], "time_std": ["mean"]})
	print(avg_time)

	no_comms = df[df["alpha"].isin([0.0, 0.5, 1.0])][
		["strategy", "alpha", "ncomms", "ncomms_std"]
	]
	print(no_comms)

##############################################
# Main
##############################################

def main():

	realSN_list=["facebook","deezer","twitch","pokec-a","pokec-g"]

	# Get performance stats
	for network in realSN_list:
		get_stats(network,network,only_log=True)

	# Scalability (Figure 3)
	main_df = pd.DataFrame()
	for nodes in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
		df = create_time_df(f"ER_{nodes}_r0001_K2_c05", nodes, 0.001, path_1e03x10)
		main_df = pd.concat([main_df, df], ignore_index=True)

	main_df1 = pd.DataFrame()
	for dp in range(1, 6):
		df1 = create_time_df(f"ER_10000_r0{dp}_K2_c05",
							 10000, float(dp/10), path_10000x10)
		# keep "density" column as is
		main_df1 = pd.concat([main_df1, df1], ignore_index=True)

	plot_scalability_1x2(main_df, main_df1, filename="Figure3")

	# Files for p_sensitive experiments (used in Figures 4 & 6)
	node_files = [
		f"{log_path}/fexp_correction/quality/node_c/color-node_1000_r01_K2_c{p_sens}.csv"
		for p_sens in ["01", "02", "03", "04", "05"]
	]
	full_files = [
		f"{log_path}/fexp_correction/quality/full_c/color-full_1000_r01_K2_c{p_sens}.csv"
		for p_sens in ["01", "02", "03", "04", "05"]
	]

	# Figure 4
	plot_mouflon_alpha("color-node_1000_r01_K2_c05",
					   "color-full_1000_r01_K2_c05",
					   filename="Figure4")

	# Figure 5
	plot_mouflon_psensitive(node_files, full_files, filename="Figure5")

	# Figure 6
	plot_strategies_alpha("color-node_1000_r01_K2_c05", filename="Figure6")

	# Figure 7
	plot_strategies_psensitive(node_files, filename="Figure7")

	# Figure 9 (appendix B)
	plot_multiple_alpha_real(
		## change list if needed
		#["facebook_final","deezer_final","twitch_8graphs","pokec-a_8graphs_thresh_chng","pokec-g"],
		realSN_list,
		draw_error=True,
		filename="Figure9"
	)
	#plot_multiple_alpha_real(realSN_list,draw_error=True,filename="figureB1")

if __name__ == '__main__':
	main()
