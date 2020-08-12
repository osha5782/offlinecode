# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:17:02 2020

@author: hughm
"""


import numpy as np
import pandas as pd
import networkx as nx
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_filepaths(filepath, pick_labeled=False):
    """
    Get a list of all file directories from data folder
    """
    files = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(filepath):
        for file in f:
            files.append(os.path.join(r, file))
    # list comprehension to keep only the labeled version of the adj matrices
    files = [s for s in files if "label" in s] if pick_labeled is True \
        else files
    return files


def gen_graph(df):
    """
    Turn adjacency matrix into a networkx object
    """
    # Set the key column as index
    d = df.set_index('Key')
    # Set the diagonal values (self-loops)
    np.fill_diagonal(d.values, 0)
    # Create graph
    g = nx.convert_matrix.from_pandas_adjacency(d)
    return g


def get_rank_order(df_scores, subject_id):
    """
    Put graph matrix in order and rank
    """
    # Add a rank column
    df_scores[subject_id] = df_scores['score'].rank()
    # Sort in order of ranks
    df_scores.sort_values(subject_id, ascending=False, inplace=True)
    # Delete the score column of the df
    df_ranks = df_scores.drop('score', axis=1)
    df_ranks.reset_index(drop=True, inplace=True)
    return df_ranks


def add_labels(df):
    all_parcellations = [
        'L_Cerebellum', 'L_Thalamus', 'L_Caudate', 'L_Putamen', 'L_Pallidum',
        'Brain-Stem', 'L_Accumbens', 'L_VentralDC', 'R_Cerebellum',
        'R_Thalamus', 'R_Caudate', 'R_Putamen', 'R_Pallidum', 'R_Accumbens',
        'R_VentralDC', 'L_4', 'L_3b', 'L_1', 'L_2', 'L_3a', 'L_SFL', 'L_SCEF',
        'L_6ma', 'L_6mp', 'L_6d', 'L_6a', 'L_6v', 'L_6r', 'L_24dd', 'L_24dv',
        'R_4', 'R_3b', 'R_1', 'R_2', 'R_3a', 'R_SFL', 'R_SCEF', 'R_6ma',
        'R_6mp', 'R_6d', 'R_6a', 'R_6v', 'R_6r', 'R_24dd', 'R_24dv', 'L_a24',
        'L_p32', 'L_10r', 'L_s32', 'L_RSC', 'L_v23ab', 'L_d23ab', 'L_31pv',
        'L_31pd', 'L_31a', 'L_TPOJ3', 'L_IP1', 'L_PGi', 'L_PGs', 'R_a24',
        'R_p32', 'R_10r', 'R_s32', 'R_RSC', 'R_v23ab', 'R_d23ab', 'R_31pv',
        'R_31pd', 'R_31a', 'R_TPOJ3', 'R_IP1', 'R_PGi', 'R_PGs', 'L_8Av',
        'L_8Ad', 'L_a47r', 'L_IFSp', 'L_IFSa', 'L_p9-46v', 'L_46', 'L_p47r',
        'L_PSL', 'L_PFcm', 'L_PFt', 'L_AIP', 'L_PF', 'L_PFm', 'R_8Av', 'R_8Ad',
        'R_a47r', 'R_IFSp', 'R_IFSa', 'R_p9-46v', 'R_46', 'R_p47r', 'R_PSL',
        'R_PFcm', 'R_PFt', 'R_AIP', 'R_PF', 'R_PFm', 'L_a24pr', 'L_p32pr',
        'L_MI', 'L_AVI', 'L_FOP5', 'R_a24pr', 'R_p32pr', 'R_MI', 'R_AVI',
        'R_FOP5', 'L_EC', 'L_PreS', 'L_PeEc', 'L_PHA1', 'L_PHA3', 'L_PHA2',
        'R_EC', 'R_PreS', 'R_PeEc', 'R_PHA1', 'R_PHA3', 'R_PHA2',
        'L_Hippocampus', 'L_Amygdala', 'R_Hippocampus', 'R_Amygdala', 'L_55b',
        'L_8C', 'L_44', 'L_45', 'L_47l', 'L_IFJa', 'L_PBelt', 'L_STSdp',
        'L_STSvp', 'L_TE1p', 'L_PHT', 'L_STSda', 'L_TE1a', 'L_TGv', 'L_STSva',
        'L_V1', 'L_V2', 'L_V3', 'L_V4', 'L_V6', 'L_V3A', 'L_V7', 'L_IPS1',
        'L_V3B', 'L_V6A', 'L_V8', 'L_FFC', 'L_PIT', 'L_VMV1', 'L_VMV3',
        'L_VMV2', 'L_VVC', 'L_MST', 'L_LO1', 'L_LO2', 'L_MT', 'L_PH', 'L_V4t',
        'L_FST', 'L_V3CD', 'L_LO3', 'R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V6',
        'R_V3A', 'R_V7', 'R_IPS1', 'R_V3B', 'R_V6A', 'R_V8', 'R_FFC', 'R_PIT',
        'R_VMV1', 'R_VMV3', 'R_VMV2', 'R_VVC', 'R_MST', 'R_LO1', 'R_LO2',
        'R_MT', 'R_PH', 'R_V4t', 'R_FST', 'R_V3CD', 'R_LO3', 'L_A1', 'L_RI',
        'L_A5', 'L_TPOJ1', 'L_MBelt', 'L_LBelt', 'L_A4', 'L_FOP4', 'R_A1',
        'R_RI', 'R_PBelt', 'R_A5', 'R_STSdp', 'R_TPOJ1', 'R_MBelt', 'R_LBelt',
        'R_A4', 'R_8C', 'R_44', 'R_FOP4', 'L_FEF', 'L_7PC', 'L_LIPv', 'L_VIP',
        'L_LIPd', 'R_FEF', 'R_7PC', 'R_LIPv', 'R_VIP', 'R_LIPd', 'R_PCV',
        'R_7Pm', 'R_7Am', 'R_MIP', 'R_TPOJ2', 'L_7AL', 'L_MIP', 'L_52',
        'L_PFop', 'L_10d', 'L_a10p', 'L_10pp', 'L_p10p', 'R_10d', 'R_a10p',
        'R_10pp', 'R_p10p', 'L_47m', 'L_11l', 'L_13l', 'L_OFC', 'L_47s',
        'L_pOFC', 'R_47m', 'R_11l', 'R_13l', 'R_OFC', 'R_47s', 'R_pOFC',
        'L_p24pr', 'L_33pr', 'L_d32', 'L_8BM', 'L_9m', 'L_8BL', 'L_10v',
        'L_25', 'L_a32pr', 'L_p24', 'R_p24pr', 'R_33pr', 'R_d32', 'R_8BM',
        'R_9m', 'R_8BL', 'R_10v', 'R_25', 'R_a32pr', 'R_p24', 'L_PEF', 'L_9p',
        'L_IFJp', 'L_a9-46v', 'L_9-46d', 'L_9a', 'L_i6-8', 'L_s6-8', 'R_PEF',
        'R_9p', 'R_47l', 'R_IFJa', 'R_IFJp', 'R_a9-46v', 'R_9-46d', 'R_9a',
        'R_i6-8', 'R_s6-8', 'L_PoI2', 'L_Pir', 'L_AAIC', 'L_PoI1', 'L_Ig',
        'R_PoI2', 'R_Pir', 'R_AAIC', 'R_PoI1', 'R_Ig', 'L_43', 'L_OP4',
        'L_OP1', 'L_OP2-3', 'L_FOP1', 'L_FOP3', 'L_FOP2', 'R_43', 'R_OP4',
        'R_OP1', 'R_OP2-3', 'R_FOP1', 'R_FOP3', 'R_FOP2', 'R_PFop', 'L_STV',
        'R_STV', 'R_52', 'R_PI', 'L_PI', 'L_TA2', 'L_STGa', 'L_TGd', 'L_TE2a',
        'L_TF', 'L_TE2p', 'L_TE1m', 'R_TA2', 'R_STGa', 'R_STSda', 'R_STSvp',
        'R_TGd', 'R_TE1a', 'R_TE1p', 'R_TE2a', 'R_TF', 'R_TE2p', 'R_PHT',
        'R_TGv', 'R_STSva', 'R_TE1m', 'L_7Pm', 'L_5m', 'L_5mv', 'L_5L',
        'L_7Am', 'L_7PL', 'R_5m', 'R_5mv', 'R_23c', 'R_5L', 'R_7AL', 'R_7PL',
        'L_POS2', 'L_PCV', 'L_7m', 'L_POS1', 'L_23d', 'L_23c', 'L_ProS',
        'L_DVT', 'R_POS2', 'R_7m', 'R_POS1', 'R_23d', 'R_ProS', 'R_DVT',
        'L_IP2', 'L_IP0', 'R_IP2', 'R_IP0', 'L_TPOJ2', 'L_PGp', 'R_PGp', 'L_H',
        'R_55b', 'R_45', 'R_H'
        ]
    # Replacing id with labels steps
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df["Key"] = all_parcellations
    df.set_index("Key", inplace=True)
    df.columns = all_parcellations
    df.reset_index(inplace=True)
    return df


"""
Calculate some graph metrics
"""


def degree_metric(g, sub_id):
    """
    Calculate degree centrality
    Convert dictionary of DC scores into df and set index
    """
    # Calculate betweenness centrality
    dc = nx.degree_centrality(g)
    # Convert dc dictionary to dataframe
    dc_df = pd.DataFrame(dc.items(), columns=['Parcellation', sub_id])
    return dc_df


def eigenvector_metric(g, sub_id):
    """
    Calculate eigenvector centrality
    Convert dictionary of EVC scores into df and set index
    """
    # Calculate betweenness centrality
    evc = nx.betweenness_centrality(g)
    # Convert evc dictionary to dataframe
    evc_df = pd.DataFrame(evc.items(), columns=['Parcellation', sub_id])
    return evc_df


def betweenness_metric(g, sub_id):
    """
    Calculate betweenness centrality
    Convert dictionary of BC scores into df and set index
    """
    # Calculate betweenness centrality
    bc = nx.betweenness_centrality(g)
    # Convert bc dictionary to dataframe
    bc_df = pd.DataFrame(bc.items(), columns=['Parcellation', sub_id])
    return bc_df


def pagerank_metric(g, sub_id):
    """
    Calculate pagerank centrality
    Convert dictionary of PRC scores into df and set index
    """
    # Calculate pagerank centrality
    prc = nx.pagerank(g)
    # Convert prc dictionary to dataframe
    prc_df = pd.DataFrame(prc.items(), columns=['Parcellation', sub_id])
    return prc_df


def globaleff_metric(g):
    """
    Calculate global efficiency
    """
    # Calculate global efficiency
    ge = nx.global_efficiency(g)
    return ge


def calc_centrality(df, centrality, sub_id):
    """
    Wrapper script for calculating metrics
    """
    # Turn the adjacency matrix (df) into a graph object
    g = gen_graph(df)
    # calculate the score for the specified measure of centrality
    if centrality == "degree":
        # calculate degree centrality for each node of the network
        centrality_scores = degree_metric(g, sub_id)
    elif centrality == "eigenvector":
        # calculate eigenvector centrality for each node of the network
        centrality_scores = eigenvector_metric(g, sub_id)
    elif centrality == "pagerank":
        # calculate PageRank centrality for each node of the network
        centrality_scores = pagerank_metric(g, sub_id)
    elif centrality == "betweenness":
        # calculate betweenness centrality for each node of the network
        centrality_scores = betweenness_metric(g, sub_id)
    else:
        # if no valid centrality measure then quit the function
        raise ValueError(
            'Input valid measure of centrality; \
                (degree, eigenvector, pagerank or betweenness)')
    return centrality_scores


def get_graph_metrics(filepath, metric, ranks=False, needs_labels=False):
    """
    Wrapper script to get global efficiency or centrality metrics
    as combined raw scores or ranked values.
    """
    # If only one filepath is passed, convert this str to a list
    filepath = [filepath] if isinstance(filepath, str) is True else filepath
    if metric == 'ge':
        ge_scores = []
        for f in tqdm(filepath):
            df = pd.read_csv(f)
            if needs_labels is not False:
                df = add_labels(df)
            g = gen_graph(df)
            ge_scores.append(globaleff_metric(g))
        return ge_scores
    elif metric != 'ge' and ranks is False:
        # Calculate centrality measure for each df and combine as list of df's
        scores = []
        for f in tqdm(filepath):
            sub_id = f.split('\\')[-2]
            df = pd.read_csv(f)
            if needs_labels is not False:
                df = add_labels(df)
            centrality_score = calc_centrality(df, metric, sub_id)
            scores.append(centrality_score)
        # Combine all the centrality scores into one df
        all_scores = scores[0]
        for i in range(1, len(scores)):
            all_scores = all_scores.merge(
                scores[i], on='Parcellation', how='left'
                )
        all_scores.set_index('Parcellation', inplace=True)
        # Sort the df by the average order of scores
        all_scores = all_scores.reindex(all_scores.mean(axis=1)
                                        .sort_values(ascending=False).index)
        return all_scores
    elif metric != 'ge' and ranks is True:
        ranks = []
        for f in tqdm(filepath):
            sub_id = f.split('\\')[-2]
            df = pd.read_csv(f)
            if needs_labels is not False:
                df = add_labels(df)
            centrality_score = calc_centrality(df, metric, 'score')
            ranked_df = get_rank_order(centrality_score, sub_id)
            ranks.append(ranked_df)
        # Combine all the centrality scores into one df
        all_ranks = ranks[0]
        for i in range(1, len(ranks)):
            all_ranks = all_ranks.merge(
                ranks[i], on='Parcellation', how='left'
                )
        all_ranks.set_index('Parcellation', inplace=True)
        # Sort the df by the average order of ranks
        all_ranks = all_ranks.reindex(all_ranks.mean(axis=1)
                                      .sort_values(ascending=False).index)
        return all_ranks


"""
Pilot of Graph Framework using real patient data
Read in all the graph data, calculate global efficiency, aggregated PageRank
scores and aggreagated ranks
"""

# Get all the filepaths of structural csv's
fpaths_hc_open = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni\\"
                               "data\\openneuro_normals_clean",
                               pick_labeled=True
                               )
fpaths_hc_schiz = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni\\"
                                "data\\schiz_connectivity_clean",
                                pick_labeled=True
                                )
fpaths_ad_taiwan = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                                 "\\data\\taiwan_graphdata_no_mask_diag\\ad",
                                 pick_labeled=True
                                 )
fpaths_mci_taiwan = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni\\"
                                  "data\\taiwan_graphdata_no_mask_diag\\mci",
                                  pick_labeled=True
                                  )
fpaths_ad_shenzhen = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                                   "\\data\\shenzhen_structural_csv\\AD")
fpaths_mci_shenzhen = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                                    "\\data\\shenzhen_structural_csv"
                                    "\\MCI_clean"
                                    )
fpaths_hc_adni = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                               "\\data\\ADNI_structural_csv\\CN"
                               )
fpaths_ad_adni = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                               "\\data\\ADNI_structural_csv\\AD"
                               )
fpaths_emci_adni = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                                 "\\data\\ADNI_structural_csv\\EMCI"
                                 )
fpaths_lmci_adni = get_filepaths("C:\\Users\\hughm\\Desktop\\Spyder\\omni"
                                 "\\data\\ADNI_structural_csv\\LMCI"
                                 )


# Get all global efficiency scores
ge_hc_open = get_graph_metrics(
    fpaths_hc_open, metric="ge"
    )
ge_hc_schiz = get_graph_metrics(
    fpaths_hc_schiz, metric="ge")
ge_ad_taiwan = get_graph_metrics(
    fpaths_ad_taiwan, metric="ge"
    )
ge_mci_taiwan = get_graph_metrics(
    fpaths_mci_taiwan, metric="ge"
    )
ge_ad_shenzhen = get_graph_metrics(
    fpaths_ad_shenzhen, metric="ge"
    )
ge_mci_shenzhen = get_graph_metrics(
    fpaths_mci_shenzhen, metric="ge"
    )
ge_hc_adni = get_graph_metrics(
    fpaths_hc_adni, metric="ge", needs_labels=True
    )
ge_ad_adni = get_graph_metrics(
    fpaths_ad_adni, metric="ge", needs_labels=True
    )
ge_emci_adni = get_graph_metrics(
    fpaths_emci_adni, metric="ge", needs_labels=True
    )
ge_lmci_adni = get_graph_metrics(
    fpaths_lmci_adni, metric="ge", needs_labels=True
    )

# Get all PageRank scores
scores_hc_open = get_graph_metrics(
    fpaths_hc_open, metric="pagerank"
    )
scores_hc_schiz = get_graph_metrics(
    fpaths_hc_schiz, metric="pagerank"
    )
scores_ad_taiwan = get_graph_metrics(
    fpaths_ad_taiwan, metric="pagerank"
    )
scores_mci_taiwan = get_graph_metrics(
    fpaths_mci_taiwan, metric="pagerank"
    )
scores_ad_shenzhen = get_graph_metrics(
    fpaths_ad_shenzhen, metric="pagerank"
    )
scores_mci_shenzhen = get_graph_metrics(
    fpaths_mci_shenzhen, metric="pagerank"
    )
scores_hc_adni = get_graph_metrics(
    fpaths_hc_adni, metric="pagerank", needs_labels=True
    )
scores_ad_adni = get_graph_metrics(
    fpaths_ad_adni, metric="pagerank", needs_labels=True
    )
scores_emci_adni = get_graph_metrics(
    fpaths_emci_adni, metric="pagerank", needs_labels=True
    )
scores_lmci_adni = get_graph_metrics(
    fpaths_lmci_adni, metric="pagerank", needs_labels=True
    )

# Get all PageRank ranks
ranks_hc_open = get_graph_metrics(
    fpaths_hc_open, metric="pagerank", ranks=True
    )
ranks_hc_schiz = get_graph_metrics(
    fpaths_hc_schiz, metric="pagerank", ranks=True
    )
ranks_ad_taiwan = get_graph_metrics(
    fpaths_ad_taiwan, metric="pagerank", ranks=True
    )
ranks_mci_taiwan = get_graph_metrics(
    fpaths_mci_taiwan, metric="pagerank", ranks=True
    )
ranks_ad_shenzhen = get_graph_metrics(
    fpaths_ad_shenzhen, metric="pagerank", ranks=True
    )
ranks_mci_shenzhen = get_graph_metrics(
    fpaths_mci_shenzhen, metric="pagerank", ranks=True
    )
ranks_hc_adni = get_graph_metrics(
    fpaths_hc_adni, metric="pagerank", ranks=True, needs_labels=True
    )
ranks_ad_adni = get_graph_metrics(
    fpaths_ad_adni, metric="pagerank", ranks=True, needs_labels=True
    )
ranks_emci_adni = get_graph_metrics(
    fpaths_emci_adni, metric="pagerank", ranks=True, needs_labels=True
    )
ranks_lmci_adni = get_graph_metrics(
    fpaths_lmci_adni, metric="pagerank", ranks=True, needs_labels=True
    )


"""
Index the groups that are missing subject ID's with new column headers
"""

# Add new column headers (index scans #1 through to #x)
def index_df(unindexed_df, dataset):
    column_headers = []
    for i in range(len(unindexed_df.columns)):
        column_headers.append(dataset.format(str(i+1)))
    unindexed_df.columns = column_headers
    indexed_df = unindexed_df.copy()
    return indexed_df


# The datasets that are missing subject ID's in filepath:
scores_ad_shenzhen = index_df(scores_ad_shenzhen, dataset="SHN AD #{}")
scores_mci_shenzhen = index_df(scores_mci_shenzhen, dataset="SHN MCI #{}")
scores_hc_adni = index_df(scores_hc_adni, dataset="ADNI HC #{}")
scores_ad_adni = index_df(scores_ad_adni, dataset="ADNI AD #{}")
scores_emci_adni = index_df(scores_emci_adni, dataset="ADNI EMCI #{}")
score_lmci_adni = index_df(scores_lmci_adni, dataset="ADNI LMCI #{}")
ranks_ad_shenzhen = index_df(ranks_ad_shenzhen, dataset="SHN AD #{}")
ranks_mci_shenzhen = index_df(ranks_mci_shenzhen, dataset="SHN MCI #{}")
ranks_hc_adni = index_df(ranks_hc_adni, dataset="ADNI HC #{}")
ranks_ad_adni = index_df(ranks_ad_adni, dataset="ADNI AD #{}")
ranks_emci_adni = index_df(ranks_emci_adni, dataset="ADNI EMCI #{}")
ranks_lmci_adni = index_df(ranks_lmci_adni, dataset="ADNI LMCI #{}")


"""
Framework Pilot: Study & distribution plot of Global Efficiency
"""

# set up the matplotlib figure
sns.set_context("paper", font_scale=1.2)
f, ax = plt.subplots(2, 1, figsize=(12.8, 9.6), sharex=True)
f.subplots_adjust(hspace=0)
ax1, ax2 = ax.flatten()

# Plot the first subplot
sns.distplot(
    ge_hc_adni, label="CN Old (ADNI)", color='tab:red', ax=ax1
    )
sns.distplot(
    ge_hc_open, label="CN Young (CNP)", color='tab:brown', ax=ax1
    )
sns.distplot(
    ge_hc_schiz, label="CN Young (SCZ)", color='tab:orange',
    ax=ax1
    )
ax1.legend(loc='upper left')

# Plot the second subplot
sns.distplot(
    ge_ad_taiwan, label="AD (TWN)", color='tab:green', ax=ax2
    )
sns.distplot(
    ge_mci_taiwan, label="MCI (TWN)", color='tab:olive', ax=ax2
    )
sns.distplot(
    ge_ad_adni, label="AD (ADNI)", color='m', ax=ax2
    )
sns.distplot(
    ge_lmci_adni, label="LMCI (ADNI)", color='tab:pink', ax=ax2
    )
sns.distplot(
    ge_emci_adni, label="EMCI (ADNI)", color='tab:purple', ax=ax2
    )
sns.distplot(
    ge_ad_shenzhen, label="AD (SHN)", color='tab:blue', ax=ax2
    )
sns.distplot(
    ge_mci_shenzhen, label="MCI (SHN)", color='tab:cyan', ax=ax2
    )
ax2.legend(loc='upper left')
plt.xlabel("Global Efficiency Score", fontsize=20)
plt.show()


# Single plot for Shenzhen comparison only
sns.set_context("paper", font_scale=1.2)
f, ax = plt.subplots(1, 1)
sns.distplot(
    ge_mci_shenzhen, label="Shenzhen MCI", color='tab:blue', ax=ax
    )
sns.distplot(
    ge_ad_shenzhen, label="Shenzhen AD", color='tab:red', ax=ax
    )
ax.legend(loc='upper left')
plt.xlabel("Global Efficiency Score", fontsize=10)
plt.show()


"""
Pilot Study of PageRank "Ranks" analysis (heatmapping)
"""


def heatmap_rankdrop(base_df, comparison_df, title=None):
    base_df = base_df.reset_index().iloc[:, 0]
    # Re-order the df being heatmapped into the baseline order
    comparison_df = comparison_df.reindex(base_df)
    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 80))
    sns.heatmap(comparison_df, vmin=min(comparison_df.min()),
                vmax=max(comparison_df.max()), cmap='coolwarm',
                cbar_kws={"shrink": 0.99, "pad": 0.03, "aspect": 70})
    # figure labeling & other aesthetics
    ax.set(
           xlabel="Subject ID",
           ylabel="Node of Parcellation Atlas"
           )
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(comparison_df)))
    for i in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        i.set_fontsize(70)
    plt.setp(ax.get_xticklabels(), rotation=45)
    return ax


ax_ad = heatmap_rankdrop(base_df=ranks_hc_schiz,
                         comparison_df=ranks_hc_open)


"""
Pilot Study: 'Anomaly' Detection over PageRank
"""

# Pick the desired threshold (std's required to detect 'anomaly')
thresh = 2


def get_anomalies(base_df, comparison_df, std):
    # Re-order the comparison df to the baseline df order
    baseline_order = pd.DataFrame(base_df.reset_index().iloc[:, 0])
    anomaly_df = comparison_df.reindex(baseline_order.Parcellation)

    for row in tqdm(list(anomaly_df.index)):
        for col in list(anomaly_df.columns):
            anomaly_df.loc[row, col] = ((anomaly_df.loc[row, col]
                                        - base_df.mean(axis=1).loc[row])
                                        / (base_df.std(axis=1).loc[row]))
            if anomaly_df.loc[row, col] > -thresh:
                anomaly_df.loc[row, col] = 0
    anomaly_df = anomaly_df.abs()
    anomaly_df = anomaly_df.loc[(anomaly_df != 0).any(axis=1)]
    anomaly_df['sum'] = anomaly_df.sum(axis=1)
    anomaly_df.sort_values('sum', ascending=False, inplace=True)
    anomaly_df.drop("sum", axis=1, inplace=True)
    col_order = (anomaly_df == 0).sum().sort_values(inplace=False).index
    anomaly_df = anomaly_df[col_order]
    return anomaly_df


# Input base_df and comparison_df to get anomaly detection df
anomaly_df = get_anomalies(
    base_df=scores_hc_schiz, comparison_df=scores_hc_open, std=thresh
    )

# Plotting of 'anomalies'
heatmap_input_df = anomaly_df

# set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))
# plot the heatmap
sns.heatmap(heatmap_input_df, vmin=min(heatmap_input_df.min()),
            vmax=5, cmap="Reds")
# figure labeling / aesthetics
ax.set(
       xlabel="Scan Index Number",
       ylabel="Node of Parcellation Atlas"
       )
ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(heatmap_input_df)))
for i in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    i.set_fontsize(30)
plt.setp(ax.get_xticklabels(), rotation=45)
plt.show()


# Above process repeated for the 3 MCI to AD comparisons
anomaly_hc_schiz = get_anomalies(
    base_df=scores_hc_adni, comparison_df=scores_ad_adni, thresh=2
    )
anomaly_hc_adni = get_anomalies(
    base_df=scores_mci_taiwan, comparison_df=scores_ad_taiwan, thresh=2
    )
anomaly_emci_adni = get_anomalies(
    base_df=scores_mci_shenzhen, comparison_df=scores_ad_shenzhen, thresh=2
    )

f, ax = plt.subplots(1, 3, figsize=(45, 15))
ax1, ax2, ax3 = ax.flatten()

sns.heatmap(anomaly_hc_schiz, vmin=min(anomaly_hc_schiz.min()),
            vmax=5, cmap="Reds", ax=ax1)
ax1.set_title("HC to AD (ADNI)", fontsize=50)

sns.heatmap(anomaly_hc_adni, vmin=min(anomaly_hc_adni.min()),
            vmax=5, cmap="Reds", ax=ax2)
ax2.set_title("MCI to AD (TWN)", fontsize=50)

sns.heatmap(anomaly_emci_adni, vmin=min(anomaly_emci_adni.min()),
            vmax=5, cmap="Reds", ax=ax3)
ax3.set_title("MCI to AD (SHN)", fontsize=50)
plt.show()
