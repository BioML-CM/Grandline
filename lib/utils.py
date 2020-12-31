

import pandas as pd
import numpy as np
import scipy
import tensorflow as tf




def prepare_adjacency(ppi_fname, gene_list):
    d = len(gene_list)
    
    # Read ppi network
    G_df = pd.read_csv(ppi_fname)
    
    # Get subnetwork
    H_df = G_df[(G_df['gene1'].isin(gene_list)) & (G_df['gene2'].isin(gene_list))]
    
    # Map row and col ids to gene names
    gene_name_id_dict = dict(zip(gene_list, range(d)))
 
    
    gene_list1 = [gene_name_id_dict[g] for g in H_df['gene1']]
    H_df.loc[:, 'row'] = gene_list1
    gene_list2 = [gene_name_id_dict[g] for g in H_df['gene2']]
    H_df.loc[:, 'col'] = gene_list2
    
    V = np.ones(H_df.shape[0] * 2)
    I = np.concatenate([H_df['row'].values, H_df['col'].values])
    J = np.concatenate([H_df['col'].values, H_df['row'].values])

    A = scipy.sparse.coo_matrix((V, (I, J)))
    
    # Add isolated vertices to A
    M, M = A.shape
    if len(gene_list) > M:
        rows = scipy.sparse.coo_matrix((d-M, M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((d, d-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])    
    
    return A






def remove_fake_nodes(num_nodes, node_list, importance_vals, node_cluster_labels):
    new_node_list = []
    new_importance_vals = []
    new_node_cluster_labels = []
    
    for n, imp, c in zip(node_list, importance_vals, node_cluster_labels):
        if n < num_nodes:
            new_node_list += [n]
            new_importance_vals += [imp]
            new_node_cluster_labels += [c]
            
    return new_node_list, new_importance_vals, new_node_cluster_labels



def get_node_importance_df(perms, node_importance, d):
    '''
    d: number of nodes
    '''

    # -2 because GradCAM calculate node_importance before the last pool
    node_list = perms[0]
    importance_vals = np.repeat(np.array(node_importance), 2**(len(perms)-2))
    node_cluster_labels = np.repeat(np.array(perms[-2]), 2**(len(perms)-2))
    
    node_list, importance_vals, node_cluster_labels = remove_fake_nodes(d, node_list, importance_vals, node_cluster_labels) 
    
    important_df = pd.DataFrame(np.array([importance_vals, node_list, node_cluster_labels]).T, columns=['important', 'node', 'cluster'])
    
    return important_df.sort_values('node')


