import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat

from math import pi

#class Compound_Database_Analysis():

def mol_table_construction(compound_dataset_name):

    ''' Construct a table of descriptors that are of interest to study from raw data. File format used is CSV.
        Input is a list of known inhibitors of SarsCov-2 proteases taken from COVID Moonshot project.
        (https://covid.postera.ai/covid/submissions/compounds) '''

    #file = 'submissions.csv'
    mols = pd.read_csv(compound_dataset_name) #read input
    num_mols = (len(mols)) #check how many molecules are in file
    print(f'{num_mols} molecules in dataset.')

    print("Constructing table")
    table = pd.DataFrame() #construct dataframe
    for i, mol in mols.iterrows():
        #Chem.SanitizeMol(mol)
        table.loc[i, 'SMILES'] = mol['SMILES']
        table.loc[i, 'molecule'] = Chem.MolFromSmiles(mol['SMILES'])
        table.loc[i, 'MW'] = mol['MW']
        table.loc[i, 'cLogP'] = mol['cLogP']
        table.loc[i, 'TPSA'] = mol['TPSA']
        table.loc[i, 'rotatable bonds'] = mol['Rotatable Bonds']
        table.loc[i, 'HBA'] = mol['HBA']
        table.loc[i, 'HBD'] = mol['HBD']

    print("Finished constructing table")
    table.dropna(inplace = True) #remove mols with incomplete data
    difference = (num_mols - len(table)) #difference in starting number when purged mols removed
    print(f'{difference} molecules removed from dataset due to incomplete data')
    table.to_csv('data.csv') #save table to csv
    print(table.head())

    return table

def principal_component_analysis(table):

    ''' Principal Component Analysis (PCA) of molecular descriptors. Molecules are described by their physico-chemical
        properties. 2D at moment but possibility for 3D. '''

    print("Beginning PCA")
    descriptors = table[['MW','cLogP','TPSA','rotatable bonds','HBA','HBD']].values #chosen molecular descriptors for PCA
    descriptors_std = StandardScaler().fit_transform(descriptors) #standardisation of scales for the descriptors
    pca = PCA()
    descriptors_2d = pca.fit_transform(descriptors_std)
    descriptors_pca = pd.DataFrame(descriptors_2d) #Save PCA values to new dataframe
    descriptors_pca.index = table.index
    descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]

    var = pca.explained_variance_ratio_ #shows PC that explains most of the variance
    var_cum = pca.explained_variance_ratio_.cumsum() #shows the amount of variance explained as we add principal components.
    print(f'Explained Variance Ratio: {var}')
    print(f'Cumulative Summed Explained Variance Ratio: {var_cum}')
    print("Total Summed Explained Variance Ratio", sum(pca.explained_variance_ratio_))

    # create scree plot
    print("Creating Scree plot")
    plt.rcParams['axes.linewidth'] = 1.5
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.title('Scree plot', loc='center', fontsize=20, fontweight='bold')
    plt.plot([i + 1 for i in range(len(var))], var, 'b-', linewidth=2)
    plt.plot([i + 1 for i in range(len(var_cum))], var_cum, 'r-', linewidth=2)
    plt.xticks([i + 1 for i in range(len(var_cum))])
    plt.ylabel('% Variance Explained', fontsize=16, fontweight='bold')
    plt.xlabel('Principal Component (PC)', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.tight_layout()
    plt.tick_params('both', width=2, labelsize=12)
    plt.show()

    # perform normalisation of components.
    # normalisation to plot PCA values between 0-1 scale and include the vectors
    scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
    scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))
    #scale3 = 1.0/(max(descriptors_pca['PC3']) - min(descriptors_pca['PC3']))
    #scale4 = 1.0/(max(descriptors_pca['PC4']) - min(descriptors_pca['PC4']))
    #scale5 = 1.0/(max(descriptors_pca['PC5']) - min(descriptors_pca['PC5']))
    #scale6 = 1.0/(max(descriptors_pca['PC6']) - min(descriptors_pca['PC6']))

    #add normalised values to PCA table
    descriptors_pca['PC1_normalized'] = [i*scale1 for i in descriptors_pca['PC1']]
    descriptors_pca['PC2_normalized'] = [i*scale2 for i in descriptors_pca['PC2']]
    #descriptors_pca['PC3_normalized'] = [i*scale1 for i in descriptors_pca['PC3']]
    #descriptors_pca['PC4_normalized'] = [i*scale2 for i in descriptors_pca['PC4']]
    #descriptors_pca['PC5_normalized'] = [i*scale1 for i in descriptors_pca['PC5']]
    #descriptors_pca['PC6_normalized'] = [i*scale2 for i in descriptors_pca['PC6']]

    print("Creating PCA plot")
    plt.rcParams['axes.linewidth'] = 1.5
    plt.figure(figsize=(6,6))

    ax = sb.scatterplot(x = 'PC1_normalized', y = 'PC2_normalized', data = descriptors_pca, s = 20,
                palette = sb.color_palette("Set2", 3), linewidth = 0.2, alpha = 1)

    plt.xlabel('PC1', fontsize = 20, fontweight = 'bold')
    ax.xaxis.set_label_coords(0.98, 0.45)
    plt.ylabel('PC2', fontsize = 20, fontweight = 'bold')
    ax.yaxis.set_label_coords(0.45, 0.98)

    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    lab = ['MW', 'cLogP','TPSA','rotatable bonds','HBA','HBD']
    l = np.transpose(pca.components_[0:2, :])

    n = l.shape[0]
    for i in range(n):
        plt.arrow(0, 0, l[i,0], l[i,1], color = 'k', alpha = 0.5, linewidth = 1.8, head_width = 0.025)
        plt.text(l[i,0]*1.25, l[i,1]*1.25, lab[i], color = 'k', va = 'center', ha = 'center', fontsize = 16)

    circle = plt.Circle((0,0), 1, color ='gray', fill = False, clip_on = True, linewidth = 1.5, linestyle = '--')
    plt.tick_params ('both', width = 2, labelsize = 18)

    ax.add_artist(circle)
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    #plt.tight_layout()
    plt.show()
    print("PCA complete")

def t_sne_kmeans(table):

    ''' For t-SNE (T-distributed Stochastic Neighbor Embedding) analysis, molecues are compared based on
        their structural features and not physico-chemcial properties. K-means performed on t-SNE data
        to identify clusters of molecules with similar structural features. '''

    print("Beginning t-SNE analysis")
    smi = list(table['molecule'])
    fps = [MACCSkeys.GenMACCSKeys(x) for x in smi] #will use MACCSKeys for this
    tanimoto_sim_mat = GetTanimotoSimMat(fps) #computes a similartity matrix between all the molecules
    n_mole = len(fps)
    similarity_matrix = np.ones([n_mole,n_mole])
    i_lower = np.tril_indices(n = n_mole, m = n_mole, k = -1)
    i_upper = np.triu_indices(n = n_mole, m = n_mole, k = 1)
    similarity_matrix[i_lower] = tanimoto_sim_mat
    similarity_matrix[i_upper] = similarity_matrix.T[i_upper]
    distance_matrix = np.subtract(1,similarity_matrix) #similarity matrix of all vs all molecules in our table

    TSNE_sim = TSNE(n_components = 2, init = 'pca', random_state = 90, angle = 0.3,
                perplexity = 50).fit_transform(distance_matrix) #tune the parameters according to your dataset
    tsne_result = pd.DataFrame(data = TSNE_sim , columns = ["TC1","TC2"]) #new table containing tSNE results
    print("t-SNE analysis results", tsne_result.head(5))

    print("Beginning K-means clustering")
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10] #explore a range of cluster sizes to best clasify our molecules
    for n_clusters in range_n_clusters:

        kmeans = KMeans(n_clusters = n_clusters, random_state =  10)
        cluster_labels = kmeans.fit_predict(tsne_result[['TC1','TC2']])
        silhouette_avg = silhouette_score(tsne_result[['TC1','TC1']], cluster_labels) ###check this

        #print silhouette scores. scores between [-1,1] with 1 being the best, hence the better our data is
        #distributed inside the clusters
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)


    kmeans = KMeans(n_clusters = 2, random_state = 10) ##add automation for n-clusters for best performing cluster
    clusters = kmeans.fit(tsne_result[['TC1','TC2']])
    tsne_result['Cluster'] = pd.Series(clusters.labels_, index = tsne_result.index)
    print("t-SNE results with cluster number for each element", tsne_result.head(5)) #tSNE table now contains the number of clusters for each element

    print("Plotting t-SNE and K-means results")
    plt.rcParams['axes.linewidth'] = 1.5
    fig, ax = plt.subplots(figsize = (6, 6))
    ax = sb.scatterplot(x = 'TC1', y = 'TC2', data = tsne_result, hue = 'Cluster', s = 22, palette = sb.color_palette("Set2", 2),
                        linewidth = 0.2, alpha = 1)

    plt.xlabel('tSNE 1', fontsize = 24, fontweight = 'bold')
    plt.ylabel('tSNE 2', fontsize = 24, fontweight = 'bold')
    plt.tick_params('both', width = 2, labelsize = 18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles = handles[1:], labels = labels[1:])
    plt.legend(loc = 'best', frameon = False, prop = {'size': 16}, ncol = 2)

    #plt.tight_layout()
    plt.show()
    print("t-SNE and K-means clustering analysis complete")

    return tsne_result

def lipinski_rule_of_5(table):

    ''' Radar chart of Lipinski's Rule of Five. Provides information on compounds ability to be
        administered orally. '''

    print("Constructing Lipinski's Rule of Five radar chart")
    data = pd.DataFrame() #create a new table containing the normalized bRo5 values of our compounds
    data['MW'] = [i/500 for i in table['MW']]
    data['cLogP'] = [i/5 for i in table['cLogP']]
    data['HBA'] = [i/10 for i in table['HBA']]
    data['HBD'] = [i/5 for i in table['HBD']]
    data['RotB'] = [i/10 for i in table['rotatable bonds']]
    data['TPSA'] = [i/140 for i in table['TPSA']]

    categories = list(data.columns) #set up the parameters for the angles of the radar plot
    n = len(categories)
    values = data[categories].values[0]
    values = np.append(values,values[:1])
    angles = [n / float(n) * 2 * pi for n in range(n)]
    angles += angles[:1]

    Ro5_up = [1,1,1,1,1,1,1] #upper limit for Lipinski's rule of five
    Ro5_low = [0.5,0.1,0,0.25,0.1,0.5,0.5] #lower limit for Lipinski's rule of five

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([1, 1, 1, 1], projection = 'polar')

    plt.xticks(angles[:-1], categories, color = 'k', size = 20, ha = 'center', va = 'top',fontweight = 'book')
    plt.tick_params(axis = 'y', width = 4, labelsize = 12, grid_alpha = 0.05)

    ax.set_rlabel_position(0)
    ax.plot(angles, Ro5_up, linewidth = 2, linestyle = '-', color = 'red')
    ax.plot(angles, Ro5_low, linewidth = 2, linestyle = '-', color = 'red')

    ax.fill(angles, Ro5_up, 'red', alpha=0.2)
    ax.fill(angles, Ro5_low, 'orangered', alpha=0.2)

    for i in data.index:            ##check this
        values = data[categories].values[i]
        values = np.append(values,values[:1])
        ax.plot(angles, values, linewidth = 0.7, color = 'steelblue', alpha = 0.5)
        ax.fill(angles, values, 'C2', alpha = 0.025)

    ax.grid(axis = 'y', linewidth = 1.5, linestyle = 'dotted', alpha=0.8)
    ax.grid(axis = 'x', linewidth = 2, linestyle = '-', alpha = 1)

    plt.show()

    return data

if __name__ == "__main__":
    table = mol_table_construction(compound_dataset_name='submissions.csv')
    principal_component_analysis(table)
    t_sne_kmeans(table)
    lipinski_rule_of_5(table)
    print("Compound dataset analysis complete")
