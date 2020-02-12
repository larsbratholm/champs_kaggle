import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import scipy
import sklearn.manifold
import sklearn.decomposition

def get_variance_by_type(data, couplings, type_to_idx):
    n = len(type_to_idx)

    scores = {}

    for type_, idx in type_to_idx.items():
        idx = np.asarray(idx)
        type_data = data[:,idx]
        type_couplings = couplings[idx]
        scores[type_] = np.var(type_data - type_couplings[None], axis=1)
    return scores

def extract_weighted_subset(data, couplings, type_to_idx, subset=None, metric='mae'):
    """
    Extract 20000 points of each type and optionally weight them according to their MAE/RMSE in the ensemble
    """
    np.random.seed(42)
    if subset is None:
        subset = np.arange(data.shape[0])
    else:
        subset = np.asarray(subset, dtype=int)

    if metric == 'rmse':
        metric_fun = lambda x: np.std(x, axis=1)[:,None]
    elif metric == 'mae':
        metric_fun = lambda x: np.mean(abs(x), axis=1)[:,None]
    elif metric is None:
        metric_fun = lambda x: 1
    else:
        print("Unknown metric", metric)
        raise SystemExit


    #weights = np.exp(len(type_to_idx)*scores)
    #difference = data[subset,:]-couplings[None]
    x = np.empty((len(subset), len(type_to_idx) * 20000))
    for i, (type_, idx) in enumerate(type_to_idx.items()):
        # weight differences by the chosen metric to get
        # different types to be on same scale
        unscaled_data = (data[subset]-couplings[None])[:,idx]
        score = metric_fun(unscaled_data)
        scaled_data = unscaled_data / score

        x[:,i*20000:(i+1)*20000] = scaled_data[:,np.random.choice(np.arange(len(idx)), size=20000, replace=False)]
    return x

def plot_correlation(data, couplings, name, type_to_idx, subset=[0,1,2,3,4,5,11], 
        filename='correlation_matrix.pdf', linkage=None):
    """
    Plot correlation between solutions
    """
    subset = np.asarray(subset, dtype=int)
    name = np.asarray(name)

    x = extract_weighted_subset(data, couplings, type_to_idx, subset=subset, metric='rmse')
    corr = np.corrcoef(x)

    if linkage not in (None, 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'):
        print("Unknown linkage", linkage)
        raise SystemExit

    # Unsorted plots
    if linkage is None:
        # Plot heatmap
        sns.heatmap((corr), square=True, linewidths=.25, cbar_kws={"shrink": .5},
                        cmap=sns.diverging_palette(220, 10, as_cmap=True),
                        yticklabels=name[subset], xticklabels=subset+1,
                        center=0.5, vmax=1, vmin=0)
    # Sorted from clustering
    else:
        d = scipy.cluster.hierarchy.distance.pdist(x)#, 'cityblock')
        L = scipy.cluster.hierarchy.linkage(d, method=linkage, optimal_ordering=True)

        sns.clustermap((corr), square=True, linewidths=.25, cbar_kws={"shrink": .5},
                cmap = sns.diverging_palette(220, 10, as_cmap=True),
                yticklabels=name[subset], xticklabels=subset+1,
                center=0.5, vmax=1, vmin=0, row_linkage=L, col_linkage=L)
        #plt.xticks(rotation=-45)

    plt.yticks(rotation=0)

    plt.savefig(filename, bbox_inches = "tight")
    plt.clf()

def visualize_methods(data, couplings, name, type_to_idx, scores, manifold='mds', n_samples=100, 
        subset=[0,1,2,3,4,5,11], scale=True, filename="manifold.pdf"):
    """
    Get reduced dimensional projection
    """
    # Set grid
    #plt.style.use('seaborn-whitegrid')
    # Set fig size
    plt.rcParams["figure.figsize"] = (16,9)
    # Set font size
    plt.rcParams["font.size"] = 30

    if scale:
        metric = 'mae'
    else:
        metric = None

    x = extract_weighted_subset(data, couplings, type_to_idx, subset=np.arange(n_samples), metric=metric)

    # Use manhattan distance when possible
    if manifold in ['mds', 'tsne_mds']:
        dissimilarity = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distance = np.mean(abs(x[i]-x[j]))
                dissimilarity[i,j] = distance
                dissimilarity[j,i] = distance

        if manifold == 'mds':
            m = sklearn.manifold.MDS(n_components=2, n_init=10, random_state=42, dissimilarity='precomputed')
            y = m.fit_transform(dissimilarity)
        else:
            if manifold == 'tsne_mds':
                m = sklearn.manifold.MDS(n_components=50, n_init=10, random_state=42, dissimilarity='precomputed')
                x = m.fit_transform(dissimilarity)
            m = sklearn.manifold.TSNE(n_components=2, random_state=42, verbose=10, perplexity=15,
                    metric='manhattan', init='random', n_iter=40000, learning_rate=50)
            y = m.fit_transform(x)
    elif manifold == 'tsne':
        m = sklearn.manifold.TSNE(n_components=2, random_state=42, verbose=10, perplexity=15,
                metric='manhattan', init='random', n_iter=40000, learning_rate=50)
        y = m.fit_transform(x)
    elif manifold == 'pca':
        m = sklearn.decomposition.PCA(n_components=2)
        y = m.fit_transform(x)
    elif manifold == 'tsne_pca':
        m = sklearn.decomposition.PCA(n_components=50)
        x = m.fit_transform(x)
        m = sklearn.manifold.TSNE(n_components=2, random_state=42, verbose=10, perplexity=15,
                metric='manhattan', init='random', n_iter=40000, learning_rate=50)
        y = m.fit_transform(x)
    elif manifold == 'lsa':
        m = sklearn.decomposition.TruncatedSVD(n_components=2)
        y = m.fit_transform(x)
    else:
        print("Unknown manifold %s" % manifold)
        quit()

    # Get scores for coloring
    score_averages = sum(type_scores[:n_samples] for type_scores in scores.values())/8

    fig, ax = plt.subplots()

    im = ax.scatter(y[:,0], y[:,1], c=score_averages, s=120, cmap="viridis_r")
    # Add colorbar
    fig.colorbar(im, ax=ax)

    # Add the rank in the plot for the subset items
    txt = [str(i+1) if i in subset else '' for i in range(n_samples)]

    for i, string in enumerate(txt):
        ax.annotate(string, (y[i,0], y[i,1]))

    # Remove ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Circle
    #plt.scatter(y[subset,0], y[subset,1], edgecolors='k', facecolors='None', s=320)
    # remove top and right spine
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(f'{script_dir}/data/data.pkl', "rb") as f:
            scores, data, id_to_type, id_to_idx, type_to_idx, rank, name, filenames, couplings = pickle.load(f)
    except FileNotFoundError:
        print("No data pickle found")
        raise SystemExit

    # Correlation plot of top methods
    #plot_correlation(data, couplings, name, type_to_idx, subset=[0,1,2,3,4,5,11],
    #        filename=f"{script_dir}/output/correlation_matrix.pdf")

    # Clustered correlation plot of top 50 methods
    #plot_correlation(data, couplings, name, type_to_idx, subset=np.arange(50),
    #        filename=f"{script_dir}/output/correlation_matrix_clustering.pdf", linkage='complete')

    # Solutions projected down to a 2D manifold
    visualize_methods(data, couplings, name, type_to_idx, scores, scale=True,
            subset=[0,1,2,3,4,5,11], filename=f"{script_dir}/output/manifold.pdf")
