from __future__ import division
from cluster import ClusterGenerator
from functions import matrice_cluster
import numpy as np
import click
from transformers import pipeline
import os
from tqdm import tqdm
import benchmark.__init__ as init
from nltk.tokenize import sent_tokenize
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s ",
    level=logging.INFO,
)

@click.group
def cli():
    """Represents the root cli function"""
    pass


@cli.command
def version():
    """Represents cli 'version' command"""
    click.echo(init.__version__)

@cli.command
@click.option(
    "-s",
    "--seed",
    "seed",
    type=int,
    # callback=validate_file_path,
    required=False,
    default=1
    show_default=True,
    help="initialize the random number generator",
)

@click.option(
    "-ns",
    "--n_samples",
    "n_samples",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-nf",
    "--n_feats",
    "n_feats",
    type=int,
    # callback=validate_dir_path,
    default=3,
    show_default=True,
    help="number of total features",
)

@click.option(
    "-k",
    "--n_clusters",
    "n_clusters",
    type=int,
    # callback=validate_dir_path,
    default=5,
    show_default=True,
    help="number of total clusters",
)
@click.option(
    "-cl",
    "--clusters_label",
    "clusters_label",
    type=list,
    # callback=validate_dir_path,
    help="number of total samples",
)
@click.option(
    "-ctr",
    "--centroids",
    "centroids",
    type=dict(int, list),
    multiple=True,
    # callback=validate_dir_path,
    help="centroids coordinates set by the user",
)
@click.option(
    "--ps",
    "-par_shapes",
    "parameter_shapes",
    type=(click.Choice(['hyper Spher', 'hyper Rectangle']), float),
    multiple=True,
    help="the minimum probability of private data for labels",
)

@click.option(
    "-wc",
    "--weight_cluster",
    "weight_cluster",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-d",
    "--distributions",
    "distributions",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-sc",
    "--scale",
    "scale",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-r",
    "--rotate",
    "rotate",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-shp",
    "--shapes",
    "shapes",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-ch",
    "--chevauchement",
    "chevauchement",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-prs",
    "--parametres_shapes",
    "parametres_shapes",
    type=int,
    # callback=validate_dir_path,
    default=2000,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "--dry-run",
    "dry_run",
    type=bool,
    is_flag=True,
    default=False,
    help="passthrough, will not write anything",
)



def pii_detect(
    input_file: str,
    out_dir: str,
    thresh,
    overwrite: bool = False,
    dry_run: bool = False,
    to_test: bool = False,
):
    """Represents cli 'pii_detect' command"""
    # validate_args(sentence, thresh)
    tresh_dict = dict(thresh)
    thresholds = {
        tag: tresh_dict[tag] if tag in tresh_dict.keys() else threshs[tag]
        for tag in threshs.keys()
    }
    file_name = os.path.basename(input_file).split(".")[0]
    with open(input_file) as f:
        list_sent = [line.rstrip() for line in f]
    text = " ".join(list_sent)
    df = sent_tokenize(text)
    logging.info("loading pipeline")
    pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logging.info("prediction in progress")
    detected_labels: list = [
        predict(pipe, sent, thresholds) for sent in tqdm(df, total=len(df))
    ]
    logging.info("saving results...")
    out(input_file, out_dir, file_name, detected_labels, dry_run, overwrite, to_test)


if __name__ == "__main__":
    cli()



shapes = [["hyper Sphere", "hyper Sphere"], ["hyper Sphere"]]
k = 4
gnr = ClusterGenerator(
    seed=1,
    n_samples=5000,
    n_feats=2,
    k=4,
    clusters_label=np.arange(k),
    centroids={0: [0.5, 0.5], 1: [0.5, 0.4], 3: (0, 0)},
    par_shapes={
        0: ("hyper Sphere", 0.9),
        1: ("hyper Sphere", 0.4),
        2: ("hyper Sphere", 0.2),
    },
    weight_cluster=[1 / k for _ in range(k)],
    distributions="gaussian",
    parametres_distributions={0: (0.3, 0.6), 1: (0.3, 0.6), 3: (-0.2, 0.2)},
    scale=False,
    rotate=False,
    shapes=shapes,
    chevauchement=None,
    parametres_shapes={
        0: [[[0.5, 0.5], 0.5, "-"], [[0.5, 0.6], 0.3, "-"]],
        1: [[[0.5, 0.1], 0.2, "-"]],
    },
)

X, y = gnr.generate_data()
M = matrice_cluster(X, y)

print(M)
# # cercle


# fig, ax = plt.subplots()

# # # plt.grid(linestyle='--')

# x, y = list(zip(*M[0]))  # cercle

# ax.scatter(x, y)

# x, y = list(zip(*M[2]))  # cercle

# ax.scatter(x, y)


# ax.set_aspect(1)


# shapes = [['hyper Sphere', 'hyper Sphere'], ['hyper Sphere']]
# k = 4
# gnr = ClusterGenerator(seed=1, n_samples=5000, n_feats=2, k=4, clusters_label=np.arange(k),
#                        centroids={0: [0.5, 0.5], 1: [0.5, 0.5], 2: [0.3, 0.4]}, par_shapes={0: ('hyper Sphere', 0.9), 1: ('hyper Sphere', 0.4), 2: ('hyper Sphere', 0.2)},
#                        weight_cluster=[1 / k for _ in range(k)], distributions='gaussian',
#                        parametres_distributions={0: (0.3, 0.6), 1: (0.3, 0.6), 3: (-0.2, 0.2)}, scale=False, rotate=False,
#                        shapes=shapes, chevauchement=None,
#                        parametres_shapes={0: [[[0.5, 0.5], 0.5, '-'], [[0.5, 0.6], 0.3, '-']], 1: [[[0.5, 0.1], 0.2, '-']]})

# X,y=gnr.generate_data()
# M=matrice_cluster(X,y)


# # cercle

# # # plt.grid(linestyle='--')

# x, y = list(zip(*M[0]))# cercle

# ax.scatter(x,y)

# x, y = list(zip(*M[2]))# cercle

# ax.scatter(x,y)

# # x, y = list(zip(*M[1]))# cercle

# # ax.scatter(x,y)
# # x, y = list(zip(*M[3]))# cercle

# # ax.scatter(x,y)


# ax.set_aspect(1)


# shapes=[['hyper Sphere','hyper Sphere'],['hyper Sphere']]
# k=4
# gnr=ClusterGenerator(seed=1, n_samples=5000, n_feats=4, k=4,clusters_label=np.arange(k),
#                      centroids={0:[0.5,0.5,0,0],1:[0.5,0.5,0,0],2:[0.3,0.4,0,0]},par_shapes={0:('hyper Sphere',0.9),1:('hyper Sphere',0.4),2:('hyper Sphere',0.2)},
#                       weight_cluster=[1/k for _ in range(k)],distributions='gaussian',
#                      parametres_distributions={0:(0.3,0.6,0,0),1:(0.3,0.6,0,0),3:(-0.2,0.2,0,0)},scale=False, rotate=False,
#                      shapes=shapes,chevauchement=None,
#                      parametres_shapes={0:[[[0.5,0.5,0,0],0.5,'-'],[[0.5,0.6,0,0],0.3,'-']],1:[[[[0.5,0.1,0,0],0.2,'-']]]})

# X,y=gnr.generate_data()
# M=matrice_cluster(X,y)


# fig, ax = plt.subplots()


# nb_feats= int(input('nb of features to plot '))
# nb_clusters= int(input('nb of clusters to plot '))
# listes=combinliste(np.arange(nb_feats),3)
# Y=y ##  à commenter en cas de 2eme exécution
# y=list(set(y.reshape(-1)))[:nb_clusters] ## à excuter qu'une seule fois ( pour une deuxieme exécution commenter cette ligne et la precedente)

# opacity: dict = {i:1 for i in y}


# # sc: StandardScaler = StandardScaler() # parametre à demander à l'utilisateur
# # X: np.ndarray = sc.fit_transform(X)
# # pca: PCA = PCA(n_components=X.shape[1]) # nombre de dimensions -- > à demander à l'utilisateur
# # X: np.ndarray = pca.fit_transform(X)
# j=1
# data: list = []
# name: dict = {i:str('%d.%d' % (j, i)) for i, val in zip(y, y)} # num des clusters


# data_plot: pd.DataFrame = pd.DataFrame(X)
# data_plot["prediction"] = Y
#     # data_plot["prediction"] = prediction.replace([-1,1], name)
# data: dict ={}
# # fig = make_subplots(rows=1, cols=1)
# for (i1,i2,i3) in listes:
#   dataa=[]
#   for i, val in zip(y, y):
#     data_semi_plot: pd.DataFrame = data_plot[data_plot["prediction"] == val]
#     dataa.append(go.Scatter3d(x=data_semi_plot[i1], y=data_semi_plot[i2], z=data_semi_plot[i3], name=name[val], mode='markers',marker=dict(size=6), opacity=opacity[i]))
#   data[(i1,i2,i3)]=dataa


# fig = make_subplots(
#     rows=len(listes),
#     cols=1,
#     specs=[[{"type": "scatter3d"}]]*len(listes),
#     subplot_titles=[str('%d.%d.%d '% (i,j,k)) for (i,j,k )in listes])

# count=0
# for ((i1,i2,i3),d) in data.items():
#   count+=1
#   fig.add_traces(d, rows=[count]*len(d), cols=[1]*len(d))


# fig.update_layout(width=1000,height=4000,showlegend=True)

# fig.show()

# shapes=[['hyper Sphere','hyper Sphere'],['hyper Sphere']]
# k=4
# gnr=ClusterGenerator(seed=1, n_samples=5000, n_feats=4, k=4,clusters_label=np.arange(k),
#                      centroids={0:[0.5,0.5,0,0],1:[0.5,0.5,0,0],2:(0.3,0.4,0,0)},par_shapes={0:('hyper Sphere',0.9),1:('hyper Sphere',0.4),2:('hyper Sphere',0.2)},
#                       weight_cluster=[1/k for _ in range(k)],distributions='gaussian',
#                      parametres_distributions={0:(0.3,0.6,0,0),1:(0.3,0.6,0,0),3:(-0.2,0.2,0,0)},scale=False, rotate=False,
#                      shapes=shapes,chevauchement=None,
#                      parametres_shapes={0:([[0.5,0.5,0,0],0.5,'-'],[[0.5,0.6,0,0],0.3,'-']),1:([[[0.5,0.1,0,0],0.2,'-']])})

# X,y=gnr.generate_data()
# M=matrice_cluster(X,y)


# nb_feats= int(input('nb of features to plot '))
# nb_clusters= int(input('nb of clusters to plot '))
# listes=combinliste(np.arange(nb_feats),3)
# Y=y ##  à commenter en cas de 2eme exécution
# y=list(set(y.reshape(-1)))[:nb_clusters] ## à excuter qu'une seule fois ( pour une deuxieme exécution commenter cette ligne et la precedente)

# opacity: dict = {i:1 for i in y}


# # sc: StandardScaler = StandardScaler() # parametre à demander à l'utilisateur
# # X: np.ndarray = sc.fit_transform(X)
# # pca: PCA = PCA(n_components=X.shape[1]) # nombre de dimensions -- > à demander à l'utilisateur
# # X: np.ndarray = pca.fit_transform(X)
# j=1
# data: list = []
# name: dict = {i:str('%d.%d' % (j, i)) for i, val in zip(y, y)} # num des clusters


# data_plot: pd.DataFrame = pd.DataFrame(X)
# data_plot["prediction"] = Y
#     # data_plot["prediction"] = prediction.replace([-1,1], name)
# data: dict ={}
# # fig = make_subplots(rows=1, cols=1)
# for (i1,i2,i3) in listes:
#   dataa=[]
#   for i, val in zip(y, y):
#     data_semi_plot: pd.DataFrame = data_plot[data_plot["prediction"] == val]
#     dataa.append(go.Scatter3d(x=data_semi_plot[i1], y=data_semi_plot[i2], z=data_semi_plot[i3], name=name[val], mode='markers',marker=dict(size=6), opacity=opacity[i]))
#   data[(i1,i2,i3)]=dataa


# fig = make_subplots(
#     rows=len(listes),
#     cols=1,
#     specs=[[{"type": "scatter3d"}]]*len(listes),
#     subplot_titles=[str('%d.%d.%d '% (i,j,k)) for (i,j,k )in listes])

# count=0
# for ((i1,i2,i3),d) in data.items():
#   count+=1
#   fig.add_traces(d, rows=[count]*len(d), cols=[1]*len(d))


# fig.update_layout(width=1000,height=4000,showlegend=True)

# fig.show()
