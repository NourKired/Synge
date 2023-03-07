from __future__ import division
from synge.cluster import ClusterGenerator
from synge.functions import matrice_cluster
import numpy as np
from matplotlib import pyplot as plt


# shapes = [["hyper Sphere", "hyper Sphere"], ["hyper Sphere"]]
# n_cluster = 4
# gnr = ClusterGenerator(
#     seed=1,
#     n_samples=5000,
#     n_feats=2,
#     n_cluster=4,
#     clusters_label=np.arange(n_cluster),
#     centroids={0: [0.5, 0.5], 1: [0.5, 0.4], 3: (0, 0)},
#     par_shapes={
#         0: ("hyper Sphere", 0.9),
#         1: ("hyper Sphere", 0.4),
#         2: ("hyper Sphere", 0.2),
#     },
#     weight_cluster=[1 / n_cluster for _ in range(n_cluster)],
#     distributions="gaussian",
#     parametres_distributions={0: (0.3, 0.6), 1: (0.3, 0.6), 3: (-0.2, 0.2)},
#     scale=False,
#     rotate=False,
#     shapes=shapes,
#     chevauchement=None,
#     parametres_shapes={
#         0: [[[0.5, 0.5], 0.5, "-"], [[0.5, 0.6], 0.3, "-"]],
#         1: [[[0.5, 0.1], 0.2, "-"]],
#     },
# )

# X, y = gnr.generate_data()
# M = matrice_cluster(X, y)

# print("ok")
# # cercle


# fig, ax = plt.subplots()

# # # plt.grid(linestyle='--')

# x, y = list(zip(*M[0]))  # cercle

# ax.scatter(x, y)

# x, y = list(zip(*M[2]))  # cercle

# ax.scatter(x, y)


# ax.set_aspect(1)


# # shapes = [['hyper Sphere', 'hyper Sphere'], ['hyper Sphere']]
# k = 4
# gnr = ClusterGenerator(
#     seed=1,
#     n_samples=5000,
#     n_feats=2,
#     n_cluster=4,
#     clusters_label=np.arange(k),
#     centroids={0: [0.5, 0.5], 1: [0.5, 0.5], 2: [0.3, 0.4]},
#     par_shapes={
#         0: ("hyper Sphere", 0.9),
#         1: ("hyper Sphere", 0.4),
#         2: ("hyper Sphere", 0.2),
#     },
#     weight_cluster=[1 / k for _ in range(k)],
#     distributions="gaussian",
#     parametres_distributions={0: (0.3, 0.6), 1: (0.3, 0.6), 3: (-0.2, 0.2)},
#     scale=False,
#     rotate=False,
#     shapes=shapes,
#     chevauchement=None,
#     parametres_shapes={
#         0: [[[0.5, 0.5], 0.5, "-"], [[0.5, 0.6], 0.3, "-"]],
#         1: [[[0.5, 0.1], 0.2, "-"]],
#     },
# )

# X, y = gnr.generate_data()
# M = matrice_cluster(X, y)
# print("ok")

# # cercle

# # # plt.grid(linestyle='--')

# x, y = list(zip(*M[0]))  # cercle

# ax.scatter(x, y)

# x, y = list(zip(*M[2]))  # cercle

# ax.scatter(x, y)

# # x, y = list(zip(*M[1]))# cercle

# # ax.scatter(x,y)
# # x, y = list(zip(*M[3]))# cercle

# # ax.scatter(x,y)


# ax.set_aspect(1)


shapes = [["hyper Sphere", "hyper Sphere"], ["hyper Sphere"]]
k = 4
gnr = ClusterGenerator(
    seed=1,
    n_samples=5000,
    n_feats=4,
    n_cluster=k,
    clusters_label=np.arange(k),
    centroids={0: [0.5, 0.5, 0, 0], 1: [0.5, 0.5, 0, 0], 2: [0.3, 0.4, 0, 0]},
    par_shapes={
        0: ("hyper Sphere", 0.9),
        1: ("hyper Sphere", 0.4),
        2: ("hyper Sphere", 0.2),
    },
    weight_cluster=[1 / k for _ in range(k)],
    distributions="gaussian",
    parametres_distributions={
        0: (0.3, 0.6, 0, 0),
        1: (0.3, 0.6, 0, 0),
        3: (-0.2, 0.2, 0, 0),
    },
    scale=False,
    rotate=False,
    shapes=shapes,
    chevauchement=None,
    parametres_shapes={
        0: [[[0.5, 0.5, 0, 0], 0.5, "-"], [[0.5, 0.6, 0, 0], 0.3, "-"]],
        1: [[[0.5, 0.1, 0, 0], 0.2, "-"]],
    },
)

X, y = gnr.generate_data()
M = matrice_cluster(X, y)


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
