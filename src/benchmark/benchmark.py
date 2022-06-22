from __future__ import division
from cluster import ClusterGenerator
from functions import matrice_cluster
import numpy as np
import click
import __init__ as init
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
    default=1,
    show_default=True,
    help="initialize the random number generator",
)
@click.option(
    "-ns",
    "--n_samples",
    "n_samples",
    type=int,
    # callback=validate_dir_path,
    default=800,
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
    help="label of clusters",
)
@click.option(
    "-ctr",
    "--centroids",
    "centroids",
    type=dict,
    multiple=True,
    # callback=validate_dir_path,
    help="centroids coordinates set by the user",
)
@click.option(
    "--ps",
    "-par_shapes",
    "par_shapes",
    type=dict,
    multiple=True,
    help="principale shape of cluster",
)
@click.option(
    "-wc",
    "--weight_cluster",
    "weight_cluster",
    type=list,
    # callback=validate_dir_path,
    show_default=True,
    help="percent of samples in each cluster",
)
@click.option(
    "-d",
    "--distributions",
    "distributions",
    type=list,
    # callback=validate_dir_path,
    help="destributions of each cluster",
)
@click.option(
    "-sc",
    "--scale",
    "scale",
    type=list,
    # callback=validate_dir_path,
    show_default=True,
    help="if scale or not ",
)
@click.option(
    "-r",
    "--rotate",
    "rotate",
    type=list,
    # callback=validate_dir_path,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-shp",
    "--shapes",
    "shapes",
    type=list,
    # callback=validate_dir_path,
    show_default=True,
    help="number of total samples to generate",
)
@click.option(
    "-ch",
    "--chevauchement",
    "chevauchement",
    type=list,
    # callback=validate_dir_path,
    show_default=True,
    help="list of tuple= (label of 1st cluster, label of 2nd cluster, percentage of overlap of the 2nd with the 1st)",
)
@click.option(
    "-prs",
    "--parametres_shapes",
    "parametres_shapes",
    type=dict,
    # callback=validate_dir_path,
    show_default=True,
    help="list of tuple of size n cluster ; typle=(main form,a)",
)
@click.option(
    "-prd",
    "--parametres_distributions",
    "parametres_distributions",
    type=dict,
    # callback=validate_dir_path,
    show_default=True,
    help="parameters of distributions",
)
@click.option(
    "--dry-run",
    "dry_run",
    type=bool,
    is_flag=True,
    default=False,
    help="passthrough, will not write anything",
)
def data_generator(
    seed,
    n_samples,
    n_feats,
    n_clusters,
    clusters_label,
    centroids,
    par_shapes,
    weight_cluster,
    distributions,
    scale,
    rotate,
    shapes,
    chevauchement,
    parametres_distributions,
    parametres_shapes,
    dry_run,
):
    # seed=1
    n_samples = 5000
    n_feats = 2
    n_clusters = 4
    clusters_label = np.arange(4)
    centroids = {0: [0.5, 0.5], 1: [0.5, 0.4], 3: (0, 0)}
    par_shapes = {
        0: ("hyper Sphere", 0.9),
        1: ("hyper Sphere", 0.4),
        2: ("hyper Sphere", 0.2),
    }
    weight_cluster = [1 / 4 for _ in range(4)]
    distributions = "gaussian"
    scale = False
    rotate = False
    shapes = [["hyper Sphere", "hyper Sphere"], ["hyper Sphere"]]
    chevauchement = None
    parametres_distributions = {0: (0.3, 0.6), 1: (0.3, 0.6), 3: (-0.2, 0.2)}
    parametres_shapes = {
        0: [[[0.5, 0.5], 0.5, "-"], [[0.5, 0.6], 0.3, "-"]],
        1: [[[0.5, 0.1], 0.2, "-"]],
    }
    dry_run = False
    gnr = ClusterGenerator(
        seed=seed,
        n_samples=n_samples,
        n_feats=n_feats,
        k=n_clusters,
        clusters_label=clusters_label,
        centroids=centroids,
        par_shapes=par_shapes,
        weight_cluster=weight_cluster,
        distributions=distributions,
        parametres_distributions=parametres_distributions,
        scale=scale,
        rotate=rotate,
        shapes=shapes,
        chevauchement=chevauchement,
        parametres_shapes=parametres_shapes,
    )

    X, y = gnr.generate_data()
    M = matrice_cluster(X, y)
    # print(M)
    return M


if __name__ == "__main__":
    cli()
    # seed = 1,
    # n_samples = 5000,
    # n_feats = 2,
    # n_clusters = 4,
    # clusters_label=np.arange(4),
    # centroids={0: [0.5, 0.5], 1: [0.5, 0.4], 3: (0, 0)},
    # par_shapes={
    #         0: ("hyper Sphere", 0.9),
    #         1: ("hyper Sphere", 0.4),
    #         2: ("hyper Sphere", 0.2),
    #     },
    # weight_cluster=[1 / 4 for _ in range(4)],
    # distributions=["gaussian"],
    # scale = [False],
    # rotate = [False],
    # shapes=[["hyper Sphere", "hyper Sphere"], ["hyper Sphere"]],
    # chevauchement=None,
    # parametres_distributions={0: (0.3, 0.6), 1: (0.3, 0.6), 3: (-0.2, 0.2)},
    # parametres_shapes={
    #         0: [[[0.5, 0.5], 0.5, "-"], [[0.5, 0.6], 0.3, "-"]],
    #         1: [[[0.5, 0.1], 0.2, "-"]],}
    # ):
    # k=n_clusters
    # gnr = ClusterGenerator(
    #     seed,
    #     n_samples,
    #     n_feats,
    #     k,
    #     clusters_label,
    #     centroids,
    #     par_shapes,
    #     weight_cluster,
    #     distributions,
    #     parametres_distributions,
    #     scale,
    #     rotate,
    #     shapes,
    #     chevauchement,
    #     parametres_shapes
    # )

    # X, y = gnr.generate_data()
    # M = matrice_cluster(X, y)
    # return M
