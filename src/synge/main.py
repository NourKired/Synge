import click
import synge.__init__ as init
from synge.cluster import ClusterGenerator
from synge.functions import matrice_cluster
from synge.visualization import visualize_


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
    required=False,
    default=None,
    show_default=True,
    help="Initialize the random number generator",
)
@click.option(
    "-n",
    "--n-samples",
    "n_samples",
    type=int,
    default=1000,
    required=False,
    show_default=True,
    help="Total number of samples",
)
@click.option(
    "-nf",
    "--n_feats",
    "n_feats",
    type=int,
    default=3,
    required=False,
    show_default=True,
    help="Total number of features",
)
@click.option(
    "-nc",
    "--n-cluster",
    "n_cluster",
    type=int,
    default=3,
    required=False,
    show_default=True,
    help="Total number of clusters",
)
@click.option(
    "-cl",
    "--clusters-label",
    "clusters_label",
    required=False,
    show_default=True,
    help="labels of clusters",
)
@click.option(
    "-cnt",
    "--centroids",
    "centroids",
    required=False,
    show_default=True,
    help="labels of clusters",
)
@click.option(
    "-pr",
    "--par-shapes",
    "par_shapes",
    multiple=False,
    required=False,
    help="parametres shapes of clusters",
)
@click.option(
    "-wc",
    "--weight-cluster",
    "weight_cluster",
    multiple=False,
    required=False,
    help="main shapes parameters of each cluster",
)
@click.option(
    "-d",
    "--distributions",
    "distributions",
    multiple=False,
    required=False,
    help="distributions of clusters",
)
@click.option(
    "-pd",
    "--parametres_distributions",
    "parametres_distributions",
    multiple=False,
    required=False,
    help="distributions parametres of clusters",
)
@click.option(
    "-ch",
    "--chv",
    "chevauchement",
    multiple=False,
    required=False,
    help="overlap shapes on each cluster",
)
@click.option(
    "-pshp",
    "--prt_shapes",
    "parametres_shapes",
    multiple=False,
    required=False,
    help="paramaters secondary shapes on each cluster",
)
@click.option(
    "-sp",
    "--shapes",
    "shapes",
    multiple=False,
    required=False,
    help="secondary shapes on each cluster",
)
@click.option(
    "-sc",
    "--scale",
    "scale",
    type=bool,
    is_flag=True,
    default=False,
    help="scale dataset",
)
@click.option(
    "-r",
    "--rotate",
    "rotate",
    type=bool,
    is_flag=True,
    default=False,
    help="rotate dataset",
)
@click.option(
    "-vi",
    "--vis",
    "visualize",
    type=bool,
    is_flag=True,
    default=False,
    help="passthrough, will not write anything",
)
# @click.option(
#     "-f",
#     "--force",
#     "overwrite",
#     type=bool,
#     is_flag=True,
#     default=False,
#     help="overwrite existing file",
# )
# @click.option(
#     "--dry-run",
#     "dry_run",
#     type=bool,
#     is_flag=True,
#     default=False,
#     help="passthrough, will not write anything",
# )
def synge(
    seed,
    n_samples,
    n_feats,
    n_cluster,
    clusters_label,
    centroids,
    par_shapes,
    weight_cluster,
    distributions,
    parametres_distributions,
    chevauchement,
    parametres_shapes,
    shapes,
    scale,
    rotate,
    visualize,
):
    print(
        seed,
        n_samples,
        n_feats,
        n_cluster,
        clusters_label,
        centroids,
        par_shapes,
        weight_cluster,
        distributions,
        parametres_distributions,
        scale,
        rotate,
        shapes,
        chevauchement,
        parametres_shapes,
        visualize,
    )
    gnr = ClusterGenerator(
        seed=seed,
        n_samples=n_samples,
        n_feats=n_feats,
        n_cluster=n_cluster,
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
    if visualize:
        visualize_(X, y)
    return M


if __name__ == "__main__":
    cli()
