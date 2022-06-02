# Data generator

![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
A high-dimensional synthetic data generator to evaluate and compare
allow to evaluate and compare clustering algorithms, especially those dedicated to complex data spaces.

## Foreword

### Data generation

In an effort to systematically generate test data sets for data analysis, we use certain mathematical tools such as probability distributions. data, we use certain mathematical tools such as probability distributions. By applying these principles, the proposed method provides the mechanism by which the data sets are not only are not only automatically generated but also controlled by the parameters of the user's input. the user's input

### Cluster positioning

- Once the data has been generated, the tool will then determine how and where to place these clusters in the
the output space.

For each cluster we have:

1. Matrix of generated data
a matrix Xj of size (nj ,d).

    j : index of the cluster

    nj : number of points in the cluster.

    d : number of total dimensions in the space.

2. Cluster form
There are two main shapes in the tool (Hyper sphere and Hyper Rectangle)
3. Centroids and the parameter of the shape (noted: a)

a : parameter which corresponds to a radius ( hyper spherical main shape) or length (
hyper rectangle)

### Application of shapes to clusters and positioning

- It is well recognized that the performance of different data analysis algorithms depends
on the test data sets. Among the existing clustering algorithms, such as
partitioning methods, which can easily identify clusters with irregular shapes, but they are
shapes, but they are unable to find clusters with irregular shapes and tend to divide a cluster into
to divide a cluster into different groups. Although density-based methods can
handle clusters of arbitrary shapes and sizes, they are very sensitive to the density of each
of each cluster, which will lead to failure in detecting clusters with unevenly distributed data.
unevenly distributed data. A recent study also shows that some detection and
clustering analysis algorithms are actually complementary to each other.
In our synthetic data generation method. the user can configure the density,
distribution and shape of the cluster.
the shape of each cluster will be expressed as a logical expression (Normal form
conjunctive form).

### Application of cluster overlap

- To apply the overlap, the 2nd cluster is progressively brought closer to the 1st one in a uniform way on each axis. We then calculate the number of points that overlap
with the shape of the 1st cluster.

### Imbrication

- We want unlimited nesting types, and this is the case with our tool, the nesting
can be set by the user in a totally manual way by manipulating the shape, the cluster
placement of the clusters as well as the overlapping.

## Technology

### Nltk

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

### RE (Regular Expression)

A regular expression is a method used in programming for pattern matching. Regular expressions provide a flexible and concise means to match strings of text.

## For developers

### Prerequisites

#### Python

The repository targets python `3.8` and higher.

#### Poetry

The repository uses [Poetry](https://python-poetry.org) as python packaging and dependency management. Be sure to have it properly installed before.

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

#### Docker

You can follow the link below on how to install and configure **Docker** on your local machine:

- [Docker Install Documentation](https://docs.docker.com/install/)

### Build

Project is built by [poetry](https://python-poetry.org).

```sh
poetry install
```

### Lint

Code linting is performed by [flake8](https://flake8.pycqa.org).

> ⚠️ Be sure to write code compliant with `flake8` rules or else you'll be rejected by the CI.

```sh
poetry run flake8 --count --show-source --statistics
```

### Unit Test

Unit tests are performed by the [unittest](https://docs.python.org) testing framework.

> ⚠️ Be sure to write tests that succeed or else you'll be rejected by the CI.

```sh
poetry run python -m unittest discover
```

### Build & run docker image (locally)

Build a local docker image using the following command line:

```sh
docker build -t converter-excel-to-csv .
```

Once built, you can run the container locally with the following command line:

```sh
docker run -ti --rm converter-excel-to-csv
```

## Contributing

So you want to contribute? Great. We appreciate any help you're willing to give. Don't hesitate to open issues and/or
submit pull requests.
