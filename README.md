# Reconstruction of surfaces from point cloud data

In this project, we give a new method for the manifold reconstruction of surfaces represented in high dimensional space as point cloud data, based on the computation of quotient identifications from the original data onto a 2-dimensional projection. 

Our method is based on the “naive” algorithm given in the paper [[1]](#1) which provides a general framework called Riemannian manifold learning for manifold reconstruction from point clouds. The framework attempts to preserve the underlying manifold's intrinsic Riemannian structure from the data,
by preserving geodesic distances to a base point and local angles, in the projection to a lower dimensional representation. When applied to point cloud data representing surfaces (2-manifolds), we obtain a 2-dimensional projection.

Our contribution is to provide a algorithm to preserve, in these projection, the global structure of the original surfaces. The algorithm identifies the edges which make up the boundary of the projection and how these edges are glued together in order to recover the original geometry. The novelty of this idea is that it allows us to differentiate between the projections of a cylinder and plane, as well as represent more complex geometries like the real projective plane in 2-dimensions.

## Prerequisites

This package requires the following python packages:

```python
numpy
scikit-learn
matplotlib
scipy
kneed
gurobipy
```

## Installation

In the terminal, go to the folder which contains the folder `qrml_pack` and enter the following:
```console
>> python -m pip install -e qrml_pack/
```
To import the package in python, enter the following:
```python
import qrml
```

## Usage

We give a simple example of applying our algorithm to a point cloud with a thousand points, uniformly sampled from a cylinder of radius one and height six.

```python
import qrml
import numpy             as np

n_points = 1000
np.random.seed(1)

theta = np.random.uniform(low=0, high=2*np.pi, size=n_points)
x = np.cos(theta)
y = np.sin(theta)
z = np.random.uniform(low=-3, high=3, size=n_points)

cylinder = np.stack([x, y, z], axis=-1)
```
<img src="https://github.com/shesturnedtheweansagainstus/quotient_rml/blob/main/images/cylinder.png" width="600" height="600" />

We encapsulate our projection/quotient data and functions in the object `qrml.Simplex`. We have the parameters `k`, `threshold_var`, `edge_sen` and `k0` for our implementation of the "naive" algorithm in [[1]](#1). The method `build_simplex` computes a 1-skeleton on our pointcloud which approximates the underlying manifold structure of our data and `normal_coords` computes the projection.

```python
params = {'k':10, 'threshold_var':0.08, 'edge_sen':1, 'k0':100}

S = qrml.Simplex()
S.build_simplex(cylinder, **params)
S.normal_coords(**params)
```

We plot our projection and compute the boundary of our projection via `show_boundary`. We give this boundary an orientation and plot its correspondence to the original point cloud.

```python
S.show_boundary(alpha=1, tol=2, c=cylinder[:, 2], show_pointcloud=True, **params)
```

<img src="https://github.com/shesturnedtheweansagainstus/quotient_rml/blob/main/images/cylinder_projection.jpeg" width="600" height="600" />
<img src="https://github.com/shesturnedtheweansagainstus/quotient_rml/blob/main/images/cylinder_3d_boundary.jpeg" width="600" height="600"/> 

We compute and plot the quotient identifications of our boundary via `plot_quotient`. The main parameters of this method are `alpha`, `tol`, `quotient_tol`, `tol1`. The dotted lines represent non-glued edges and the solid lines represent glued edges. Glued edges with the same colour map are glued together in the orientation specified by the gradient of the colouring.

We see that the below representation matches the cannonical construction of the cylinder from a square via the induced quotient topology from identifying the top and bottom edges with the same orientation. 

```python
_ = S.plot_quotient(c=cylinder[:, 2], alpha=1, tol=2, quotient_tol=15, tol1=5, show_pointcloud=True)
```

<img src="https://github.com/shesturnedtheweansagainstus/quotient_rml/blob/main/images/cylinder_quotient.jpeg" />

More detailed explanations of the above functions and parameters as well as diagnostics and limitations can be found in the notebooks in `tutorials`. 

## Acknowledgements

This project was funded by the London Mathematical Society Undergraduate Research Bursaries (2022) and was supervised by [Ximena Fernández](https://ximenafernandez.github.io/).

### References
<a id="1">[1]</a> 
Tong, L., Zha, H. Riemannian manifold learning. *IEEE Transactions on Pattern Analysis
and Machine Intelligence* 30.5 (2008): 796-809.
