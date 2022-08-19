# Reconstruction of surfaces from point cloud data

In this project, we give a new method for the manifold reconstruction of surfaces represented in high dimensional spaces as point cloud data, based on the computation of quotient identifications from the original data onto a 2-dimensional projection. 

Our method is based on the “naive” algorithm given in the paper [[1]](#1) which provides a general framework called Riemannian manifold learning for manifold reconstruction from point clouds. The framework attempts to preserve the underlying manifold's intrinsic Riemannian structure from the data,
by preserving geodesic distances to a base point and local angles, in the projection to a lower dimensional representation. When applied to point cloud data representing surfaces (2-manifolds), we obtain a 2-dimensional projection.

Our contribution is to provide a algorithm to preserve, in these projection, the global structure of the original surfaces. The algorithm identifies the edges which make up the boundary of the projection and how these edges are glued together in order to recover the original geometry. The novelty of this idea is that it allows us to differentiate between the projections of a cylinder and plane, as well as represent more complex geometries like the real projective plane in 2-dimensions.

## Installation

In the terminal, go to the folder which contains the folder `qrml_pack` and enter the following:
```console
>> python -m pip install -e qrml_pack/
```
To import the package, enter the following:
```python
import qrml
```

## Usage

We give a simple example of applying our algorithm to a cylinder point cloud with a thousand points.

```python
import qrml
import numpy             as np

n_points = 1000
theta = np.random.uniform(low=0, high=2*np.pi, size=n_points)
x = np.cos(theta)
y = np.sin(theta)
z = np.random.uniform(low=-3, high=3, size=n_points)
cylinder = np.stack([x, y, z], axis=-1)
```
<img src="https://github.com/shesturnedtheweansagainstus/quotient_rml/blob/main/images/cylinder.png" width="200" height="200" />

### References
<a id="1">[1]</a> 
Tong, L., Zha, H. Riemannian manifold learning. *IEEE Transactions on Pattern Analysis
and Machine Intelligence* 30.5 (2008): 796-809.
