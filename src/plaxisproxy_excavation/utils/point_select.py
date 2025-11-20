from typing import List, Tuple, Optional, Iterable
import math
from random import Random
import random

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import Point


class NeighborPointPicker:
    """
    Spatial grid / KD-tree based neighbor search for 2D / 3D points.

    Construction modes
    ------------------
    1) From numeric arrays (recommended for performance):
        # 3D
        NeighborPointPicker(
            grid_size=1.0,
            dim=3,
            xs=xs, ys=ys, zs=zs,
            use_kdtree=True,
        )

        # 2D
        NeighborPointPicker(
            grid_size=1.0,
            dim=2,
            xs=xs, ys=ys,
            use_kdtree=True,
        )

    2) From a list of Point objects (always 3D Point; for dim=2 only x,y used):
        NeighborPointPicker(grid_size=1.0, dim=3, points=points, use_kdtree=True)
        NeighborPointPicker(grid_size=1.0, dim=2, points=points, use_kdtree=True)

    Backends
    --------
    - KD-tree (cKDTree, SciPy) when use_kdtree=True:
        - Build time: O(N log N)
        - Query: ~O(log N)
        - max_nearest_neighbor_distance can be fully vectorized with k=2.

    - Grid-based when use_kdtree=False:
        - Build: O(N)
        - Query: scan neighbor cells in growing shells, cost depends on
          local density and chosen grid_size.

    Notes
    -----
    - Coordinates can be negative; we normalize by subtracting global min
      on each axis when computing cell indices, so everything is safe.
    - For millions of points, choose grid_size such that the average cell
      population stays reasonable if you use the grid backend.
    """

    def __init__(
        self,
        grid_size: float = 1.0,
        dim: int = 3,
        points: Optional[List[Point]] = None,
        xs: Optional[Iterable[float]] = None,
        ys: Optional[Iterable[float]] = None,
        zs: Optional[Iterable[float]] = None,
        use_kdtree: bool = False,
    ) -> None:
        """
        Create a NeighborPointPicker.

        Exactly one of the following input modes must be used:
        - points: a list of Point objects
        - xs, ys (, zs): numeric coordinates

        Args
        ----
        grid_size : float
            Cell size of the spatial grid.
        dim : int
            Dimension of the space; 2 or 3. Default is 3.
        points : list[Point], optional
            List of Point instances.
        xs, ys, zs : iterable of float, optional
            Coordinate components; for dim=3, zs is required; for dim=2 it is ignored.
        use_kdtree : bool
            If True, use scipy.spatial.cKDTree for nearest neighbor queries.
            If False, use grid-based search.
        """
        if grid_size <= 0.0:
            raise ValueError("grid_size must be > 0.")

        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")

        self.grid_size: float = float(grid_size)
        self._dim: int = dim

        # Decide input mode
        from_points = points is not None
        from_arrays = xs is not None and ys is not None

        if from_points and from_arrays:
            raise ValueError(
                "You must provide either `points` OR (`xs`, `ys`[, `zs`]), not both."
            )
        if not from_points and not from_arrays:
            raise ValueError("You must provide input data (`points` OR arrays).")

        if from_points:
            if not points:
                raise ValueError("NeighborPointPicker requires at least one Point.")
            self.points: List[Point] = points
            (
                self._coords,
                self._x_min,
                self._x_max,
                self._y_min,
                self._y_max,
                self._z_min,
                self._z_max,
            ) = self._build_coords_and_bounds_from_points(points, self._dim)
        else:
            # numeric arrays branch
            self.points = []  # optional; numeric-only mode
            (
                self._coords,
                self._x_min,
                self._x_max,
                self._y_min,
                self._y_max,
                self._z_min,
                self._z_max,
            ) = self._build_coords_and_bounds_from_xyz(xs, ys, zs, self._dim)  # type: ignore[arg-type]

        # Number of cells along each axis (>= 1)
        self._nx: int = max(1, int(math.floor((self._x_max - self._x_min) / self.grid_size)) + 1)
        self._ny: int = max(1, int(math.floor((self._y_max - self._y_min) / self.grid_size)) + 1)
        if self._dim == 3:
            self._nz: int = max(1, int(math.floor((self._z_max - self._z_min) / self.grid_size)) + 1)
        else:
            self._nz = 1

        # Backend selector
        self._use_kdtree: bool = bool(use_kdtree)
        self._kdtree: Optional[cKDTree] = None

        # Grid cells (only used when use_kdtree=False)
        self._cells: List[List[int]] = []

        if self._use_kdtree:
            # KD-tree backend
            self._kdtree = cKDTree(self._coords)
        else:
            # Grid backend
            num_cells = self._nx * self._ny * self._nz
            self._cells = [[] for _ in range(num_cells)]
            self._build_grid()

    # ----------------------------------------------------------------------
    # Coordinate / bounds builders
    # ----------------------------------------------------------------------
    @staticmethod
    def _build_coords_and_bounds_from_points(
        points: List[Point],
        dim: int,
    ) -> Tuple[np.ndarray, float, float, float, float, float, float]:
        """
        Build coords array and bounding box from a list of Point objects.
        Single Python-level pass.

        dim=3 -> coords shape (N, 3), x/y/z bounds.
        dim=2 -> coords shape (N, 2), x/y bounds, z_min=z_max=0.0.
        """
        n = len(points)
        if dim == 3:
            coords = np.empty((n, 3), dtype=float)

            x0, y0, z0 = points[0].get_point()
            x0 = float(x0); y0 = float(y0); z0 = float(z0)

            coords[0, 0] = x0
            coords[0, 1] = y0
            coords[0, 2] = z0

            x_min = x_max = x0
            y_min = y_max = y0
            z_min = z_max = z0

            for i in range(1, n):
                x, y, z = points[i].get_point()
                x = float(x); y = float(y); z = float(z)

                coords[i, 0] = x
                coords[i, 1] = y
                coords[i, 2] = z

                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                if z < z_min: z_min = z
                if z > z_max: z_max = z

            return coords, x_min, x_max, y_min, y_max, z_min, z_max

        else:  # dim == 2
            coords = np.empty((n, 2), dtype=float)

            x0, y0, _z0 = points[0].get_point()
            x0 = float(x0); y0 = float(y0)

            coords[0, 0] = x0
            coords[0, 1] = y0

            x_min = x_max = x0
            y_min = y_max = y0

            for i in range(1, n):
                x, y, _z = points[i].get_point()
                x = float(x); y = float(y)

                coords[i, 0] = x
                coords[i, 1] = y

                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y

            return coords, x_min, x_max, y_min, y_max, 0.0, 0.0

    @staticmethod
    def _build_coords_and_bounds_from_xyz(
        xs: Iterable[float],
        ys: Iterable[float],
        zs: Optional[Iterable[float]],
        dim: int,
    ) -> Tuple[np.ndarray, float, float, float, float, float, float]:
        """
        Build coords array and bounding box from arrays / iterables.

        dim=3 -> require xs, ys, zs and build coords shape (N, 3).
        dim=2 -> use xs, ys; zs (if given) is ignored; coords shape (N, 2).
        """
        xs_arr = np.asarray(list(xs), dtype=float)
        ys_arr = np.asarray(list(ys), dtype=float)

        if xs_arr.size == 0:
            raise ValueError("NeighborPointPicker requires at least one coordinate.")

        if xs_arr.size != ys_arr.size:
            raise ValueError("xs and ys must have the same length.")

        if dim == 3:
            if zs is None:
                raise ValueError("For dim=3, zs must be provided.")
            zs_arr = np.asarray(list(zs), dtype=float)
            if zs_arr.size != xs_arr.size:
                raise ValueError("xs, ys, zs must have the same length.")

            coords = np.empty((xs_arr.size, 3), dtype=float)
            coords[:, 0] = xs_arr
            coords[:, 1] = ys_arr
            coords[:, 2] = zs_arr

            x_min = float(xs_arr.min())
            x_max = float(xs_arr.max())
            y_min = float(ys_arr.min())
            y_max = float(ys_arr.max())
            z_min = float(zs_arr.min())
            z_max = float(zs_arr.max())

            return coords, x_min, x_max, y_min, y_max, z_min, z_max

        else:  # dim == 2
            coords = np.empty((xs_arr.size, 2), dtype=float)
            coords[:, 0] = xs_arr
            coords[:, 1] = ys_arr

            x_min = float(xs_arr.min())
            x_max = float(xs_arr.max())
            y_min = float(ys_arr.min())
            y_max = float(ys_arr.max())
            return coords, x_min, x_max, y_min, y_max, 0.0, 0.0

    # ----------------------------------------------------------------------
    # Grid helpers (used when use_kdtree=False)
    # ----------------------------------------------------------------------
    def _flat_index(self, ix: int, iy: int, iz: int) -> int:
        """
        Convert 3D cell indices (ix, iy, iz) to a flat index in [0, nx * ny * nz).
        """
        return ix + iy * self._nx + iz * self._nx * self._ny

    def _coord_to_cell_index(self, x: float, y: float, z: float = 0.0) -> Tuple[int, int, int]:
        """
        Map a coordinate to cell indices (ix, iy, iz).
        """
        gx = int((x - self._x_min) // self.grid_size)
        gy = int((y - self._y_min) // self.grid_size)

        ix = min(max(gx, 0), self._nx - 1)
        iy = min(max(gy, 0), self._ny - 1)

        if self._dim == 3:
            gz = int((z - self._z_min) // self.grid_size)
            iz = min(max(gz, 0), self._nz - 1)
        else:
            iz = 0

        return ix, iy, iz

    def _build_grid(self) -> None:
        """
        Assign each point to a grid cell based on its coordinates.
        Only used when use_kdtree=False.
        """
        if self._dim == 3:
            for idx, (x, y, z) in enumerate(self._coords):
                ix, iy, iz = self._coord_to_cell_index(x, y, z)
                flat = self._flat_index(ix, iy, iz)
                self._cells[flat].append(idx)
        else:
            for idx, (x, y) in enumerate(self._coords):
                ix, iy, iz = self._coord_to_cell_index(x, y, 0.0)
                flat = self._flat_index(ix, iy, iz)
                self._cells[flat].append(idx)

    # ----------------------------------------------------------------------
    # Nearest neighbor core (KD-tree first, fallback to grid)
    # ----------------------------------------------------------------------
    def _nearest_core(
        self,
        qx: float,
        qy: float,
        qz: Optional[float],
        max_search_radius: Optional[int] = None,
        exclude_index: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Internal core for nearest neighbor search.

        If KD-tree is enabled, uses cKDTree.
        Otherwise falls back to grid-based search.

        qx, qy are always used; qz is only used when dim=3.
        """
        # ----- KD-tree backend -----
        if self._use_kdtree and self._kdtree is not None:
            if self._dim == 3:
                if qz is None:
                    raise ValueError("qz must be provided for dim=3.")
                q = np.array([qx, qy, qz], dtype=float)
            else:
                q = np.array([qx, qy], dtype=float)

            if exclude_index is None:
                # Basic nearest neighbor (could be the point itself)
                dist, idx = self._kdtree.query(q, k=1)
                idx_i = int(idx)
                return idx_i, self._coords[idx_i]
            else:
                # Need nearest neighbor different from exclude_index
                n_points = self._coords.shape[0]
                if n_points == 1:
                    raise RuntimeError("No neighbor other than the excluded index.")

                k = 2
                while True:
                    k = min(k, n_points)
                    dists, indices = self._kdtree.query(q, k=k)
                    dists = np.atleast_1d(dists)
                    indices = np.atleast_1d(indices).astype(int)

                    mask = indices != exclude_index
                    if not mask.any():
                        if k == n_points:
                            raise RuntimeError("No neighbor other than the excluded index.")
                        k *= 2
                        continue

                    valid_dists = dists[mask]
                    valid_indices = indices[mask]
                    pos = int(valid_dists.argmin())
                    nn_idx = int(valid_indices[pos])
                    return nn_idx, self._coords[nn_idx]

        # ----- Grid backend -----
        if not self._cells:
            raise RuntimeError("Grid backend is not initialized.")

        if self._dim == 3:
            if qz is None:
                raise ValueError("qz must be provided for dim=3.")
            q = np.array([qx, qy, qz], dtype=float)
            ix0, iy0, iz0 = self._coord_to_cell_index(qx, qy, qz)
        else:
            q = np.array([qx, qy], dtype=float)
            ix0, iy0, iz0 = self._coord_to_cell_index(qx, qy, 0.0)

        best_idx: Optional[int] = None
        best_dist2: float = float("inf")

        if max_search_radius is None:
            max_r = max(self._nx, self._ny, self._nz)
        else:
            max_r = max_search_radius

        for r in range(0, max_r + 1):
            ix_min = max(ix0 - r, 0)
            ix_max = min(ix0 + r, self._nx - 1)
            iy_min = max(iy0 - r, 0)
            iy_max = min(iy0 + r, self._ny - 1)
            iz_min = max(iz0 - r, 0)
            iz_max = min(iz0 + r, self._nz - 1)

            candidate_indices: List[int] = []

            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    for iz in range(iz_min, iz_max + 1):
                        if max(abs(ix - ix0), abs(iy - iy0), abs(iz - iz0)) != r:
                            continue
                        flat = self._flat_index(ix, iy, iz)
                        cell_indices = self._cells[flat]
                        if not cell_indices:
                            continue
                        if exclude_index is None:
                            candidate_indices.extend(cell_indices)
                        else:
                            candidate_indices.extend(
                                idx for idx in cell_indices if idx != exclude_index
                            )

            if not candidate_indices:
                continue

            cand_coords = self._coords[candidate_indices]
            diff = cand_coords - q
            dist2 = (diff * diff).sum(axis=1)

            local_min_pos = int(dist2.argmin())
            local_min = float(dist2[local_min_pos])
            if local_min < best_dist2:
                best_dist2 = local_min
                best_idx = candidate_indices[local_min_pos]

            # Heuristic early exit
            break

        if best_idx is None:
            raise RuntimeError("Nearest point search failed (no candidates found).")

        return best_idx, self._coords[best_idx]

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def get_dim(self) -> int:
        """Return the dimension (2 or 3)."""
        return self._dim

    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return (x_min, x_max, y_min, y_max, z_min, z_max).

        For dim=2, z_min and z_max are (0.0, 0.0).
        """
        return (
            self._x_min,
            self._x_max,
            self._y_min,
            self._y_max,
            self._z_min,
            self._z_max,
        )

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """
        Return the number of cells along each axis as (nx, ny, nz).

        For dim=2, nz == 1.
        """
        return self._nx, self._ny, self._nz

    def nearest_to_coord(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        max_search_radius: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Find the nearest stored point to a given coordinate.

        For dim=3, z must be provided.
        For dim=2, z is ignored.
        """
        if self._dim == 3:
            if z is None:
                raise ValueError("z must be provided for dim=3.")
            return self._nearest_core(x, y, z, max_search_radius=max_search_radius)
        else:
            return self._nearest_core(x, y, 0.0, max_search_radius=max_search_radius)

    def nearest_to_point(
        self,
        p: Point,
        max_search_radius: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Find the nearest stored point to a given Point instance.

        For dim=2, the z component of Point is ignored.
        """
        x, y, z = p.get_point()
        if self._dim == 3:
            return self._nearest_core(x, y, z, max_search_radius=max_search_radius)
        else:
            return self._nearest_core(x, y, 0.0, max_search_radius=max_search_radius)

    def pick_random_point_and_nearest(
        self,
        rng: Optional[Random] = None,
        max_search_radius: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, int, np.ndarray]:
        """
        Randomly pick one stored point, then find its nearest neighbor
        (different from itself).
        """
        n = self._coords.shape[0]
        if n == 0:
            raise ValueError("No points stored in NeighborPointPicker.")

        if rng is None:
            rng = Random()

        query_index = rng.randrange(n)
        if self._dim == 3:
            qx, qy, qz = self._coords[query_index]
            neighbor_index, neighbor_coord = self._nearest_core(
                qx, qy, qz,
                max_search_radius=max_search_radius,
                exclude_index=query_index,
            )
        else:
            qx, qy = self._coords[query_index]
            neighbor_index, neighbor_coord = self._nearest_core(
                qx, qy, 0.0,
                max_search_radius=max_search_radius,
                exclude_index=query_index,
            )
        return query_index, self._coords[query_index], neighbor_index, neighbor_coord

    def max_nearest_neighbor_distance(
        self,
        sample_step: int = 1,
        max_search_radius: Optional[int] = None,
    ) -> float:
        """
        Estimate the maximum spacing between points as:

            max_i distance(point_i, nearest_neighbor_of_i)

        sample_step:
            - 1  -> use all points (most accurate)
            - k  -> use every k-th point (approximate, faster)

        Behavior:
        - If KD-tree backend is enabled, uses vectorized cKDTree.query(k=2).
        - Otherwise, falls back to grid-based repeated queries.
        """
        n = self._coords.shape[0]
        if n < 2:
            return 0.0

        if sample_step < 1:
            raise ValueError("sample_step must be >= 1.")

        # KD-tree fast path
        if self._use_kdtree and self._kdtree is not None:
            if sample_step == 1:
                query_coords = self._coords
            else:
                indices = np.arange(0, n, sample_step, dtype=int)
                query_coords = self._coords[indices]

            # For each query point, k=2:
            #  -> first neighbor is itself (distance 0)
            #  -> second neighbor is nearest other point
            dists, idxs = self._kdtree.query(query_coords, k=2)
            nn_dists = np.atleast_2d(dists)[:, 1]  # shape (M, 2) or (M,)

            return float(nn_dists.max())

        # Grid-based fallback
        max_d2: float = 0.0

        for idx in range(0, n, sample_step):
            if self._dim == 3:
                qx, qy, qz = self._coords[idx]
                nn_idx, nn_coord = self._nearest_core(
                    qx, qy, qz,
                    max_search_radius=max_search_radius,
                    exclude_index=idx,
                )
            else:
                qx, qy = self._coords[idx]
                nn_idx, nn_coord = self._nearest_core(
                    qx, qy, 0.0,
                    max_search_radius=max_search_radius,
                    exclude_index=idx,
                )
            diff = nn_coord - self._coords[idx]
            d2 = float((diff * diff).sum())
            if d2 > max_d2:
                max_d2 = d2

        return math.sqrt(max_d2)


# Optional alias to keep the old name
neighbor_point_picker = NeighborPointPicker
