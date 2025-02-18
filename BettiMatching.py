import numpy as np
import torch
import heapq
import copy
import random as rd
from operator import itemgetter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import panel as pn
from panel.interact import interact, fixed
from panel import widgets

pn.extension()
import gudhi.wasserstein


class UnionFind:
    def __init__(self, n, dual=False):
        self.n = n
        self.dual = dual
        self.parent = list(range(n))
        self.rank = n * [0]
        self.birth = list(range(n))

    def set_birth(self, x, val):
        self.birth[x] = val
        return

    def get_birth(self, x):
        y = self.find(x)
        return self.birth[y]

    def get_parent(self, x):
        return self.parent[x]

    def find(self, x):
        y = x
        z = self.parent[y]
        while z != y:
            y = z
            z = self.parent[y]
        y = self.parent[x]
        while z != y:
            self.parent[x] = z
            x = y
            y = self.parent[x]
        return z

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            if self.dual == False:
                self.birth[x] = min(self.birth[x], self.birth[y])
            else:
                self.birth[x] = max(self.birth[x], self.birth[y])
        else:
            self.parent[x] = y
            if self.dual == False:
                self.birth[y] = min(self.birth[x], self.birth[y])
            else:
                self.birth[y] = max(self.birth[x], self.birth[y])
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def get_component(self, x):
        component = []
        x = self.find(x)
        for y in range(self.n):
            z = self.find(y)
            if z == x:
                component.append(y)
        return component


class UnionFindBirthValue(UnionFind):

    def __init__(self, n, dual=False):
        super().__init__(n, dual)
        self.death_values = n * [0]

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y, threshold):
        x_comp = self.find(x)
        y_comp = self.find(y)

        if x_comp == y_comp:
            # case where x and y are already in the same component
            return

        if x_comp > y_comp:
            if self.dual:
                self.parent[y_comp] = x_comp
                self.death_values[y_comp] = threshold
            else:
                self.parent[x_comp] = y_comp
                self.death_values[x_comp] = threshold
        else:
            if self.dual:
                self.parent[x_comp] = y_comp
                self.death_values[x_comp] = threshold
            else:
                self.parent[y_comp] = x_comp
                self.death_values[y_comp] = threshold

    def get_birth(self, x):
        return self.find(x)

    def set_birth(self, x, val):
        pass

    def find_assigned_comp_at_thr(self, x, threshold):
        y = x
        if x < 0:
            return -1
        while self.death_values[x] >= threshold and self.parent[x] != x:
            x = self.parent[x]

        return x

    def component_entry_value(self, x, component_idx, birth_threshold):
        if x < 0:
            return -1

        if x == component_idx:
            return birth_threshold

        while self.parent[x] != x and self.parent[x] != component_idx:
            x = self.parent[x]

        if self.parent[x] == component_idx:
            return self.death_values[x]
        else:
            # case where x is never part of the component
            return -1

    def belongs_to_component(self, x, component_idx, threshold):
        if x < 0:
            return False

        if x == component_idx:
            return True

        while self.parent[x] != component_idx and self.death_values[x] >= threshold:
            x = self.parent[x]

        return self.parent[x] == component_idx and self.death_values[x] >= threshold


class UnionFindWithHistory(UnionFind):
    def __init__(self, n, dual=False):
        super().__init__(n, dual)
        self.next_component = n * [(0, 0)]
        self.death_value = n * [0]

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y, threshold):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            self.death_value[y] = threshold
            if self.dual == False:
                self.birth[x] = min(self.birth[x], self.birth[y])
            else:
                self.birth[x] = max(self.birth[x], self.birth[y])
        else:
            self.parent[x] = y
            self.death_value[x] = threshold
            if self.dual == False:
                self.birth[y] = min(self.birth[x], self.birth[y])
            else:
                self.birth[y] = max(self.birth[x], self.birth[y])
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def find_assigned_comp_at_thr(self, x, threshold):
        while self.death_value[x] > threshold and self.parent[x] != x:
            x = self.parent[x]
        return x


class BoundaryMatrix:
    def __init__(self):
        self.columns = {}
        self.pivots = {}
        self.columns_to_reduce = None
        self.reduced = False

    def set_one(self, i, j):
        if j not in self.columns.keys():
            self.columns[j] = [-i]
            return

        if -i not in self.columns[j]:
            heapq.heappush(self.columns[j], -i)
            return

    def add_column(self, i, j):
        # self.columns[j] = list(heapq.merge(self.columns[j],self.columns[i]))
        # self.columns[j].extend(self.columns[i])
        # heapq.heapify(self.columns[j])
        heapq.heappop(self.columns[j])
        for item in self.columns[i][1:]:
            if item == self.columns[j][0]:
                heapq.heappop(self.columns[j])
            else:
                heapq.heappush(self.columns[j], item)
        return

    def get_pivot(self, j):
        if j not in self.columns.keys():
            return -1

        count = 0
        while (len(self.columns[j]) != 0) and (count % 2 == 0):
            pivot = -heapq.heappop(self.columns[j])
            count = 1
            while (len(self.columns[j]) != 0) and (-self.columns[j][0] == pivot):
                heapq.heappop(self.columns[j])
                count += 1
        if count % 2 == 0:
            return -1

        else:
            heapq.heappush(self.columns[j], -pivot)
            return pivot

    def reduce(self, clearing=False):
        if self.reduced == True:
            return

        if self.columns_to_reduce == None:
            clearing = False
            self.columns_to_reduce = [sorted(self.columns.keys()), [], []]
        for dim in [2, 1]:
            for column in self.columns_to_reduce[dim]:
                pivot = self.get_pivot(column)
                while pivot in self.pivots.keys():
                    previous = self.pivots[pivot]
                    self.add_column(previous, column)
                    pivot = self.get_pivot(column)
                if pivot != -1:
                    self.pivots[pivot] = column
                    if clearing == True and dim != 1:
                        self.columns.pop(pivot)
                        self.columns_to_reduce[dim - 1].remove(pivot)
                else:
                    self.columns.pop(column)
        self.reduced = True
        return

    def get_Pairings(self):
        return list(zip(self.pivots.keys(), self.pivots.values()))

    def get_column(self, j):
        column = []
        if j not in self.columns.keys():
            return column

        column_heap = copy.deepcopy(self.columns[j])
        while len(column_heap) != 0:
            pivot = -heapq.heappop(column_heap)
            count = 1
            while (len(column_heap) != 0) and (-column_heap[0] == pivot):
                heapq.heappop(column_heap)
                count += 1
            if count % 2 == 1:
                column.append(pivot)
        return column


class CubicalPersistence:
    def __init__(
        self,
        Picture,
        relative=False,
        reduced=False,
        filtration="sublevel",
        construction="V",
        valid="positive",
        get_image_columns_to_reduce=False,
        get_critical_edges=False,
        training=False,
        birth_UF=False,
    ):
        self.reduced = reduced
        assert filtration in ["sublevel", "superlevel"]
        self.filtration = filtration
        assert construction in ["V", "T"]
        self.construction = construction
        assert valid in ["all", "nonnegative", "positive"]
        self.valid = valid
        assert not (get_image_columns_to_reduce and get_critical_edges)
        self.get_image_columns_to_reduce = get_image_columns_to_reduce
        self.get_critical_edges = get_critical_edges
        if self.get_critical_edges:
            self.critical_edges = []
        if type(Picture) == torch.Tensor:
            Picture = torch.squeeze(Picture)
        self.m, self.n = Picture.shape
        if relative == False:
            self.PixelMap = Picture
        else:
            self.m += 2
            self.n += 2
            if type(Picture) == torch.Tensor:
                if self.filtration == "sublevel":
                    min = torch.min(Picture)
                    if training:
                        self.PixelMap = min * torch.ones((self.m, self.n)).cuda()
                    else:
                        self.PixelMap = min * torch.ones((self.m, self.n))
                    self.PixelMap[1 : self.m - 1, 1 : self.n - 1] = Picture
                else:
                    max = torch.max(Picture)
                    if training:
                        self.PixelMap = max * torch.ones((self.m, self.n)).cuda()
                    else:
                        self.PixelMap = max * torch.ones((self.m, self.n))
                    self.PixelMap[1 : self.m - 1, 1 : self.n - 1] = Picture
            else:
                if self.filtration == "sublevel":
                    min = np.min(Picture)
                    self.PixelMap = min * np.ones((self.m, self.n))
                    self.PixelMap[1 : self.m - 1, 1 : self.n - 1] = Picture
                else:
                    max = np.max(Picture)
                    self.PixelMap = max * np.ones((self.m, self.n))
                    self.PixelMap[1 : self.m - 1, 1 : self.n - 1] = Picture
        if self.construction == "V":
            self.M = 2 * self.m - 1
            self.N = 2 * self.n - 1
        else:
            self.M = 2 * self.m + 1
            self.N = 2 * self.n + 1
        if type(self.PixelMap) == torch.Tensor:
            self.ValueMap = torch.zeros((self.M, self.N))
        else:
            self.ValueMap = np.zeros((self.M, self.N))
        self.IndexMap = -np.ones((self.M, self.N), dtype=int)
        self.num_cubes = self.M * self.N
        self.num_edges = int((self.M * self.N - 1) / 2)
        self.edges = self.num_edges * [0]
        self.coordinates = self.num_cubes * [0]
        self.intervals = [[], []]
        self.columns_to_reduce = [[], [], []]
        self.set_CubeMap()
        self.compute_persistence(valid=valid, birth_uf=birth_UF)

    def set_CubeMap(self):
        if type(self.PixelMap) == torch.Tensor:
            if self.filtration == "sublevel":
                PixelMap = np.array(self.PixelMap.cpu().detach().numpy(), dtype=float)
            else:
                PixelMap = -np.array(self.PixelMap.cpu().detach().numpy(), dtype=float)
        else:
            if self.filtration == "sublevel":
                PixelMap = np.array(self.PixelMap, dtype=float)
            else:
                PixelMap = -np.array(self.PixelMap, dtype=float)
        if self.construction == "V":
            counter = int(self.num_cubes - 1)
            counter_edges = int(self.num_edges - 1)
            max = np.max(PixelMap)
            while max != -np.infty:
                argmax = np.where(PixelMap == max)
                for i, j in zip(argmax[0], argmax[1]):
                    for k in [-1, 1]:
                        for l in [-1, 1]:
                            if (
                                2 * i + k >= 0
                                and 2 * i + k <= self.M - 1
                                and 2 * j + l >= 0
                                and 2 * j + l <= self.N - 1
                            ):
                                if self.IndexMap[2 * i + k, 2 * j + l] == -1:
                                    self.ValueMap[2 * i + k, 2 * j + l] = self.PixelMap[
                                        i, j
                                    ]
                                    self.IndexMap[2 * i + k, 2 * j + l] = counter
                                    self.coordinates[counter] = (2 * i + k, 2 * j + l)
                                    counter = int(counter - 1)
                for i, j in zip(argmax[0], argmax[1]):
                    for k in [-1, 1]:
                        if 2 * i + k >= 0 and 2 * i + k <= self.M - 1:
                            if self.IndexMap[2 * i + k, 2 * j] == -1:
                                self.ValueMap[2 * i + k, 2 * j] = self.PixelMap[i, j]
                                self.IndexMap[2 * i + k, 2 * j] = counter
                                self.coordinates[counter] = (2 * i + k, 2 * j)
                                self.edges[counter_edges] = counter
                                counter = int(counter - 1)
                                counter_edges = int(counter_edges - 1)
                        if 2 * j + k >= 0 and 2 * j + k <= self.N - 1:
                            if self.IndexMap[2 * i, 2 * j + k] == -1:
                                self.ValueMap[2 * i, 2 * j + k] = self.PixelMap[i, j]
                                self.IndexMap[2 * i, 2 * j + k] = counter
                                self.coordinates[counter] = (2 * i, 2 * j + k)
                                self.edges[counter_edges] = counter
                                counter = int(counter - 1)
                                counter_edges = int(counter_edges - 1)
                for i, j in zip(argmax[0], argmax[1]):
                    self.ValueMap[2 * i, 2 * j] = self.PixelMap[i, j]
                    self.IndexMap[2 * i, 2 * j] = counter
                    self.coordinates[counter] = (2 * i, 2 * j)
                    counter = int(counter - 1)
                    PixelMap[i, j] = -np.infty
                max = np.max(PixelMap)
        else:
            counter = int(0)
            counter_edges = int(0)
            min = np.min(PixelMap)
            while min != np.infty:
                argmin = np.where(PixelMap == min)
                for i, j in zip(argmin[0], argmin[1]):
                    for k in [-1, 1]:
                        for l in [-1, 1]:
                            if self.IndexMap[2 * i + 1 + k, 2 * j + 1 + l] == -1:
                                self.ValueMap[2 * i + 1 + k, 2 * j + 1 + l] = (
                                    self.PixelMap[i, j]
                                )
                                self.IndexMap[2 * i + 1 + k, 2 * j + 1 + l] = counter
                                self.coordinates[counter] = (
                                    2 * i + 1 + k,
                                    2 * j + 1 + l,
                                )
                                counter = int(counter + 1)
                for i, j in zip(argmin[0], argmin[1]):
                    for k in [-1, 1]:
                        if self.IndexMap[2 * i + 1 + k, 2 * j + 1] == -1:
                            self.ValueMap[2 * i + 1 + k, 2 * j + 1] = self.PixelMap[
                                i, j
                            ]
                            self.IndexMap[2 * i + 1 + k, 2 * j + 1] = counter
                            self.coordinates[counter] = (2 * i + 1 + k, 2 * j + 1)
                            self.edges[counter_edges] = counter
                            counter = int(counter + 1)
                            counter_edges = int(counter_edges + 1)
                        if self.IndexMap[2 * i + 1, 2 * j + 1 + k] == -1:
                            self.ValueMap[2 * i + 1, 2 * j + 1 + k] = self.PixelMap[
                                i, j
                            ]
                            self.IndexMap[2 * i + 1, 2 * j + 1 + k] = counter
                            self.coordinates[counter] = (2 * i + 1, 2 * j + 1 + k)
                            self.edges[counter_edges] = counter
                            counter = int(counter + 1)
                            counter_edges = int(counter_edges + 1)
                for i, j in zip(argmin[0], argmin[1]):
                    self.ValueMap[2 * i + 1, 2 * j + 1] = self.PixelMap[i, j]
                    self.IndexMap[2 * i + 1, 2 * j + 1] = counter
                    self.coordinates[counter] = (2 * i + 1, 2 * j + 1)
                    counter = int(counter + 1)
                    PixelMap[i, j] = np.infty
                min = np.min(PixelMap)

    def index_to_coordinates(self, idx):
        return self.coordinates[idx]

    def pixel_index_map(self):
        if self.construction == "V":
            return self.IndexMap[::2, ::2]
        elif self.construction == "T":
            raise NotImplementedError("Function for T-construction is not implemented")
        else:
            raise ValueError(
                f"Invalid construction: {self.construction}. Choose from: 'V', 'T'"
            )

    def index_to_dim(self, idx):
        i, j = self.index_to_coordinates(idx)
        if i % 2 == 0 and j % 2 == 0:
            dim = 0
        elif i % 2 + j % 2 == 1:
            dim = 1
        else:
            dim = 2
        return dim

    def index_to_value(self, idx):
        if idx == np.infty or idx == -np.infty:
            if self.filtration == "sublevel":
                return np.infty

            else:
                return -np.infty

        x, y = self.index_to_coordinates(idx)
        return self.ValueMap[x, y]

    def fine_to_coarse(self, interval):
        return (self.index_to_value(interval[0]), self.index_to_value(interval[1]))

    def valid_interval(self, interval, valid="positive"):
        if valid in ["all", "nonnegative"]:
            return True

        else:
            if self.filtration == "sublevel":
                return self.index_to_value(interval[0]) < self.index_to_value(
                    interval[1]
                )

            else:
                return self.index_to_value(interval[0]) > self.index_to_value(
                    interval[1]
                )

    def birth_death_pixels_dim0(self, dim: int, start: int = 1):

        def original_index_v(x, y):
            return (int(x / 2), int(y / 2))

        def original_index(x, y):
            return (x, y)

        func = original_index_v if self.construction == "V" else original_index

        sorted_intervals = self.sorted_intervals(dim, refined=True)

        if start >= len(sorted_intervals):
            return []

        birth_points = [
            func(*self.index_to_coordinates(x)) for x, y in sorted_intervals[start:]
        ]
        death_points = [
            func(*self.index_to_coordinates(self.get_generating_vertex(y)))
            for x, y in sorted_intervals[start:]
        ]
        combined = torch.Tensor(list(zip(birth_points, death_points))).to(torch.long)
        return combined

    def important_components(
        self, dim: int, num_intervals: int = 1, refined: bool = False
    ):
        num_intervals = max(1, min(num_intervals, len(self.intervals[dim])))
        return self.sorted_intervals(dim, refined=refined)[:num_intervals]

    def sorted_intervals(
        self, dim: int, return_indices: bool = False, refined: bool = False
    ):

        def interval_length(interval):
            index, interval = interval
            if self.filtration == "superlevel":
                death_value = 0.0 if interval[1] == -np.infty else interval[1]
            else:
                death_value = 1.0 if interval[1] == np.infty else interval[1]

            return abs(death_value - interval[0])

        interval_actual_values = [
            self.fine_to_coarse(interval) for interval in self.intervals[dim]
        ]
        enumerated_intervals = list(enumerate(interval_actual_values))

        sorted_enumerated_intervals = sorted(
            enumerated_intervals, key=interval_length, reverse=True
        )

        if refined:
            if return_indices:
                return [
                    (index, self.intervals[dim][index])
                    for index, interval in sorted_enumerated_intervals
                ]
            else:
                return [
                    self.intervals[dim][index]
                    for index, interval in sorted_enumerated_intervals
                ]
        else:
            if return_indices:
                return sorted_enumerated_intervals
            else:
                return [interval for index, interval in sorted_enumerated_intervals]

    def get_boundary(self, idx):
        boundary = []
        x, y = self.index_to_coordinates(idx)
        if x % 2 != 0:
            boundary.extend([self.IndexMap[x - 1, y], self.IndexMap[x + 1, y]])
        if y % 2 != 0:
            boundary.extend([self.IndexMap[x, y - 1], self.IndexMap[x, y + 1]])
        return boundary

    def get_dual_boundary(self, idx):
        boundary = []
        x, y = self.index_to_coordinates(idx)
        if x % 2 == 0:
            if x == 0:
                boundary.extend([self.num_cubes, self.IndexMap[x + 1, y]])
            elif x == self.M - 1:
                boundary.extend([self.num_cubes, self.IndexMap[x - 1, y]])
            else:
                boundary.extend([self.IndexMap[x - 1, y], self.IndexMap[x + 1, y]])
        if y % 2 == 0:
            if y == 0:
                boundary.extend([self.num_cubes, self.IndexMap[x, y + 1]])
            elif y == self.N - 1:
                boundary.extend([self.num_cubes, self.IndexMap[x, y - 1]])
            else:
                boundary.extend([self.IndexMap[x, y - 1], self.IndexMap[x, y + 1]])
        return boundary

    def compute_dim0_birthUF(self, valid="positive"):
        self.intervals[0] = []
        UF = UnionFindBirthValue(self.num_cubes, dual=False)

        for edge in self.columns_to_reduce[1]:
            boundary = self.get_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])

            if x == y:
                continue

            birth_x = UF.get_birth(x)
            birth_y = UF.get_birth(y)

            if birth_x > birth_y:
                birth = birth_x
                interval_id = x
            else:
                birth = birth_y
                interval_id = y

            if self.valid_interval((birth, edge), valid=valid):
                self.intervals[0].append((birth, edge))
            UF.union(x, y, self.index_to_value(edge))

        birth_largest = UF.find(0)
        self.intervals[0].insert(0, (birth_largest, np.infty))
        return UF

    def compute_dim0(self, valid="positive"):
        if self.reduced == False:
            self.intervals[0] = [(0, np.infty)]
        else:
            self.intervals[0] = []
        UF = UnionFind(self.num_cubes, dual=False)
        for edge in self.columns_to_reduce[1]:
            boundary = self.get_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])
            if x == y:
                continue
            birth = max(UF.get_birth(x), UF.get_birth(y))
            if self.valid_interval((birth, edge), valid=valid):
                self.intervals[0].append((birth, edge))
            UF.union(x, y)
        return UF

    def compute_dim1(self, valid="positive"):
        if self.get_image_columns_to_reduce:
            UF = UnionFind(self.num_cubes + 1, dual=True)
            for edge in self.edges[::-1]:
                boundary = self.get_dual_boundary(edge)
                x = UF.find(boundary[0])
                y = UF.find(boundary[1])
                if x == y:
                    self.columns_to_reduce[1].append(edge)
                    continue
                birth = min(UF.get_birth(x), UF.get_birth(y))
                self.columns_to_reduce[2].append(birth)
                if self.valid_interval((edge, birth), valid=valid):
                    self.intervals[1].append((edge, birth))
                UF.union(x, y)
            self.columns_to_reduce[1].reverse()
            self.columns_to_reduce[2].sort()
        elif self.get_critical_edges:
            UF = UnionFind(self.num_cubes + 1, dual=True)
            for edge in self.edges[::-1]:
                boundary = self.get_dual_boundary(edge)
                x = UF.find(boundary[0])
                y = UF.find(boundary[1])
                if x == y:
                    self.columns_to_reduce[1].append(edge)
                    continue
                self.critical_edges.append(edge)
                birth = min(UF.get_birth(x), UF.get_birth(y))
                if self.valid_interval((edge, birth), valid=valid):
                    self.intervals[1].append((edge, birth))
                UF.union(x, y)
            self.columns_to_reduce[1].reverse()
        else:
            UF = UnionFind(self.num_cubes + 1, dual=True)
            for edge in self.edges[::-1]:
                boundary = self.get_dual_boundary(edge)
                x = UF.find(boundary[0])
                y = UF.find(boundary[1])
                if x == y:
                    self.columns_to_reduce[1].append(edge)
                    continue
                birth = min(UF.get_birth(x), UF.get_birth(y))
                if self.valid_interval((edge, birth), valid=valid):
                    self.intervals[1].append((edge, birth))
                UF.union(x, y)
            self.columns_to_reduce[1].reverse()
        return UF

    def compute_persistence(self, valid="positive", birth_uf=False):
        self.uf_1 = self.compute_dim1(valid=valid)
        if birth_uf:
            self.uf_0 = self.compute_dim0_birthUF(valid=valid)
        else:
            self.uf_0 = self.compute_dim0(valid=valid)

        return

    def get_intervals(self, refined=False):
        if refined:
            return copy.deepcopy(self.intervals)

        intervals = [
            [self.fine_to_coarse(interval) for interval in self.intervals[dim]]
            for dim in range(2)
        ]
        return intervals

    def get_Betti_numbers(self, threshold=0.5):
        betti = [0, 0]
        for dim in [0, 1]:
            for i, j in self.intervals[dim]:
                if self.valid_interval((i, j), valid="positive"):
                    a = self.index_to_value(i)
                    b = self.index_to_value(j)
                    if self.filtration == "sublevel":
                        if a <= threshold and threshold < b:
                            betti[dim] += 1
                    else:
                        if a >= threshold and threshold > b:
                            betti[dim] += 1
        return betti

    def plot_image(self):
        plt.figure(figsize=(4, 4))
        plt.imshow(self.PixelMap, cmap="gray")
        plt.axis("off")

    def BarCode(self, color="r", ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        fig, ax = plt.subplots(figsize=(w, h))
        intervals = [
            self.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.intervals[dim]
        ]
        if self.filtration == "sublevel":
            max_val = max(
                intervals, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val = min(intervals, key=lambda x: x[0])[0]
        else:
            max_val = max(intervals, key=lambda x: x[0])[0]
            min_val = min(
                intervals, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        x_min = min_val - (max_val - min_val) * 0.1
        x_max = max_val + (max_val - min_val) * 0.1
        for dim in range(2):
            num_intervals = len(self.intervals[dim])
            height = dim + 1 / (num_intervals + 1)
            for i, j in self.intervals[dim]:
                if j == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.index_to_value(i), x_max),
                            (height, height),
                            color=color,
                        )
                    else:
                        plt.plot(
                            (self.index_to_value(i), x_min),
                            (height, height),
                            color=color,
                        )
                else:
                    plt.plot(self.fine_to_coarse((i, j)), (height, height), color=color)
                height += 1 / (num_intervals + 1)
        plt.plot((x_min, x_max), (1, 1), color="k", linewidth=0.8)
        plt.ylabel("Dimension")
        plt.xlim(x_min, x_max)
        plt.ylim(0, 2)
        plt.yticks([0.5, 1.5], [0, 1])
        return

    def component_map(
        self,
        threshold: float = 0.5,
        component_idx: int = 0,
        base_prob: float = 0.0,
        device: str = "cpu",
    ):
        pixel_index_map = torch.from_numpy(self.pixel_index_map())
        pixel_map = (
            torch.from_numpy(self.PixelMap)
            if type(self.PixelMap) == np.ndarray
            else self.PixelMap
        )
        pixel_index_map = torch.where(pixel_map > threshold, pixel_index_map, -1)

        component_map = pixel_index_map.apply_(
            lambda x: self.uf_0.belongs_to_component(x, component_idx, threshold)
        )

        if device == "cuda":
            component_map = component_map.cuda()
            pixel_map = pixel_map.cuda()

        print(f"component map max: {torch.max(component_map)}")
        print(f"component map min: {torch.min(component_map)}")
        print(
            f"histogram of component map: {torch.histc(component_map.to(torch.float32), bins=10)}"
        )

        print(f"pixel map max: {torch.max(pixel_map)}")
        print(f"pixel map min: {torch.min(pixel_map)}")
        print(
            f"histogram of pixel map: {torch.histc(pixel_map.to(torch.float32), bins=10)}"
        )

        component_map = component_map * pixel_map
        return torch.where(component_map == 0, base_prob, component_map)

    def likelihood_map(
        self, threshold: float = 0.5, component_idx: int = 0, device: str = "cpu"
    ):
        component_map = self.component_map(
            threshold=threshold, component_idx=component_idx, device=device
        )
        likelihood_map = torch.where(component_map == component_idx, 1.0, 0.0)
        return likelihood_map

    def component_map_entry_value(
        self,
        interval: tuple,
        threshold: float = 0.5,
        base_prob: float = 0.0,
        device: str = "cpu",
    ):
        """
        This function checks for each pixel if it is part of the component of interest at the given threshold. If it is part of the component,
        it returns the value when the pixel was added to the component, else it returns the base probability
        params:
            threshold: float, threshold to use for the component map
            interval: tuple, (birth_value, death_value) of the component to analyze
            base_prob: float, base probability to use for the component map, for all the pixels that are not part of the intended component
        returns:
            component_map: torch.Tensor, component map at the given threshold
        """
        component_idx = interval[0]
        birth_threshold = self.index_to_value(component_idx)
        death_threshold = self.index_to_value(interval[1])

        if threshold < death_threshold:
            raise ValueError(
                f"Threshold {threshold} is less than death threshold {death_threshold} for component {interval}. This would mean that the component is already merged into another component at this threshold"
            )

        base_prob = min(base_prob, threshold)

        pixel_index_map = torch.from_numpy(self.pixel_index_map())
        pixel_map = (
            torch.from_numpy(self.PixelMap)
            if type(self.PixelMap) == np.ndarray
            else self.PixelMap
        )
        pixel_index_map = torch.where(pixel_map > threshold, pixel_index_map, -1)
        pixel_index_map = pixel_index_map.type(torch.float64)

        component_map = pixel_index_map.apply_(
            lambda x: self.uf_0.component_entry_value(
                int(x), component_idx, birth_threshold
            )
        )

        if device == "cuda":
            component_map = component_map.cuda()

        return torch.where(component_map >= threshold, component_map, base_prob)

    def simple_analysis(self, num_components: int, fixed_threshold: float = None):
        intervals = self.important_components(
            dim=0, num_intervals=num_components, refined=True
        )

        intervals_refined = self.important_components(
            dim=0, num_intervals=num_components, refined=False
        )

        interval_and_thresholds = []

        if fixed_threshold is None:
            for interval in intervals:
                pass

    def threshold_analysis_dim0_components(
        self, num_components=0, num_bins=10, degree=3, minimal_threshold=0.0
    ):
        """
        params:
            num_components: int, number of components to analyze
            num_bins: int, number of bins to use for the histogram
            degree: int, degree of the polynomial to fit
            minimal_threshold: float, minimal threshold to consider
        returns:
            minima: array, minima of the fitted polynomial, candidates for threshold
        """
        if num_components == 0:
            return []

        interval_and_thresholds = []
        important_components = self.important_components(
            dim=0, num_intervals=num_components, refined=True
        )

        for interval in important_components:
            birht_index = interval[0]
            thresholds = self.merging_points_analysis(
                dim=0,
                interval_to_merge=birht_index,
                num_bins=num_bins,
                degree=degree,
                plot=False,
            )
            interval_and_thresholds.append(
                (
                    interval,
                    self.choose_threshold(
                        thresholds, interval, minimal_threshold=minimal_threshold
                    ),
                )
            )

        return interval_and_thresholds

    def choose_threshold(self, threshold_candidates, interval, minimal_threshold=0.0):

        birth_value = self.index_to_value(interval[0]).item()
        death_value = self.index_to_value(interval[1])
        if type(death_value) == torch.Tensor:
            death_value = death_value.item()

        minimal_threshold = max(minimal_threshold, death_value)

        threshold_candidates = threshold_candidates[threshold_candidates < birth_value]
        threshold_candidates = threshold_candidates[
            threshold_candidates > minimal_threshold
        ]
        if len(threshold_candidates) == 0:
            return minimal_threshold
        else:
            return sorted(threshold_candidates)[0]

    def merging_points_analysis(
        self, dim=0, interval_to_merge=0, num_bins=10, degree=3, plot=False
    ):
        """
        params:
            dim: int, dimension to analyze
            interval_to_merge: birth value of the interval that is of interest
            num_bins: int, number of bins to use for the histogram
            degree: int, degree of the polynomial to fit
            plot: bool, whether to plot the results
        returns:
            minima: array, minima of the fitted polynomial, candidates for threshold
        """

        num_intervals = len(self.intervals[dim])
        sums, ranges = self.merging_point_bins(
            dim=dim, interval_to_merge=interval_to_merge, num_bins=num_bins
        )

        if sums is None:
            return np.empty(0)

        normalized_sums = np.array(sums / num_intervals)
        middle_points = np.array(
            [(ranges[i] + ranges[i + 1]) / 2 for i in range(sums.shape[0])]
        )

        polynomial_coeff = np.polyfit(middle_points, normalized_sums, degree)
        polynomial = np.poly1d(polynomial_coeff)
        first_derivative = np.polyder(polynomial)
        second_derivative = np.polyder(first_derivative)

        critical_points = np.roots(first_derivative)
        second_derivative_values = second_derivative(critical_points)

        minima = critical_points[second_derivative_values > 0]

        if plot:
            w, h = matplotlib.figure.figaspect(1)
            fig = plt.figure(figsize=(w, h))
            plt.scatter(middle_points, polynomial(middle_points), color="r")
            plt.xlabel("Threshold")
            plt.ylabel("Density Function of Merging Points")
            plt.title("Density Function of Merging Points")
            plt.show()

            plt.scatter(middle_points, first_derivative(middle_points), color="r")
            plt.xlabel("Threshold")
            plt.ylabel("First Derivative Values")
            plt.title("First Derivative of Density Function of Merging Points")
            plt.show()
        return minima

    def plot_birth_values(self, dim=0, color="r", ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        fig = plt.figure(figsize=(w, h))

        birth_values = [
            self.index_to_value(interval[0]) for interval in self.intervals[dim]
        ]
        sorted_birth_values = sorted(birth_values, reverse=True)
        indices = list(range(len(sorted_birth_values)))
        plt.scatter(sorted_birth_values, indices, color=color)

        plt.xlabel("Interval Index")
        plt.ylabel("Birth Value")
        plt.title("Distribution of Birth Values")
        plt.show()

        return

    def birth_value_bins(self, dim=0, num_bins=10):
        birth_values = np.array(
            sorted(
                [self.index_to_value(interval[0]) for interval in self.intervals[dim]]
            )
        )
        return np.histogram(birth_values, bins=num_bins)

    def birth_value_bins_slope(self, dim=0, num_bins=10, degree=3):
        num_intervals = len(self.intervals[dim])
        sums, ranges = self.birth_value_bins(dim=dim, num_bins=num_bins)
        normalized_sums = np.array(sums / num_intervals)
        middle_points = np.array(
            [(ranges[i] + ranges[i + 1]) / 2 for i in range(sums.shape[0])]
        )

        polynomial_coeff = np.polyfit(middle_points, normalized_sums, degree)
        polynomial_approx = np.poly1d(polynomial_coeff)
        derivative = np.polyder(polynomial_approx)

        slope_at_middle_points = derivative(middle_points)

        roots = np.roots(derivative)

        second_derivative = np.polyder(derivative)
        second_derivative_values = second_derivative(roots)

        plt.scatter(middle_points, polynomial_approx(middle_points), color="r")
        plt.xlabel("Middle Points")
        plt.ylabel("num intervals")
        plt.title("Bins of Birth Values")
        plt.show()

        plt.scatter(middle_points, slope_at_middle_points, color="r")
        plt.xlabel("Middle Points")
        plt.ylabel("Slope")
        plt.title("Slope of Birth Values")
        plt.show()

        return

    def plot_birth_values_slope(self, dim=0, color="r", ratio=1):
        from scipy.signal import savgol_filter

        w, h = matplotlib.figure.figaspect(ratio)
        fig = plt.figure(figsize=(w, h))

        birth_values = np.array(
            sorted(
                [self.index_to_value(interval[0]) for interval in self.intervals[dim]],
                reverse=True,
            )
        )
        differences = np.diff(birth_values)

        plt.scatter(birth_values[1:], differences, color=color)
        plt.xlabel("Interval Index")
        plt.ylabel("Slope")
        plt.title("Smoothed Slope of Birth Values")
        plt.show()
        return

    def get_merging_points(self, dim=0, interval_to_merge=0):
        merging_points = []
        if dim == 0:
            uf = self.uf_0
        else:
            uf = self.uf_1

        for i in range(len(self.intervals[dim])):
            if i == interval_to_merge:
                continue

            interval = self.intervals[dim][i]

            if uf.find(interval[0]) == interval_to_merge:
                merging_points.append(self.index_to_value(interval[1]))

        return merging_points

    def plot_merging_points(
        self, dim=0, interval_to_merge=0, num_bins=10, color="r", ratio=1
    ):
        w, h = matplotlib.figure.figaspect(ratio)
        fig = plt.figure(figsize=(w, h))

        merging_points = sorted(
            self.get_merging_points(dim=dim, interval_to_merge=interval_to_merge)
        )
        num_bins = min(num_bins, len(merging_points))
        bins = np.linspace(
            min(0, min(merging_points)), max(1, max(merging_points)), num_bins
        )
        plt.hist(merging_points, bins=bins, color=color)
        plt.xlabel("Merging Points")
        plt.ylabel("Frequency")
        plt.title("Histogram of Merging Points")
        plt.show()
        return

    def get_non_merging_intervals(self, dim=0, interval_to_merge=0, num_gaps=10):
        merging_points = np.array(
            sorted(
                self.get_merging_points(dim=dim, interval_to_merge=interval_to_merge)
            )
        )

        gaps = np.diff(merging_points)
        arg_sorted_gaps = np.argsort(gaps)
        num_gaps = min(num_gaps, len(gaps))

        for i in arg_sorted_gaps[-num_gaps:]:
            print(
                f"gap: {gaps[i]} ranging from {merging_points[i]} to {merging_points[i+1]}"
            )
        return

    def plot_sum_active_intervals(self, dim=0, points=10, color="r", ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        fig = plt.figure(figsize=(w, h))
        intervals_birth = torch.tensor(
            [self.index_to_value(interval[0]) for interval in self.intervals[dim]]
        )
        intervals_death = torch.tensor(
            [self.index_to_value(interval[1]) for interval in self.intervals[dim]]
        )

        def get_sum_active_intervals(threshold):
            active_intervals = torch.where(
                intervals_birth > threshold, intervals_birth, 0
            )
            active_intervals = torch.where(
                intervals_death < threshold, active_intervals, 0
            )
            return torch.sum(active_intervals)

        points = torch.linspace(0, 1, points)
        values = [get_sum_active_intervals(point) for point in points]

        plt.plot(points.numpy(), values, color=color)
        plt.xlabel("Threshold")
        plt.ylabel("Sum of Active Intervals")
        plt.title("Sum of Active Intervals vs Threshold")
        plt.show()
        return

    def plot_component_at_threshold(
        self, dim=0, threshold=0.5, component_idx=0, color="r", ratio=1
    ):
        w, h = matplotlib.figure.figaspect(ratio)
        fig = plt.figure(figsize=(w, h))

        pixel_index_map = torch.from_numpy(self.pixel_index_map())
        pixel_map = (
            torch.from_numpy(self.PixelMap)
            if type(self.PixelMap) == np.ndarray
            else self.PixelMap
        )
        pixel_index_map = torch.where(pixel_map >= threshold, pixel_index_map, -1)

        assigned_component = pixel_index_map.apply_(
            lambda x: self.uf_0.find_assigned_comp_at_thr(x, threshold)
        )
        good_comp = torch.where(assigned_component == component_idx, 1, 0)

        plt.imshow(good_comp.numpy(), cmap="gray")
        plt.axis("off")
        plt.show()
        return

    def merging_point_bins(self, dim=0, interval_to_merge=0, num_bins=10):
        merging_points = np.array(
            sorted(
                self.get_merging_points(dim=dim, interval_to_merge=interval_to_merge)
            )
        )

        if merging_points.shape[0] == 0:
            # no other components get merged into the component of interest
            return None, None

        num_bins = min(num_bins, merging_points.shape[0])
        return np.histogram(merging_points, bins=num_bins)

    def Diagram(self, color="red", ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        fig, ax = plt.subplots(figsize=(w, h))
        intervals = [
            self.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.intervals[dim]
        ]
        if self.filtration == "sublevel":
            max_val = max(
                intervals, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val = min(intervals, key=lambda x: x[0])[0]
        else:
            max_val = max(intervals, key=lambda x: x[0])[0]
            min_val = min(
                intervals, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        x_min = min_val - (max_val - min_val) * 0.1
        x_max = max_val + (max_val - min_val) * 0.1
        for dim in range(2):
            for i, j in self.intervals[dim]:
                if j != np.infty:
                    if self.filtration == "sublevel":
                        plt.scatter(
                            self.index_to_value(i), self.index_to_value(j), color=color
                        )
                    else:
                        plt.scatter(
                            self.index_to_value(j), self.index_to_value(i), color=color
                        )
        plt.plot(
            (min(x_min, 0), max(x_max, 1)),
            (min(x_min, 0), max(x_max, 1)),
            color="k",
            linewidth=0.8,
        )
        plt.xlim(min(x_min, 0), max(x_max, 1))
        plt.ylim(min(x_min, 0), max(x_max, 1))
        return

    def get_generating_vertex(self, cube):
        boundary = [cube]
        while boundary != []:
            generating_boundary = max(boundary)
            boundary = self.get_boundary(generating_boundary)
        return generating_boundary

    def plot_intervals(self, dim=None):
        if dim == 0 or dim == None:
            x_births = []
            y_births = []
            x_deaths = []
            y_deaths = []
            for i, j in self.intervals[0]:
                if j != np.infty:
                    x, y = self.index_to_coordinates(i)
                    x_births.append(x / 2)
                    y_births.append(y / 2)
                    j_vert = self.get_generating_vertex(j)
                    x, y = self.index_to_coordinates(j_vert)
                    x_deaths.append(x / 2)
                    y_deaths.append(y / 2)
                else:
                    x, y = self.index_to_coordinates(i)
                    essential = (x / 2, y / 2)
            plt.figure(figsize=(8, 8))
            plt.imshow(self.PixelMap, cmap="gray")
            plt.scatter(essential[1], essential[0], 50, c="g", marker="*")
            plt.scatter(
                y_births, x_births, 50, c=list(range(len(x_births))), marker="*"
            )
            plt.scatter(
                y_deaths, x_deaths, 50, c=list(range(len(x_deaths))), marker="x"
            )
            plt.axis("off")
        if dim == 1 or dim == None:
            x_births = []
            y_births = []
            x_deaths = []
            y_deaths = []
            for i, j in self.intervals[1]:
                i_vert = self.get_generating_vertex(i)
                x, y = self.index_to_coordinates(i_vert)
                x_births.append(x / 2)
                y_births.append(y / 2)
                j_vert = self.get_generating_vertex(j)
                x, y = self.index_to_coordinates(j_vert)
                x_deaths.append(x / 2)
                y_deaths.append(y / 2)
            plt.figure(figsize=(8, 8))
            plt.imshow(self.PixelMap, cmap="gray")
            plt.scatter(
                y_births, x_births, 50, c=list(range(len(x_births))), marker="*"
            )
            plt.scatter(
                y_deaths, x_deaths, 50, c=list(range(len(x_deaths))), marker="x"
            )
            plt.axis("off")
        return

    def get_birth_dic_dim_0(self, intervals=6, threshold=0.5, app=False):
        if len(self.intervals[0]) == 0:
            raise ValueError("No intervals in dimension 0.")
        assert (type(intervals) == list and len(intervals) <= 6) or (
            0 <= intervals and intervals <= 6
        )

        birth_dic = {}
        death_vertices = []
        counter = 1

        if type(intervals) != list:
            num_intervals = min(intervals, len(self.intervals[0]))
            # intervals, intervals_values = self.get_largest_intervals(
            #     0, num_intervals, return_indices=True
            # )
            intervals = rd.sample(range(0, len(self.intervals[0])), num_intervals)

        if len(intervals) == 1:
            (i, j) = self.intervals[0][intervals[0]]
            birth_dic[i] = counter
            if j != np.infty:
                death_vertices.append(self.get_generating_vertex(j))
        else:
            for i, j in itemgetter(*intervals)(self.intervals[0]):
                birth_dic[i] = counter
                if j != np.infty:
                    death_vertices.append(self.get_generating_vertex(j))
                counter += 1

        if app == False:
            colormap = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a, b) = self.fine_to_coarse((i, j))
            if self.filtration == "sublevel" and a <= threshold and threshold < b:
                colormap = ListedColormap([(0, 1, 0, 1)])
            elif self.filtration == "superlevel" and a >= threshold and threshold > b:
                colormap = ListedColormap([(0, 1, 0, 1)])
            else:
                colormap = ListedColormap([(1, 0, 0, 1)])
        return birth_dic, death_vertices, colormap

    def get_CycleMap_dim_0(self, birth_dic, death_vertices, threshold=0.5, app=False):
        x_birth = []
        y_birth = []
        x_death = []
        y_death = []
        for cube in birth_dic.keys():
            x, y = self.index_to_coordinates(cube)
            if self.construction == "V":
                x_birth.append(int(x / 2))
                y_birth.append(int(y / 2))
            else:
                x_birth.append(x)
                y_birth.append(y)
        for cube in death_vertices:
            x, y = self.index_to_coordinates(cube)
            if self.construction == "V":
                x_death.append(int(x / 2))
                y_death.append(int(y / 2))
            else:
                x_death.append(x)
                y_death.append(y)
        births = (x_birth, y_birth)
        deaths = (x_death, y_death)

        UF = UnionFind(self.num_cubes)
        for edge in self.edges:
            if self.filtration == "sublevel":
                if self.index_to_value(edge) > threshold:
                    break
            else:
                if self.index_to_value(edge) < threshold:
                    break
            if app == True:
                if len(death_vertices) != 0 and death_vertices[0] < edge:
                    break
            boundary = self.get_boundary(edge)
            if (
                UF.get_birth(boundary[0]) in birth_dic.keys()
                and UF.get_birth(boundary[1]) in birth_dic.keys()
            ):
                continue
            UF.union(boundary[0], boundary[1])

        if self.construction == "V":
            CycleMap = np.zeros(self.PixelMap.shape)
        else:
            CycleMap = np.zeros(self.ValueMap.shape)

        for k in range(self.m):
            for l in range(self.n):
                vertex = self.IndexMap[2 * k, 2 * l]
                if vertex in birth_dic.keys():
                    if self.construction == "V":
                        CycleMap[k, l] = birth_dic[vertex]
                    else:
                        CycleMap[2 * k, 2 * l] = birth_dic[vertex]
                    continue
                birth = UF.get_birth(vertex)
                if birth in birth_dic.keys():
                    if self.construction == "V":
                        CycleMap[k, l] = birth_dic[birth]
                    else:
                        CycleMap[2 * k, 2 * l] = birth_dic[birth]

        if self.construction == "T":
            for k in range(0, self.M, 2):
                for l in range(1, self.N, 2):
                    if CycleMap[k, l - 1] == CycleMap[k, l + 1]:
                        CycleMap[k, l] = CycleMap[k, l - 1]
            for k in range(1, self.M, 2):
                for l in range(0, self.N, 2):
                    if CycleMap[k - 1, l] == CycleMap[k + 1, l]:
                        CycleMap[k, l] = CycleMap[k - 1, l]
            for k in range(1, self.M, 2):
                for l in range(1, self.N, 2):
                    if (
                        len(
                            set(
                                [
                                    CycleMap[k - 1, l - 1],
                                    CycleMap[k - 1, l + 1],
                                    CycleMap[k + 1, l - 1],
                                    CycleMap[k + 1, l + 1],
                                ]
                            )
                        )
                        == 1
                    ):
                        CycleMap[k, l] = CycleMap[k - 1, l - 1]
        return CycleMap, births, deaths

    def get_birth_dic_dim_1(self, intervals=6, threshold=0.5, app=False):
        if len(self.intervals[1]) == 0:
            raise ValueError("No intervals in dimension 1.")
        assert (type(intervals) == list and len(intervals) <= 6) or (
            0 <= intervals and intervals <= 6
        )

        birth_dic = {}
        death_dic = {}
        counter = 1

        if type(intervals) != list:
            num_intervals = min(intervals, len(self.intervals[1]))
            intervals, intervals_values = self.get_largest_intervals(
                1, num_intervals, return_indices=True
            )
            # intervals = rd.sample(range(0,len(self.intervals[1])), num_intervals)

        if len(intervals) == 1:
            (i, j) = self.intervals[1][intervals[0]]
            birth_dic[j] = counter
            death_dic[i] = j
        else:
            for i, j in itemgetter(*intervals)(self.intervals[1]):
                birth_dic[j] = counter
                death_dic[i] = j
                counter += 1

        if app == False:
            colormap = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a, b) = self.fine_to_coarse((i, j))
            if self.filtration == "sublevel" and a <= threshold and threshold < b:
                colormap = ListedColormap([(0, 1, 0, 1)])
            elif self.filtration == "superlevel" and a >= threshold and threshold > b:
                colormap = ListedColormap([(0, 1, 0, 1)])
            else:
                colormap = ListedColormap([(1, 0, 0, 1)])
        return birth_dic, death_dic, colormap

    def get_bounding_cycle(self, component):
        cycle_vertices = {}
        for cube in component:
            edges = self.get_boundary(cube)
            for edge in edges:
                vertices = self.get_boundary(edge)
                for vertex in vertices:
                    if vertex in cycle_vertices.keys():
                        cycle_vertices[vertex] += 1
                    else:
                        cycle_vertices[vertex] = 1
        cycle = [
            vertex
            for vertex in cycle_vertices.keys()
            if cycle_vertices[vertex] % 8 != 0
        ]
        return cycle

    def get_CycleMap_dim_1(self, birth_dic, death_dic):
        x_birth = []
        y_birth = []
        x_death = []
        y_death = []
        for square in birth_dic.keys():
            vertex = self.get_generating_vertex(square)
            x, y = self.index_to_coordinates(vertex)
            if self.construction == "V":
                x_death.append(int(x / 2))
                y_death.append(int(y / 2))
            else:
                x_death.append(x)
                y_death.append(y)
        for edge in death_dic.keys():
            vertex = self.get_generating_vertex(edge)
            x, y = self.index_to_coordinates(vertex)
            if self.construction == "V":
                x_birth.append(int(x / 2))
                y_birth.append(int(y / 2))
            else:
                x_birth.append(x)
                y_birth.append(y)
        births = (x_birth, y_birth)
        deaths = (x_death, y_death)
        CycleMap = np.zeros(self.PixelMap.shape)
        UF = UnionFind(self.num_cubes + 1, dual=True)
        for edge in self.edges[::-1]:
            if edge in death_dic.keys():
                component = UF.get_component(death_dic[edge])
                cycle = self.get_bounding_cycle(component)
                for vertex in cycle:
                    x, y = self.index_to_coordinates(vertex)
                    CycleMap[int(x / 2), int(y / 2)] = birth_dic[death_dic[edge]]
            boundary = self.get_dual_boundary(edge)
            if (
                UF.get_birth(boundary[0]) in birth_dic.keys()
                and UF.get_birth(boundary[1]) in birth_dic.keys()
            ):
                continue
            UF.union(boundary[0], boundary[1])
        return CycleMap, births, deaths

    def components_map(self, intervals: int, threshold: float):
        pass

    def plot_representative_cycles(
        self, dim=1, intervals=6, threshold=0.5, plot_birth_and_death=False, app=False
    ):
        assert dim in [0, 1]

        if dim == 0:
            birth_dic, death_vertices, colormap = self.get_birth_dic_dim_0(
                intervals=intervals, threshold=threshold, app=app
            )
            CycleMap, births, deaths = self.get_CycleMap_dim_0(
                birth_dic, death_vertices, threshold=threshold, app=app
            )

        elif dim == 1:
            birth_dic, death_dic, colormap = self.get_birth_dic_dim_1(
                intervals=intervals, threshold=threshold, app=app
            )
            CycleMap, births, deaths = self.get_CycleMap_dim_1(birth_dic, death_dic)

        if app == False:
            CycleMap_masked = np.ma.masked_where(CycleMap == 0, CycleMap)
        else:
            if self.filtration == "sublevel":
                CycleMap_masked = np.ma.masked_where(
                    (CycleMap == 0) | (self.PixelMap > threshold), CycleMap
                )
            else:
                CycleMap_masked = np.ma.masked_where(
                    (CycleMap == 0) | (self.PixelMap < threshold), CycleMap
                )

        fig = plt.figure(figsize=(6, 6))
        if app == False:
            if self.construction == "V":
                plt.imshow(self.PixelMap, cmap="gray")
            else:
                plt.imshow(self.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(self.PixelMap <= threshold, self.PixelMap, 0),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(self.PixelMap >= threshold, self.PixelMap, 0),
                        cmap="gray",
                    )
            else:
                plt.imshow(self.ValueMap, cmap="gray")
        plt.imshow(CycleMap_masked, cmap=colormap, interpolation=None)
        if plot_birth_and_death:
            plt.scatter(births[1], births[0], 300, c="g", marker="*")
            plt.scatter(deaths[1], deaths[0], 300, c="r", marker="x")
        plt.axis("off")
        plt.close(fig)
        return fig

    def plot_representative_cycles_app(self, dim=1, plot_birth_and_death=False):
        assert dim in [0, 1]

        def plot(interval, threshold):
            return self.plot_representative_cycles(
                dim=dim,
                intervals=[interval],
                threshold=threshold,
                plot_birth_and_death=plot_birth_and_death,
                app=True,
            )

        return interact(
            plot,
            interval=widgets.IntSlider(
                start=0, end=len(self.intervals[dim]) - 1, value=0, name="interval"
            ),
            threshold=widgets.FloatSlider(
                start=0.0,
                end=1.0,
                step=0.001,
                value=0.5,
                value_throttled=True,
                name="threshold",
            ),
        )


class ImagePersistence:
    def __init__(
        self,
        CubicalPersistence_0,
        CubicalPersistence_1,
        valid="all",
        use_UnionFind=True,
    ):
        self.CP_0 = CubicalPersistence_0
        self.CP_1 = CubicalPersistence_1
        assert self.CP_0.m == self.CP_1.m and self.CP_0.n == self.CP_1.n
        assert self.CP_0.reduced == self.CP_1.reduced
        self.reduced = self.CP_0.reduced
        assert self.CP_0.filtration == self.CP_1.filtration
        self.filtration = self.CP_0.filtration
        assert self.CP_0.construction == self.CP_1.construction
        assert valid in ["all", "nonnegative", "positive"]

        self.intervals = [[], []]
        if use_UnionFind:
            assert self.CP_0.get_critical_edges
            self.compute_persistence_UF(valid=valid)
        else:
            assert self.CP_1.get_image_columns_to_reduce
            self.B = BoundaryMatrix()
            self.set_BoundaryMatrix()
            self.compute_persistence(valid=valid)

    def set_BoundaryMatrix(self):
        self.B.columns_to_reduce = self.CP_1.columns_to_reduce
        for idx_col in self.B.columns_to_reduce[1]:
            i, j = self.CP_1.index_to_coordinates(idx_col)
            if i % 2 == 0:
                idx_row = self.CP_0.IndexMap[i, j + 1]
                self.B.set_one(idx_row, idx_col)
                idx_row = self.CP_0.IndexMap[i, j - 1]
                self.B.set_one(idx_row, idx_col)
            else:
                idx_row = self.CP_0.IndexMap[i + 1, j]
                self.B.set_one(idx_row, idx_col)
                idx_row = self.CP_0.IndexMap[i - 1, j]
                self.B.set_one(idx_row, idx_col)
        for idx_col in self.B.columns_to_reduce[2]:
            i, j = self.CP_1.index_to_coordinates(idx_col)
            idx_row = self.CP_0.IndexMap[i, j + 1]
            self.B.set_one(idx_row, idx_col)
            idx_row = self.CP_0.IndexMap[i, j - 1]
            self.B.set_one(idx_row, idx_col)
            idx_row = self.CP_0.IndexMap[i + 1, j]
            self.B.set_one(idx_row, idx_col)
            idx_row = self.CP_0.IndexMap[i - 1, j]
            self.B.set_one(idx_row, idx_col)
        return

    def fine_to_coarse(self, interval):
        return (
            self.CP_0.index_to_value(interval[0]),
            self.CP_1.index_to_value(interval[1]),
        )

    def valid_interval(self, interval, valid="all"):
        if valid == "all":
            return True

        elif valid == "nonnegative":
            if self.CP_0.filtration == "sublevel":
                return self.CP_0.index_to_value(
                    interval[0]
                ) <= self.CP_1.index_to_value(interval[1])

            else:
                return self.CP_0.index_to_value(
                    interval[0]
                ) >= self.CP_1.index_to_value(interval[1])

        else:
            if self.CP_0.filtration == "sublevel":
                return self.CP_0.index_to_value(interval[0]) < self.CP_1.index_to_value(
                    interval[1]
                )

            else:
                return self.CP_0.index_to_value(interval[0]) > self.CP_1.index_to_value(
                    interval[1]
                )

    def compute_persistence(self, valid="all"):
        self.B.reduce(clearing=False)
        pairings = self.B.get_Pairings()
        if self.reduced == False:
            self.intervals[0] = [(0, np.infty)]
        else:
            self.intervals[0] = []
        for i, j in pairings:
            if self.valid_interval((i, j), valid=valid):
                self.intervals[self.CP_0.index_to_dim(i)].append((i, j))
        return

    def get_boundary(self, idx):
        boundary = []
        x, y = self.CP_1.index_to_coordinates(idx)
        if x % 2 != 0:
            boundary.extend(
                [self.CP_0.IndexMap[x - 1, y], self.CP_0.IndexMap[x + 1, y]]
            )
        if y % 2 != 0:
            boundary.extend(
                [self.CP_0.IndexMap[x, y - 1], self.CP_0.IndexMap[x, y + 1]]
            )
        return boundary

    def get_dual_boundary(self, idx):
        boundary = []
        x, y = self.CP_0.index_to_coordinates(idx)
        if x % 2 == 0:
            if x == 0:
                boundary.extend([self.CP_1.num_cubes, self.CP_1.IndexMap[x + 1, y]])
            elif x == self.CP_0.M - 1:
                boundary.extend([self.CP_1.num_cubes, self.CP_1.IndexMap[x - 1, y]])
            else:
                boundary.extend(
                    [self.CP_1.IndexMap[x - 1, y], self.CP_1.IndexMap[x + 1, y]]
                )
        if y % 2 == 0:
            if y == 0:
                boundary.extend([self.CP_1.num_cubes, self.CP_1.IndexMap[x, y + 1]])
            elif y == self.CP_0.N - 1:
                boundary.extend([self.CP_1.num_cubes, self.CP_1.IndexMap[x, y - 1]])
            else:
                boundary.extend(
                    [self.CP_1.IndexMap[x, y - 1], self.CP_1.IndexMap[x, y + 1]]
                )
        return boundary

    def compute_dim0(self, valid="all"):
        self.intervals[0] = [(0, np.infty)]
        UF = UnionFind(self.CP_0.num_cubes, dual=False)
        for edge in self.CP_1.columns_to_reduce[1]:
            boundary = self.get_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])
            if x == y:
                continue
            birth = max(UF.get_birth(x), UF.get_birth(y))
            if self.valid_interval((birth, edge), valid=valid):
                self.intervals[0].append((birth, edge))
            UF.union(x, y)
        return

    def compute_dim1(self, valid="all"):
        UF = UnionFind(self.CP_1.num_cubes + 1, dual=True)
        for edge in self.CP_0.critical_edges:
            boundary = self.get_dual_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])
            if x == y:
                continue
            birth = min(UF.get_birth(x), UF.get_birth(y))
            if self.valid_interval((edge, birth), valid=valid):
                self.intervals[1].append((edge, birth))
            UF.union(x, y)
        return

    def compute_persistence_UF(self, valid="all"):
        self.compute_dim0(valid=valid)
        self.compute_dim1(valid=valid)
        return


class InducedMatching:
    def __init__(self, ImagePersistence):
        self.IP = ImagePersistence
        self.matched = [[], []]
        self.unmatched_0 = copy.deepcopy(self.IP.CP_0.intervals)
        self.unmatched_1 = copy.deepcopy(self.IP.CP_1.intervals)
        self.match()

    def find_match(self, interval, dim):
        match_0 = None
        match_1 = None
        for a, b in self.unmatched_0[dim]:
            if a == interval[0]:
                match_0 = (a, b)
                break
        if match_0 == None:
            return None

        for a, b in self.unmatched_1[dim]:
            if b == interval[1]:
                match_1 = (a, b)
                break
        if match_1 == None:
            return None

        else:
            return (match_0, interval, match_1)

    def match(self):
        for dim in range(2):
            for a, b in self.IP.intervals[dim]:
                match = self.find_match((a, b), dim)
                if match == None:
                    continue
                else:
                    self.matched[dim].append(match)
                    self.unmatched_0[dim].remove(match[0])
                    self.unmatched_1[dim].remove(match[2])

    def get_matching(self):
        matched = [
            [
                (
                    self.IP.CP_0.fine_to_coarse(match[0]),
                    self.IP.CP_1.fine_to_coarse(match[2]),
                )
                for match in self.matched[dim]
            ]
            for dim in range(2)
        ]
        unmatched_0 = [
            [
                self.IP.CP_0.fine_to_coarse(interval)
                for interval in self.unmatched_0[dim]
            ]
            for dim in range(2)
        ]
        unmatched_1 = [
            [
                self.IP.CP_1.fine_to_coarse(interval)
                for interval in self.unmatched_1[dim]
            ]
            for dim in range(2)
        ]
        return matched, unmatched_0, unmatched_1

    def BarCode(self, plot_image=False, colors=["r", "g", "grey"], ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        fig, ax = plt.subplots(figsize=(w, h))
        intervals_0 = [
            self.IP.CP_0.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.IP.CP_0.intervals[dim]
        ]
        intervals_1 = [
            self.IP.CP_1.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.IP.CP_1.intervals[dim]
        ]
        if self.IP.CP_0.filtration == "sublevel":
            max_val_0 = max(
                intervals_0, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_0 = min(intervals_0, key=lambda x: x[0])[0]
        else:
            max_val_0 = max(intervals_0, key=lambda x: x[0])[0]
            min_val_0 = min(
                intervals_0, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        if self.IP.CP_1.filtration == "sublevel":
            max_val_1 = max(
                intervals_1, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_1 = min(intervals_1, key=lambda x: x[0])[0]
        else:
            max_val_1 = max(intervals_1, key=lambda x: x[0])[0]
            min_val_1 = min(
                intervals_1, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        max_val = max(max_val_0, max_val_1)
        min_val = min(min_val_0, min_val_1)
        x_min = min_val - (max_val - min_val) * 0.1
        x_max = max_val + (max_val - min_val) * 0.1
        for dim in range(2):
            num_intervals = (
                len(self.unmatched_0[dim])
                + len(self.unmatched_1[dim])
                + len(self.matched[dim])
            )
            alpha = 1 / 4
            delta = 1 / (num_intervals + 1)
            height = dim + delta
            for i, j in self.unmatched_0[dim]:
                if j == np.infty:
                    if self.IP.filtration == "sublevel":
                        ax.plot(
                            (self.IP.CP_0.index_to_value(i), x_max),
                            (height, height),
                            color=colors[0],
                        )
                    else:
                        ax.plot(
                            (self.IP.CP_0.index_to_value(i), x_min),
                            (height, height),
                            color=colors[0],
                        )
                else:
                    ax.plot(
                        self.IP.CP_0.fine_to_coarse((i, j)),
                        (height, height),
                        color=colors[0],
                    )
                height += delta
            for i, j in self.unmatched_1[dim]:
                if j == np.infty:
                    if self.IP.filtration == "sublevel":
                        ax.plot(
                            (self.IP.CP_1.index_to_value(i), x_max),
                            (height, height),
                            color=colors[1],
                        )
                    else:
                        ax.plot(
                            (self.IP.CP_1.index_to_value(i), x_min),
                            (height, height),
                            color=colors[1],
                        )
                else:
                    ax.plot(
                        self.IP.CP_1.fine_to_coarse((i, j)),
                        (height, height),
                        color=colors[1],
                    )
                height += delta
            for (i_0, j_0), (i_im, j_im), (i_1, j_1) in self.matched[dim]:
                if j_0 == np.infty:
                    if self.IP.filtration == "sublevel":
                        ax.plot(
                            (self.IP.CP_0.index_to_value(i_0), x_max),
                            (height - delta * alpha, height - delta * alpha),
                            color=colors[0],
                        )
                    else:
                        ax.plot(
                            (self.IP.CP_0.index_to_value(i_0), x_min),
                            (height - delta * alpha, height - delta * alpha),
                            color=colors[0],
                        )
                else:
                    ax.plot(
                        self.IP.CP_0.fine_to_coarse((i_0, j_0)),
                        (height - delta * alpha, height - delta * alpha),
                        color=colors[0],
                    )
                if plot_image == True:
                    if j_im == np.infty:
                        if self.IP.filtration == "sublevel":
                            ax.plot(
                                (self.IP.CP_0.index_to_value(i_im), x_max),
                                (height, height),
                                color=colors[2],
                            )
                            ax.fill_between(
                                (self.IP.CP_0.index_to_value(i_im), x_max),
                                (height - delta * alpha, height - delta * alpha),
                                (height + delta * alpha, height + delta * alpha),
                                color="grey",
                                alpha=0.3,
                            )
                        else:
                            ax.plot(
                                (self.IP.CP_0.index_to_value(i_im), x_min),
                                (height, height),
                                color=colors[2],
                            )
                            ax.fill_between(
                                ((self.IP.CP_0.index_to_value(i_im), x_min)),
                                (height - delta * alpha, height - delta * alpha),
                                (height + delta * alpha, height + delta * alpha),
                                color="grey",
                                alpha=0.3,
                            )
                    else:
                        ax.plot(
                            self.IP.fine_to_coarse((i_im, j_im)),
                            (height, height),
                            color=colors[2],
                        )
                        ax.fill_between(
                            self.IP.fine_to_coarse((i_im, j_im)),
                            (height - delta * alpha, height - delta * alpha),
                            (height + delta * alpha, height + delta * alpha),
                            color="grey",
                            alpha=0.3,
                        )
                if j_1 == np.infty:
                    if self.IP.filtration == "sublevel":
                        ax.plot(
                            (self.IP.CP_1.index_to_value(i_1), x_max),
                            (height + delta * alpha, height + delta * alpha),
                            color=colors[1],
                        )
                    else:
                        ax.plot(
                            (self.IP.CP_1.index_to_value(i_1), x_min),
                            (height + delta * alpha, height + delta * alpha),
                            color=colors[1],
                        )
                else:
                    ax.plot(
                        self.IP.CP_1.fine_to_coarse((i_1, j_1)),
                        (height + delta * alpha, height + delta * alpha),
                        color=colors[1],
                    )
                height += delta
        plt.plot((x_min, x_max), (1, 1), color="k", linewidth=0.8)
        plt.ylabel("Dimension")
        plt.xlim(x_min, x_max)
        plt.ylim(0, 2)
        plt.yticks([0.5, 1.5], [0, 1])
        return


class BettiMatching:
    def __init__(
        self,
        Picture_0,
        Picture_1,
        relative=False,
        reduced=False,
        filtration="sublevel",
        construction="V",
        comparison="union",
        valid="positive",
        valid_image="all",
        use_UnionFind_for_image=True,
        training=False,
    ):
        assert valid in ["all", "nonnegative", "positive"]
        assert valid_image in ["all", "nonnegative", "positive"]
        assert filtration in ["sublevel", "superlevel"]
        self.filtration = filtration
        assert construction in ["V", "T"]
        self.construction = construction
        assert comparison in ["union", "intersection"]
        self.comparison = comparison

        if comparison == "union":
            if filtration == "sublevel":
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.minimum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.minimum(Picture_0, Picture_1)
            else:
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.maximum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.maximum(Picture_0, Picture_1)
            self.CP_0 = CubicalPersistence(
                Picture_0,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                get_critical_edges=use_UnionFind_for_image,
                training=training,
            )
            self.CP_1 = CubicalPersistence(
                Picture_1,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                get_critical_edges=use_UnionFind_for_image,
                training=training,
            )
            self.CP_comp = CubicalPersistence(
                Picture_comp,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                get_image_columns_to_reduce=not use_UnionFind_for_image,
                training=training,
            )
            self.IP_0 = ImagePersistence(
                self.CP_0,
                self.CP_comp,
                valid=valid_image,
                use_UnionFind=use_UnionFind_for_image,
            )
            self.IP_1 = ImagePersistence(
                self.CP_1,
                self.CP_comp,
                valid=valid_image,
                use_UnionFind=use_UnionFind_for_image,
            )
        else:
            if filtration == "sublevel":
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.maximum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.maximum(Picture_0, Picture_1)
            else:
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.minimum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.minimum(Picture_0, Picture_1)
            self.CP_comp = CubicalPersistence(
                Picture_comp,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                get_critical_edges=use_UnionFind_for_image,
                training=training,
            )
            self.CP_0 = CubicalPersistence(
                Picture_0,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                get_image_columns_to_reduce=not use_UnionFind_for_image,
                training=training,
            )
            self.CP_1 = CubicalPersistence(
                Picture_1,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                get_image_columns_to_reduce=not use_UnionFind_for_image,
                training=training,
            )
            self.IP_0 = ImagePersistence(
                self.CP_comp,
                self.CP_0,
                valid=valid_image,
                use_UnionFind=use_UnionFind_for_image,
            )
            self.IP_1 = ImagePersistence(
                self.CP_comp,
                self.CP_1,
                valid=valid_image,
                use_UnionFind=use_UnionFind_for_image,
            )
        self.IM_0 = InducedMatching(self.IP_0)
        self.IM_1 = InducedMatching(self.IP_1)
        self.matched = [[], []]
        self.unmatched_0 = copy.deepcopy(self.CP_0.intervals)
        self.unmatched_comp = copy.deepcopy(self.CP_comp.intervals)
        self.unmatched_1 = copy.deepcopy(self.CP_1.intervals)
        self.match()

    def match(self):
        matched_1 = copy.deepcopy(self.IM_1.matched)
        if self.comparison == "union":
            for dim in range(2):
                for match_0 in self.IM_0.matched[dim]:
                    for match_1 in matched_1[dim]:
                        if match_0[2] == match_1[2]:
                            self.matched[dim].append(
                                (match_0[0], match_0[2], match_1[0])
                            )
                            self.unmatched_0[dim].remove(match_0[0])
                            self.unmatched_comp[dim].remove(match_0[2])
                            self.unmatched_1[dim].remove(match_1[0])
                            matched_1[dim].remove(match_1)
                            break
        else:
            for dim in range(2):
                for match_0 in self.IM_0.matched[dim]:
                    for match_1 in matched_1[dim]:
                        if match_0[0] == match_1[0]:
                            self.matched[dim].append(
                                (match_0[2], match_0[0], match_1[2])
                            )
                            self.unmatched_0[dim].remove(match_0[2])
                            self.unmatched_comp[dim].remove(match_0[0])
                            self.unmatched_1[dim].remove(match_1[2])
                            matched_1[dim].remove(match_1)
                            break
        return

    def get_matching(self, refined=False):
        if refined:
            return (
                copy.deepcopy(self.matched),
                copy.deepcopy(self.unmatched_0),
                copy.deepcopy(self.unmatched_1),
            )

        matched = [
            [
                (self.CP_0.fine_to_coarse(match[0]), self.CP_1.fine_to_coarse(match[2]))
                for match in self.matched[dim]
            ]
            for dim in range(2)
        ]
        unmatched_0 = [
            [self.CP_0.fine_to_coarse(interval) for interval in self.unmatched_0[dim]]
            for dim in range(2)
        ]
        unmatched_1 = [
            [self.CP_1.fine_to_coarse(interval) for interval in self.unmatched_1[dim]]
            for dim in range(2)
        ]
        return matched, unmatched_0, unmatched_1

    def loss(self, dimensions=[0, 1]):
        loss = 0
        for dim in dimensions:
            for I_0, I_comp, I_1 in self.matched[dim]:
                (a_0, b_0) = self.CP_0.fine_to_coarse(I_0)
                if b_0 == np.infty:
                    b_0 = 1
                elif b_0 == -np.infty:
                    b_0 = 0
                (a_1, b_1) = self.CP_1.fine_to_coarse(I_1)
                if b_1 == np.infty:
                    b_1 = 1
                elif b_1 == -np.infty:
                    b_1 = 0
                loss += 2 * ((a_0 - a_1) ** 2 + (b_0 - b_1) ** 2)
            for I in self.unmatched_0[dim]:
                (a, b) = self.CP_0.fine_to_coarse(I)
                if b == np.infty:
                    b = 1
                elif b == -np.infty:
                    b = 0
                loss += (a - b) ** 2
            for I in self.unmatched_1[dim]:
                (a, b) = self.CP_1.fine_to_coarse(I)
                if b == np.infty:
                    b = 1
                elif b == -np.infty:
                    b = 0
                loss += (a - b) ** 2
        return loss

    def Betti_number_error(self, threshold=0.5, dimensions=[0, 1]):
        betti_0 = self.CP_0.get_Betti_numbers(threshold=threshold)
        betti_1 = self.CP_1.get_Betti_numbers(threshold=threshold)
        betti_err = 0
        for dim in dimensions:
            betti_err += np.abs(betti_0[dim] - betti_1[dim])
        return betti_err

    def plot_images(self, plot_comparison=False):
        rows = 1
        if plot_comparison:
            fig = plt.figure(figsize=(15, 5))
            columns = 3
        else:
            fig = plt.figure(figsize=(10, 5))
            columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.CP_0.PixelMap, cmap="gray")
        plt.axis("off")
        if plot_comparison:
            fig.add_subplot(rows, columns, 2)
            plt.imshow(self.CP_comp.PixelMap, cmap="gray")
            plt.axis("off")
            fig.add_subplot(rows, columns, 3)
        else:
            fig.add_subplot(rows, columns, 2)
        plt.imshow(self.CP_1.PixelMap, cmap="gray")
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)
        plt.close(fig)
        return fig

    def BarCode(self, plot_comparison=False, colors=["r", "b", "g"], ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        plt.figure(figsize=(w, h))
        intervals_0 = [
            self.CP_0.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.CP_0.intervals[dim]
        ]
        intervals_1 = [
            self.CP_1.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.CP_1.intervals[dim]
        ]
        if self.filtration == "sublevel":
            max_val_0 = max(
                intervals_0, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_0 = min(intervals_0, key=lambda x: x[0])[0]
        else:
            max_val_0 = max(intervals_0, key=lambda x: x[0])[0]
            min_val_0 = min(
                intervals_0, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        if self.filtration == "sublevel":
            max_val_1 = max(
                intervals_1, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_1 = min(intervals_1, key=lambda x: x[0])[0]
        else:
            max_val_1 = max(intervals_1, key=lambda x: x[0])[0]
            min_val_1 = min(
                intervals_1, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        max_val = max(max_val_0, max_val_1)
        min_val = min(min_val_0, min_val_1)
        x_min = min_val - (max_val - min_val) * 0.1
        x_max = max_val + (max_val - min_val) * 0.1
        for dim in range(2):
            num_intervals = (
                len(self.unmatched_0[dim])
                + len(self.unmatched_1[dim])
                + len(self.matched[dim])
            )
            alpha = 1 / 4
            delta = 1 / (num_intervals + 1)
            height = dim + delta
            for i, j in self.unmatched_0[dim]:
                if j == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_0.index_to_value(i), x_max),
                            (height, height),
                            color=colors[0],
                        )
                    else:
                        plt.plot(
                            (self.CP_0.index_to_value(i), x_min),
                            (height, height),
                            color=colors[0],
                        )
                else:
                    plt.plot(
                        self.CP_0.fine_to_coarse((i, j)),
                        (height, height),
                        color=colors[0],
                    )
                height += delta
            for i, j in self.unmatched_1[dim]:
                if j == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_1.index_to_value(i), x_max),
                            (height, height),
                            color=colors[1],
                        )
                    else:
                        plt.plot(
                            (self.CP_1.index_to_value(i), x_min),
                            (height, height),
                            color=colors[1],
                        )
                else:
                    plt.plot(
                        self.CP_1.fine_to_coarse((i, j)),
                        (height, height),
                        color=colors[1],
                    )
                height += delta
            for (i_0, j_0), (i_comp, j_comp), (i_1, j_1) in self.matched[dim]:
                if j_0 == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_0.index_to_value(i_0), x_max),
                            (height - delta * alpha, height - delta * alpha),
                            color=colors[0],
                        )
                    else:
                        plt.plot(
                            (self.CP_0.index_to_value(i_0), x_min),
                            (height - delta * alpha, height - delta * alpha),
                            color=colors[0],
                        )
                else:
                    plt.plot(
                        self.CP_0.fine_to_coarse((i_0, j_0)),
                        (height - delta * alpha, height - delta * alpha),
                        color=colors[0],
                    )
                if plot_comparison == True:
                    if j_comp == np.infty:
                        if self.filtration == "sublevel":
                            plt.plot(
                                (self.CP_comp.index_to_value(i_comp), x_max),
                                (height, height),
                                color=colors[2],
                            )
                        else:
                            plt.plot(
                                (self.CP_comp.index_to_value(i_comp), x_min),
                                (height, height),
                                color=colors[2],
                            )
                    else:
                        plt.plot(
                            self.CP_comp.fine_to_coarse((i_comp, j_comp)),
                            (height, height),
                            color=colors[2],
                        )
                if j_1 == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_1.index_to_value(i_1), x_max),
                            (height + delta * alpha, height + delta * alpha),
                            color=colors[1],
                        )
                    else:
                        plt.plot(
                            (self.CP_1.index_to_value(i_1), x_min),
                            (height + delta * alpha, height + delta * alpha),
                            color=colors[1],
                        )
                else:
                    plt.plot(
                        self.CP_1.fine_to_coarse((i_1, j_1)),
                        (height + delta * alpha, height + delta * alpha),
                        color=colors[1],
                    )
                height += delta
        plt.plot((x_min, x_max), (1, 1), color="k", linewidth=0.8)
        plt.ylabel("Dimension")
        plt.xlim(x_min, x_max)
        plt.ylim(0, 2)
        plt.yticks([0.5, 1.5], [0, 1])
        return

    def Diagram(self, plot_comparison=False, colors=["r", "b", "g"], ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        plt.figure(figsize=(w, h))
        intervals_0 = [
            self.CP_0.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.CP_0.intervals[dim]
        ]
        intervals_1 = [
            self.CP_1.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.CP_1.intervals[dim]
        ]
        if self.filtration == "sublevel":
            max_val_0 = max(
                intervals_0, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_0 = min(intervals_0, key=lambda x: x[0])[0]
        else:
            max_val_0 = max(intervals_0, key=lambda x: x[0])[0]
            min_val_0 = min(
                intervals_0, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        if self.filtration == "sublevel":
            max_val_1 = max(
                intervals_1, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_1 = min(intervals_1, key=lambda x: x[0])[0]
        else:
            max_val_1 = max(intervals_1, key=lambda x: x[0])[0]
            min_val_1 = min(
                intervals_1, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        max_val = max(max_val_0, max_val_1)
        min_val = min(min_val_0, min_val_1)
        x_min = min_val - (max_val - min_val) * 0.1
        x_max = max_val + (max_val - min_val) * 0.1
        for dim in range(2):
            for i, j in self.unmatched_0[dim]:
                if j != np.infty:
                    if self.filtration == "sublevel":
                        plt.scatter(
                            self.CP_0.index_to_value(i),
                            self.CP_0.index_to_value(j),
                            color=colors[0],
                        )
                    else:
                        plt.scatter(
                            self.CP_0.index_to_value(j),
                            self.CP_0.index_to_value(i),
                            color=colors[0],
                        )
                    diag = (
                        self.CP_0.index_to_value(i) + self.CP_0.index_to_value(j)
                    ) / 2
                    plt.plot(
                        (self.CP_0.index_to_value(i), diag),
                        (self.CP_0.index_to_value(j), diag),
                        color="k",
                    )
            for i, j in self.unmatched_1[dim]:
                if j != np.infty:
                    if self.filtration == "sublevel":
                        plt.scatter(
                            self.CP_1.index_to_value(i),
                            self.CP_1.index_to_value(j),
                            color=colors[1],
                        )
                    else:
                        plt.scatter(
                            self.CP_1.index_to_value(j),
                            self.CP_1.index_to_value(i),
                            color=colors[1],
                        )
                    diag = (
                        self.CP_1.index_to_value(i) + self.CP_1.index_to_value(j)
                    ) / 2
                    plt.plot(
                        (self.CP_1.index_to_value(i), diag),
                        (self.CP_1.index_to_value(j), diag),
                        color="k",
                    )
            for (i_0, j_0), (i_comp, j_comp), (i_1, j_1) in self.matched[dim]:
                if j_0 != np.infty:
                    if self.filtration == "sublevel":
                        plt.scatter(
                            self.CP_0.index_to_value(i_0),
                            self.CP_0.index_to_value(j_0),
                            color=colors[0],
                        )
                    else:
                        plt.scatter(
                            self.CP_0.index_to_value(j_0),
                            self.CP_0.index_to_value(i_0),
                            color=colors[0],
                        )
                if plot_comparison == True:
                    if j_comp != np.infty:
                        if self.filtration == "sublevel":
                            plt.scatter(
                                self.CP_comp.index_to_value(i_comp),
                                self.CP_comp.index_to_value(j_comp),
                                color=colors[2],
                            )
                    else:
                        plt.scatter(
                            self.CP_comp.index_to_value(j_comp),
                            self.CP_comp.index_to_value(i_comp),
                            color=colors[2],
                        )
                if j_1 != np.infty:
                    if self.filtration == "sublevel":
                        plt.scatter(
                            self.CP_1.index_to_value(i_1),
                            self.CP_1.index_to_value(j_1),
                            color=colors[1],
                        )
                    else:
                        plt.scatter(
                            self.CP_1.index_to_value(j_1),
                            self.CP_1.index_to_value(i_1),
                            color=colors[1],
                        )
                if j_0 != np.infty and j_1 != np.infty:
                    plt.plot(
                        (self.CP_0.index_to_value(i_0), self.CP_1.index_to_value(i_1)),
                        (self.CP_0.index_to_value(j_0), self.CP_1.index_to_value(j_1)),
                        color="k",
                    )
        plt.plot(
            (min(x_min, 0), max(x_max, 1)),
            (min(x_min, 0), max(x_max, 1)),
            color="k",
            linewidth=0.8,
        )
        plt.xlim(min(x_min, 0), max(x_max, 1))
        plt.ylim(min(x_min, 0), max(x_max, 1))
        return

    def plot_intervals(self, dim=None):
        if dim == 0 or dim == None:
            x_births_0 = []
            y_births_0 = []
            x_deaths_0 = []
            y_deaths_0 = []
            x_births_comp = []
            y_births_comp = []
            x_deaths_comp = []
            y_deaths_comp = []
            x_births_1 = []
            y_births_1 = []
            x_deaths_1 = []
            y_deaths_1 = []
            for I_0, I_comp, I_1 in self.matched[0]:
                if I_0[1] != np.infty:
                    x, y = self.CP_0.index_to_coordinates(I_0[0])
                    x_births_0.append(x / 2)
                    y_births_0.append(y / 2)
                    j_vert = self.CP_0.get_generating_vertex(I_0[1])
                    x, y = self.CP_0.index_to_coordinates(j_vert)
                    x_deaths_0.append(x / 2)
                    y_deaths_0.append(y / 2)
                else:
                    x, y = self.CP_0.index_to_coordinates(I_0[0])
                    essential_0 = (x / 2, y / 2)
                if I_comp[1] != np.infty:
                    x, y = self.CP_comp.index_to_coordinates(I_comp[0])
                    x_births_comp.append(x / 2)
                    y_births_comp.append(y / 2)
                    j_vert = self.CP_comp.get_generating_vertex(I_comp[1])
                    x, y = self.CP_comp.index_to_coordinates(j_vert)
                    x_deaths_comp.append(x / 2)
                    y_deaths_comp.append(y / 2)
                else:
                    x, y = self.CP_comp.index_to_coordinates(I_comp[0])
                    essential_comp = (x / 2, y / 2)
                if I_1[1] != np.infty:
                    x, y = self.CP_1.index_to_coordinates(I_1[0])
                    x_births_1.append(x / 2)
                    y_births_1.append(y / 2)
                    j_vert = self.CP_1.get_generating_vertex(I_1[1])
                    x, y = self.CP_1.index_to_coordinates(j_vert)
                    x_deaths_1.append(x / 2)
                    y_deaths_1.append(y / 2)
                else:
                    x, y = self.CP_1.index_to_coordinates(I_1[0])
                    essential_1 = (x / 2, y / 2)
            x_births_0_unmatched = []
            y_births_0_unmatched = []
            x_deaths_0_unmatched = []
            y_deaths_0_unmatched = []
            for i, j in self.unmatched_0[0]:
                x, y = self.CP_0.index_to_coordinates(i)
                x_births_0_unmatched.append(x / 2)
                y_births_0_unmatched.append(y / 2)
                j_vert = self.CP_0.get_generating_vertex(j)
                x, y = self.CP_0.index_to_coordinates(j_vert)
                x_deaths_0_unmatched.append(x / 2)
                y_deaths_0_unmatched.append(y / 2)
            x_births_comp_unmatched = []
            y_births_comp_unmatched = []
            x_deaths_comp_unmatched = []
            y_deaths_comp_unmatched = []
            for i, j in self.unmatched_comp[0]:
                x, y = self.CP_comp.index_to_coordinates(i)
                x_births_comp_unmatched.append(x / 2)
                y_births_comp_unmatched.append(y / 2)
                j_vert = self.CP_comp.get_generating_vertex(j)
                x, y = self.CP_comp.index_to_coordinates(j_vert)
                x_deaths_comp_unmatched.append(x / 2)
                y_deaths_comp_unmatched.append(y / 2)
            x_births_1_unmatched = []
            y_births_1_unmatched = []
            x_deaths_1_unmatched = []
            y_deaths_1_unmatched = []
            for i, j in self.unmatched_1[0]:
                x, y = self.CP_1.index_to_coordinates(i)
                x_births_1_unmatched.append(x / 2)
                y_births_1_unmatched.append(y / 2)
                j_vert = self.CP_1.get_generating_vertex(j)
                x, y = self.CP_1.index_to_coordinates(j_vert)
                x_deaths_1_unmatched.append(x / 2)
                y_deaths_1_unmatched.append(y / 2)

            fig = plt.figure(figsize=(15, 15))
            rows = 1
            columns = 3
            fig.add_subplot(rows, columns, 1)
            plt.imshow(self.CP_0.PixelMap, cmap="gray")
            plt.scatter(essential_0[1], essential_0[0], 50, c="g", marker="*")
            plt.scatter(
                y_births_0, x_births_0, 50, c=list(range(len(x_births_0))), marker="*"
            )
            plt.scatter(
                y_deaths_0, x_deaths_0, 50, c=list(range(len(x_deaths_0))), marker="x"
            )
            plt.scatter(
                y_births_0_unmatched, x_births_0_unmatched, 50, c="r", marker="*"
            )
            plt.scatter(
                y_deaths_0_unmatched, x_deaths_0_unmatched, 50, c="r", marker="x"
            )
            plt.axis("off")
            plt.title("Picture 0")
            fig.add_subplot(rows, columns, 2)
            plt.imshow(self.CP_comp.PixelMap, cmap="gray")
            plt.scatter(essential_comp[1], essential_comp[0], 50, c="g", marker="*")
            plt.scatter(
                y_births_comp,
                x_births_comp,
                50,
                c=list(range(len(x_births_comp))),
                marker="*",
            )
            plt.scatter(
                y_deaths_comp,
                x_deaths_comp,
                50,
                c=list(range(len(x_deaths_comp))),
                marker="x",
            )
            plt.scatter(
                y_births_comp_unmatched, x_births_comp_unmatched, 50, c="r", marker="*"
            )
            plt.scatter(
                y_deaths_comp_unmatched, x_deaths_comp_unmatched, 50, c="r", marker="x"
            )
            plt.axis("off")
            if self.comparison == "union":
                plt.title("Union")
            else:
                plt.title("Intersection")
            fig.add_subplot(rows, columns, 3)
            plt.imshow(self.CP_1.PixelMap, cmap="gray")
            plt.scatter(essential_1[1], essential_1[0], 50, c="g", marker="*")
            plt.scatter(
                y_births_1, x_births_1, 50, c=list(range(len(x_births_1))), marker="*"
            )
            plt.scatter(
                y_deaths_1, x_deaths_1, 50, c=list(range(len(x_deaths_1))), marker="x"
            )
            plt.scatter(
                y_births_1_unmatched, x_births_1_unmatched, 50, c="r", marker="*"
            )
            plt.scatter(
                y_deaths_1_unmatched, x_deaths_1_unmatched, 50, c="r", marker="x"
            )
            plt.axis("off")
            plt.title("Picture 1")
        if dim == 1 or dim == None:
            x_births_0 = []
            y_births_0 = []
            x_deaths_0 = []
            y_deaths_0 = []
            x_births_comp = []
            y_births_comp = []
            x_deaths_comp = []
            y_deaths_comp = []
            x_births_1 = []
            y_births_1 = []
            x_deaths_1 = []
            y_deaths_1 = []
            for I_0, I_comp, I_1 in self.matched[1]:
                i_vert = self.CP_0.get_generating_vertex(I_0[0])
                x, y = self.CP_0.index_to_coordinates(i_vert)
                x_births_0.append(x / 2)
                y_births_0.append(y / 2)
                j_vert = self.CP_0.get_generating_vertex(I_0[1])
                x, y = self.CP_0.index_to_coordinates(j_vert)
                x_deaths_0.append(x / 2)
                y_deaths_0.append(y / 2)
                i_vert = self.CP_comp.get_generating_vertex(I_comp[0])
                x, y = self.CP_comp.index_to_coordinates(i_vert)
                x_births_comp.append(x / 2)
                y_births_comp.append(y / 2)
                j_vert = self.CP_comp.get_generating_vertex(I_comp[1])
                x, y = self.CP_comp.index_to_coordinates(j_vert)
                x_deaths_comp.append(x / 2)
                y_deaths_comp.append(y / 2)
                i_vert = self.CP_1.get_generating_vertex(I_1[0])
                x, y = self.CP_1.index_to_coordinates(i_vert)
                x_births_1.append(x / 2)
                y_births_1.append(y / 2)
                j_vert = self.CP_1.get_generating_vertex(I_1[1])
                x, y = self.CP_1.index_to_coordinates(j_vert)
                x_deaths_1.append(x / 2)
                y_deaths_1.append(y / 2)
            x_births_0_unmatched = []
            y_births_0_unmatched = []
            x_deaths_0_unmatched = []
            y_deaths_0_unmatched = []
            for i, j in self.unmatched_0[0]:
                i_vert = self.CP_0.get_generating_vertex(i)
                x, y = self.CP_0.index_to_coordinates(i_vert)
                x_births_0_unmatched.append(x / 2)
                y_births_0_unmatched.append(y / 2)
                j_vert = self.CP_0.get_generating_vertex(j)
                x, y = self.CP_0.index_to_coordinates(j_vert)
                x_deaths_0_unmatched.append(x / 2)
                y_deaths_0_unmatched.append(y / 2)
            x_births_comp_unmatched = []
            y_births_comp_unmatched = []
            x_deaths_comp_unmatched = []
            y_deaths_comp_unmatched = []
            for i, j in self.unmatched_comp[0]:
                i_vert = self.CP_comp.get_generating_vertex(i)
                x, y = self.CP_comp.index_to_coordinates(i_vert)
                x_births_comp_unmatched.append(x / 2)
                y_births_comp_unmatched.append(y / 2)
                j_vert = self.CP_comp.get_generating_vertex(j)
                x, y = self.CP_comp.index_to_coordinates(j_vert)
                x_deaths_comp_unmatched.append(x / 2)
                y_deaths_comp_unmatched.append(y / 2)
            x_births_1_unmatched = []
            y_births_1_unmatched = []
            x_deaths_1_unmatched = []
            y_deaths_1_unmatched = []
            for i, j in self.unmatched_1[0]:
                i_vert = self.CP_1.get_generating_vertex(i)
                x, y = self.CP_1.index_to_coordinates(i_vert)
                x_births_1_unmatched.append(x / 2)
                y_births_1_unmatched.append(y / 2)
                j_vert = self.CP_1.get_generating_vertex(j)
                x, y = self.CP_1.index_to_coordinates(j_vert)
                x_deaths_1_unmatched.append(x / 2)
                y_deaths_1_unmatched.append(y / 2)
            fig = plt.figure(figsize=(15, 15))
            rows = 1
            columns = 3
            fig.add_subplot(rows, columns, 1)
            plt.imshow(self.CP_0.PixelMap, cmap="gray")
            plt.scatter(
                y_births_0, x_births_0, 50, c=list(range(len(x_births_0))), marker="*"
            )
            plt.scatter(
                y_deaths_0, x_deaths_0, 50, c=list(range(len(x_deaths_0))), marker="x"
            )
            plt.scatter(
                y_births_0_unmatched, x_births_0_unmatched, 50, c="r", marker="*"
            )
            plt.scatter(
                y_deaths_0_unmatched, x_deaths_0_unmatched, 50, c="r", marker="x"
            )
            plt.axis("off")
            plt.title("Picture 0")
            fig.add_subplot(rows, columns, 2)
            plt.imshow(self.CP_comp.PixelMap, cmap="gray")
            plt.scatter(
                y_births_comp,
                x_births_comp,
                50,
                c=list(range(len(x_births_comp))),
                marker="*",
            )
            plt.scatter(
                y_deaths_comp,
                x_deaths_comp,
                50,
                c=list(range(len(x_deaths_comp))),
                marker="x",
            )
            plt.scatter(
                y_births_comp_unmatched, x_births_comp_unmatched, 50, c="r", marker="*"
            )
            plt.scatter(
                y_deaths_comp_unmatched, x_deaths_comp_unmatched, 50, c="r", marker="x"
            )
            plt.axis("off")
            if self.comparison == "union":
                plt.title("Union")
            else:
                plt.title("Intersection")
            fig.add_subplot(rows, columns, 3)
            plt.imshow(self.CP_1.PixelMap, cmap="gray")
            plt.scatter(
                y_births_1, x_births_1, 50, c=list(range(len(x_births_1))), marker="*"
            )
            plt.scatter(
                y_deaths_1, x_deaths_1, 50, c=list(range(len(x_deaths_1))), marker="x"
            )
            plt.scatter(
                y_births_1_unmatched, x_births_1_unmatched, 50, c="r", marker="*"
            )
            plt.scatter(
                y_deaths_1_unmatched, x_deaths_1_unmatched, 50, c="r", marker="x"
            )
            plt.axis("off")
            plt.title("Picture 1")
        return

    def get_birth_dics_dim_0(
        self, matches=6, threshold_0=0.5, threshold_comp=0.5, threshold_1=0.5, app=False
    ):
        if len(self.matched[0]) == 0:
            raise ValueError("No matches in dimension 0.")
        assert (type(matches) == list and len(matches) <= 6) or (
            0 <= matches and matches <= 6
        )

        birth_dic_0 = {}
        birth_dic_comp = {}
        birth_dic_1 = {}
        death_vertices_0 = []
        death_vertices_comp = []
        death_vertices_1 = []
        counter = 1

        if type(matches) != list:
            num_matches = min(matches, len(self.matched[0]))
            matches = rd.sample(range(0, len(self.matched[0])), num_matches)

        if len(matches) == 1:
            (I_0, I_comp, I_1) = self.matched[0][matches[0]]
            birth_dic_0[I_0[0]] = counter
            birth_dic_comp[I_comp[0]] = counter
            birth_dic_1[I_1[0]] = counter
            if I_0[1] != np.infty:
                death_vertices_0.append(self.CP_0.get_generating_vertex(I_0[1]))
            if I_comp[1] != np.infty:
                death_vertices_comp.append(
                    self.CP_comp.get_generating_vertex(I_comp[1])
                )
            if I_1[1] != np.infty:
                death_vertices_1.append(self.CP_1.get_generating_vertex(I_1[1]))
        else:
            for I_0, I_comp, I_1 in itemgetter(*matches)(self.matched[0]):
                birth_dic_0[I_0[0]] = counter
                birth_dic_comp[I_comp[0]] = counter
                birth_dic_1[I_1[0]] = counter
                if I_0[1] != np.infty:
                    death_vertices_0.append(self.CP_0.get_generating_vertex(I_0[1]))
                if I_comp[1] != np.infty:
                    death_vertices_comp.append(
                        self.CP_comp.get_generating_vertex(I_comp[1])
                    )
                if I_1[1] != np.infty:
                    death_vertices_1.append(self.CP_1.get_generating_vertex(I_1[1]))
                counter += 1

        if app == False:
            colormap_0 = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_comp = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_1 = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a_0, b_0) = self.CP_0.fine_to_coarse(I_0)
            (a_comp, b_comp) = self.CP_comp.fine_to_coarse(I_comp)
            (a_1, b_1) = self.CP_1.fine_to_coarse(I_1)
            if self.filtration == "sublevel":
                if a_0 <= threshold_0 and threshold_0 < b_0:
                    colormap_0 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_0 = ListedColormap([(1, 0, 0, 1)])
                if a_comp <= threshold_comp and threshold_comp < b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_1 <= threshold_1 and threshold_1 < b_1:
                    colormap_1 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_1 = ListedColormap([(1, 0, 0, 1)])
            elif self.filtration == "superlevel":
                if a_0 >= threshold_0 and threshold_0 > b_0:
                    colormap_0 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_0 = ListedColormap([(1, 0, 0, 1)])
                if a_comp >= threshold_comp and threshold_comp > b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_1 >= threshold_1 and threshold_1 > b_1:
                    colormap_1 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_1 = ListedColormap([(1, 0, 0, 1)])
        return (
            birth_dic_0,
            birth_dic_comp,
            birth_dic_1,
            death_vertices_0,
            death_vertices_comp,
            death_vertices_1,
            colormap_0,
            colormap_comp,
            colormap_1,
        )

    def get_birth_dics_dim_1(
        self, matches, threshold_0=0.5, threshold_comp=0.5, threshold_1=0.5, app=False
    ):
        if len(self.matched[1]) == 0:
            raise ValueError("No matches in dimension 1.")
        assert (type(matches) == list and len(matches) <= 6) or (
            0 <= matches and matches <= 6
        )

        birth_dic_0 = {}
        birth_dic_comp = {}
        birth_dic_1 = {}
        death_dic_0 = {}
        death_dic_comp = {}
        death_dic_1 = {}
        counter = 1

        if len(self.matched[1]) == 0:
            return (
                birth_dic_0,
                birth_dic_comp,
                birth_dic_1,
                death_dic_0,
                death_dic_comp,
                death_dic_1,
            )

        if type(matches) != list:
            num_matches = min(matches, len(self.matched[1]))
            matches = rd.sample(range(0, len(self.matched[1])), num_matches)

        if len(matches) == 1:
            (I_0, I_comp, I_1) = self.matched[1][matches[0]]
            birth_dic_0[I_0[1]] = counter
            death_dic_0[I_0[0]] = I_0[1]
            birth_dic_comp[I_comp[1]] = counter
            death_dic_comp[I_comp[0]] = I_comp[1]
            birth_dic_1[I_1[1]] = counter
            death_dic_1[I_1[0]] = I_1[1]
        else:
            for I_0, I_comp, I_1 in itemgetter(*matches)(self.matched[1]):
                birth_dic_0[I_0[1]] = counter
                death_dic_0[I_0[0]] = I_0[1]
                birth_dic_comp[I_comp[1]] = counter
                death_dic_comp[I_comp[0]] = I_comp[1]
                birth_dic_1[I_1[1]] = counter
                death_dic_1[I_1[0]] = I_1[1]
                counter += 1

        if app == False:
            colormap_0 = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_comp = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_1 = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a_0, b_0) = self.CP_0.fine_to_coarse(I_0)
            (a_comp, b_comp) = self.CP_comp.fine_to_coarse(I_comp)
            (a_1, b_1) = self.CP_1.fine_to_coarse(I_1)
            if self.filtration == "sublevel":
                if a_0 <= threshold_0 and threshold_0 < b_0:
                    colormap_0 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_0 = ListedColormap([(1, 0, 0, 1)])
                if a_comp <= threshold_comp and threshold_comp < b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_1 <= threshold_1 and threshold_1 < b_1:
                    colormap_1 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_1 = ListedColormap([(1, 0, 0, 1)])
            elif self.filtration == "superlevel":
                if a_0 >= threshold_0 and threshold_0 > b_0:
                    colormap_0 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_0 = ListedColormap([(1, 0, 0, 1)])
                if a_comp >= threshold_comp and threshold_comp > b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_1 >= threshold_1 and threshold_1 > b_1:
                    colormap_1 = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_1 = ListedColormap([(1, 0, 0, 1)])
        return (
            birth_dic_0,
            birth_dic_comp,
            birth_dic_1,
            death_dic_0,
            death_dic_comp,
            death_dic_1,
            colormap_0,
            colormap_comp,
            colormap_1,
        )

    def plot_representative_cycles(
        self,
        dim=1,
        matches=6,
        threshold_0=0.5,
        threshold_comp=0.5,
        threshold_1=0.5,
        plot_birth_and_death=False,
        plot_comparison=False,
        app=False,
    ):
        assert dim in [0, 1]

        if dim == 0:
            (
                birth_dic_0,
                birth_dic_comp,
                birth_dic_1,
                death_vertices_0,
                death_vertices_comp,
                death_vertices_1,
                colormap_0,
                colormap_comp,
                colormap_1,
            ) = self.get_birth_dics_dim_0(
                matches=matches,
                threshold_0=threshold_0,
                threshold_comp=threshold_comp,
                threshold_1=threshold_1,
                app=app,
            )

            CycleMap_0, births_0, deaths_0 = self.CP_0.get_CycleMap_dim_0(
                birth_dic_0, death_vertices_0, threshold=threshold_0, app=app
            )
            if plot_comparison:
                CycleMap_comp, births_comp, deaths_comp = (
                    self.CP_comp.get_CycleMap_dim_0(
                        birth_dic_comp,
                        death_vertices_comp,
                        threshold=threshold_comp,
                        app=app,
                    )
                )
            CycleMap_1, births_1, deaths_1 = self.CP_1.get_CycleMap_dim_0(
                birth_dic_1, death_vertices_1, threshold=threshold_1, app=app
            )

        elif dim == 1:
            (
                birth_dic_0,
                birth_dic_comp,
                birth_dic_1,
                death_dic_0,
                death_dic_comp,
                death_dic_1,
                colormap_0,
                colormap_comp,
                colormap_1,
            ) = self.get_birth_dics_dim_1(
                matches=matches,
                threshold_0=threshold_0,
                threshold_comp=threshold_comp,
                threshold_1=threshold_1,
                app=app,
            )

            CycleMap_0, births_0, deaths_0 = self.CP_0.get_CycleMap_dim_1(
                birth_dic_0, death_dic_0
            )
            if plot_comparison:
                CycleMap_comp, births_comp, deaths_comp = (
                    self.CP_comp.get_CycleMap_dim_1(birth_dic_comp, death_dic_comp)
                )
            CycleMap_1, births_1, deaths_1 = self.CP_1.get_CycleMap_dim_1(
                birth_dic_1, death_dic_1
            )

        if app == False:
            CycleMap_0_masked = np.ma.masked_where(CycleMap_0 == 0, CycleMap_0)
            if plot_comparison:
                CycleMap_comp_masked = np.ma.masked_where(
                    CycleMap_comp == 0, CycleMap_comp
                )
            CycleMap_1_masked = np.ma.masked_where(CycleMap_1 == 0, CycleMap_1)
        else:
            if self.filtration == "sublevel":
                CycleMap_0_masked = np.ma.masked_where(
                    (CycleMap_0 == 0) | (self.CP_0.PixelMap > threshold_0), CycleMap_0
                )
                if plot_comparison:
                    CycleMap_comp_masked = np.ma.masked_where(
                        (CycleMap_comp == 0) | (self.CP_comp.PixelMap > threshold_comp),
                        CycleMap_comp,
                    )
                CycleMap_1_masked = np.ma.masked_where(
                    (CycleMap_1 == 0) | (self.CP_1.PixelMap > threshold_1), CycleMap_1
                )
            else:
                CycleMap_0_masked = np.ma.masked_where(
                    (CycleMap_0 == 0) | (self.CP_0.PixelMap < threshold_0), CycleMap_0
                )
                if plot_comparison:
                    CycleMap_comp_masked = np.ma.masked_where(
                        (CycleMap_comp == 0) | (self.CP_comp.PixelMap < threshold_comp),
                        CycleMap_comp,
                    )
                CycleMap_1_masked = np.ma.masked_where(
                    (CycleMap_1 == 0) | (self.CP_1.PixelMap < threshold_1), CycleMap_1
                )

        rows = 1
        if plot_comparison:
            fig = plt.figure(figsize=(15, 5))
            columns = 3
        else:
            fig = plt.figure(figsize=(10, 5))
            columns = 2
        fig.add_subplot(rows, columns, 1)
        if app == False:
            if self.construction == "V":
                plt.imshow(self.CP_0.PixelMap, cmap="gray")
            else:
                plt.imshow(self.CP_0.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(
                            self.CP_0.PixelMap <= threshold_0, self.CP_0.PixelMap, 0
                        ),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(
                            self.CP_0.PixelMap >= threshold_0, self.CP_0.PixelMap, 0
                        ),
                        cmap="gray",
                    )
        plt.imshow(CycleMap_0_masked, cmap=colormap_0, interpolation="none")
        if plot_birth_and_death:
            plt.scatter(births_0[1], births_0[0], 50, c="g", marker="*")
            plt.scatter(deaths_0[1], deaths_0[0], 50, c="r", marker="x")
        plt.axis("off")
        if plot_comparison:
            fig.add_subplot(rows, columns, 2)
            if app == False:
                if self.construction == "V":
                    plt.imshow(self.CP_comp.PixelMap, cmap="gray")
                else:
                    plt.imshow(self.CP_comp.ValueMap, cmap="gray")
            else:
                if self.construction == "V":
                    if self.filtration == "sublevel":
                        plt.imshow(
                            np.where(
                                self.CP_comp.PixelMap <= threshold_comp,
                                self.CP_comp.PixelMap,
                                0,
                            ),
                            cmap="gray",
                        )
                    else:
                        plt.imshow(
                            np.where(
                                self.CP_comp.PixelMap >= threshold_comp,
                                self.CP_comp.PixelMap,
                                0,
                            ),
                            cmap="gray",
                        )
            plt.imshow(CycleMap_comp_masked, cmap=colormap_comp, interpolation="none")
            if plot_birth_and_death:
                plt.scatter(births_comp[1], births_comp[0], 50, c="g", marker="*")
                plt.scatter(deaths_comp[1], deaths_comp[0], 50, c="r", marker="x")
            plt.axis("off")
            fig.add_subplot(rows, columns, 3)
        else:
            fig.add_subplot(rows, columns, 2)
        if app == False:
            if self.construction == "V":
                plt.imshow(self.CP_1.PixelMap, cmap="gray")
            else:
                plt.imshow(self.CP_1.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(
                            self.CP_1.PixelMap <= threshold_1, self.CP_1.PixelMap, 0
                        ),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(
                            self.CP_1.PixelMap >= threshold_1, self.CP_1.PixelMap, 0
                        ),
                        cmap="gray",
                    )
        plt.imshow(CycleMap_1_masked, cmap=colormap_1, interpolation="none")
        if plot_birth_and_death:
            plt.scatter(births_1[1], births_1[0], 50, c="g", marker="*")
            plt.scatter(deaths_1[1], deaths_1[0], 50, c="r", marker="x")
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)
        plt.close(fig)
        return fig

    def plot_representative_cycles_app(
        self, dim=1, plot_birth_and_death=False, plot_comparison=False
    ):
        assert dim in [0, 1]

        def plot(match, threshold_0, threshold_comp, threshold_1):
            return self.plot_representative_cycles(
                dim=dim,
                matches=[match],
                threshold_0=threshold_0,
                threshold_comp=threshold_comp,
                threshold_1=threshold_1,
                plot_birth_and_death=plot_birth_and_death,
                plot_comparison=plot_comparison,
                app=True,
            )

        if len(self.matched[dim]) == 0:
            return "No matches found."

        elif len(self.matched[dim]) == 1:
            i = interact(
                plot,
                match=fixed(0),
                threshold_0=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_0"
                ),
                threshold_comp=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_comp"
                ),
                threshold_1=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_1"
                ),
            )
            if plot_comparison:
                return pn.Column(pn.Row(i[1]), pn.Row(i[0][0], i[0][1], i[0][2]))

            else:
                return pn.Column(pn.Row(i[1]), pn.Row(i[0][0], i[0][2]))

        else:
            i = pn.interact(
                plot,
                match=widgets.IntSlider(
                    start=0, end=len(self.matched[dim]) - 1, value=0, name="match"
                ),
                threshold_0=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_0"
                ),
                threshold_comp=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_comp"
                ),
                threshold_1=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_1"
                ),
            )
            if plot_comparison:
                return pn.Column(
                    pn.Row(i[0][0]), pn.Row(i[1]), pn.Row(i[0][1], i[0][2], i[0][3])
                )

            else:
                return pn.Column(
                    pn.Row(i[0][0]), pn.Row(i[1]), pn.Row(i[0][1], i[0][3])
                )


class WassersteinMatching:
    def __init__(
        self,
        likelihood,
        ground_truth,
        relative=False,
        reduced=False,
        filtration="sublevel",
        construction="V",
        valid="positive",
        training=False,
    ):
        assert valid in ["all", "nonnegative", "positive"]
        assert filtration in ["sublevel", "superlevel"]
        self.filtration = filtration
        assert construction in ["V", "T"]
        self.construction = construction

        self.CP_lh = CubicalPersistence(
            likelihood,
            relative=relative,
            reduced=reduced,
            filtration=filtration,
            construction=construction,
            valid=valid,
            training=training,
        )
        self.CP_gt = CubicalPersistence(
            ground_truth,
            relative=relative,
            reduced=reduced,
            filtration=filtration,
            construction=construction,
            valid=valid,
            training=training,
        )
        self.matched = [[], []]
        self.unmatched_lh = self.CP_lh.get_intervals(refined=True)
        self.unmatched_gt = self.CP_gt.get_intervals(refined=True)
        self.match()

    def potential(self, interval):
        a, b = self.CP_lh.fine_to_coarse(interval)
        if b == np.inf:
            b = 1
        if b == -np.inf:
            b = 0
        if self.filtration == "sublevel":
            return ((a - b) ** 2 + 1) / 2 - (a**2 + (b - 1) ** 2)
        else:
            return ((a - b) ** 2 + 1) / 2 - (b**2 + (a - 1) ** 2)

    def urgency(self, interval):
        a, b = self.CP_lh.fine_to_coarse(interval)
        if b == np.inf:
            b = 1
        if b == -np.inf:
            b = 0
        if self.filtration == "sublevel":
            return 2 * b - ((a + b) ** 2) / 2
        else:
            return 2 * a - ((a + b) ** 2) / 2

    def get_potentials(self):
        potentials = [[], []]
        for dim in range(2):
            for interval in self.unmatched_lh[dim]:
                if self.potential(interval) >= 0:
                    urg = self.urgency(interval)
                    potentials[dim].append((interval, urg))
        return potentials

    def match(self):
        potentials = self.get_potentials()
        for dim in range(2):
            potentials[dim].sort(reverse=True, key=lambda x: x[1])
            num_potentials = len(potentials[dim])
            num_intervals_gt = len(self.unmatched_gt[dim])
            num_matches = min(num_potentials, num_intervals_gt)
            for idx in range(num_matches):
                match_lh = potentials[dim][idx]
                match_gt = self.unmatched_gt[dim][0]
                self.matched[dim].append((match_lh[0], match_gt))
                self.unmatched_lh[dim].remove(match_lh[0])
                self.unmatched_gt[dim].remove(match_gt)
        return

    def get_matching(self, refined=False):
        if refined:
            return (
                copy.deepcopy(self.matched),
                copy.deepcopy(self.unmatched_lh),
                copy.deepcopy(self.unmatched_gt),
            )

        matched = [
            [
                (
                    self.CP_lh.fine_to_coarse(match[0]),
                    self.CP_gt.fine_to_coarse(match[1]),
                )
                for match in self.matched[dim]
            ]
            for dim in range(2)
        ]
        unmatched_lh = [
            [self.CP_lh.fine_to_coarse(interval) for interval in self.unmatched_lh[dim]]
            for dim in range(2)
        ]
        unmatched_gt = [
            [self.CP_gt.fine_to_coarse(interval) for interval in self.unmatched_gt[dim]]
            for dim in range(2)
        ]
        return matched, unmatched_lh, unmatched_gt

    def loss(self, dimensions=[0, 1]):
        loss = 0
        for dim in dimensions:
            for I_lh, I_gt in self.matched[dim]:
                (a_0, b_0) = self.CP_lh.fine_to_coarse(I_lh)
                if b_0 == np.infty:
                    b_0 = 1
                elif b_0 == -np.infty:
                    b_0 = 0
                (a_1, b_1) = self.CP_gt.fine_to_coarse(I_gt)
                if b_1 == np.infty:
                    b_1 = 1
                elif b_1 == -np.infty:
                    b_1 = 0
                loss += (a_0 - a_1) ** 2 + (b_0 - b_1) ** 2
            for I in self.unmatched_lh[dim]:
                (a, b) = self.CP_lh.fine_to_coarse(I)
                if b == np.infty:
                    b = 1
                elif b == -np.infty:
                    b = 0
                loss += ((a - b) ** 2) / 2
            for I in self.unmatched_gt[dim]:
                (a, b) = self.CP_gt.fine_to_coarse(I)
                if b == np.infty:
                    b = 1
                elif b == -np.infty:
                    b = 0
                loss += ((a - b) ** 2) / 2
        return loss

    def get_birth_dics_dim_0(
        self, matches=6, threshold_lh=0.5, threshold_gt=0.5, app=False
    ):
        if len(self.matched[0]) == 0:
            raise ValueError("No matches in dimension 0.")
        assert (type(matches) == list and len(matches) <= 6) or (
            0 <= matches and matches <= 6
        )

        birth_dic_lh = {}
        birth_dic_gt = {}
        death_vertices_lh = []
        death_vertices_gt = []
        counter = 1

        if len(self.matched[0]) == 0:
            return birth_dic_lh, birth_dic_gt, death_vertices_lh, death_vertices_gt

        if type(matches) != list:
            num_matches = min(matches, len(self.matched[0]))
            matches = rd.sample(range(0, len(self.matched[0])), num_matches)

        if len(matches) == 1:
            (I_lh, I_gt) = self.matched[0][matches[0]]
            birth_dic_lh[I_lh[0]] = counter
            birth_dic_gt[I_gt[0]] = counter
            if I_lh[1] != np.infty:
                death_vertices_lh.append(self.CP_lh.get_generating_vertex(I_lh[1]))
            if I_gt[1] != np.infty:
                death_vertices_gt.append(self.CP_gt.get_generating_vertex(I_gt[1]))
        else:
            for I_lh, I_gt in itemgetter(*matches)(self.matched[0]):
                birth_dic_lh[I_lh[0]] = counter
                birth_dic_gt[I_gt[0]] = counter
                if I_lh[1] != np.infty:
                    death_vertices_lh.append(self.CP_lh.get_generating_vertex(I_lh[1]))
                if I_gt[1] != np.infty:
                    death_vertices_gt.append(self.CP_gt.get_generating_vertex(I_gt[1]))
                counter += 1

        if app == False:
            colormap_lh = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_gt = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a_lh, b_lh) = self.CP_lh.fine_to_coarse(I_lh)
            (a_gt, b_gt) = self.CP_gt.fine_to_coarse(I_gt)
            if self.filtration == "sublevel":
                if a_lh <= threshold_lh and threshold_lh < b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_gt <= threshold_gt and threshold_gt < b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
            elif self.filtration == "superlevel":
                if a_lh >= threshold_lh and threshold_lh > b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_gt >= threshold_gt and threshold_gt > b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
        return (
            birth_dic_lh,
            birth_dic_gt,
            death_vertices_lh,
            death_vertices_gt,
            colormap_lh,
            colormap_gt,
        )

    def get_birth_dics_dim_1(
        self, matches, threshold_lh=0.5, threshold_gt=0.5, app=False
    ):
        if len(self.matched[1]) == 0:
            raise ValueError("No matches in dimension 1.")
        assert (type(matches) == list and len(matches) <= 6) or (
            type(matches) == int and 0 <= matches and matches <= 6
        )

        birth_dic_lh = {}
        birth_dic_gt = {}
        death_dic_lh = {}
        death_dic_gt = {}
        counter = 1

        if len(self.matched[1]) == 0:
            return birth_dic_lh, birth_dic_gt, death_dic_lh, death_dic_gt

        if type(matches) != list:
            num_matches = min(matches, len(self.matched[1]))
            matches = rd.sample(range(0, len(self.matched[1])), num_matches)
        if len(matches) == 1:
            (I_lh, I_gt) = self.matched[1][matches[0]]
            birth_dic_lh[I_lh[1]] = counter
            death_dic_lh[I_lh[0]] = I_lh[1]
            birth_dic_gt[I_gt[1]] = counter
            death_dic_gt[I_gt[0]] = I_gt[1]
        else:
            for I_lh, I_gt in itemgetter(*matches)(self.matched[1]):
                birth_dic_lh[I_lh[1]] = counter
                death_dic_lh[I_lh[0]] = I_lh[1]
                birth_dic_gt[I_gt[1]] = counter
                death_dic_gt[I_gt[0]] = I_gt[1]
                counter += 1

        if app == False:
            colormap_lh = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_gt = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a_lh, b_lh) = self.CP_lh.fine_to_coarse(I_lh)
            (a_gt, b_gt) = self.CP_gt.fine_to_coarse(I_gt)
            if self.filtration == "sublevel":
                if a_lh <= threshold_lh and threshold_lh < b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_gt <= threshold_gt and threshold_gt < b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
            elif self.filtration == "superlevel":
                if a_lh >= threshold_lh and threshold_lh > b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_gt >= threshold_gt and threshold_gt > b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
        return (
            birth_dic_lh,
            birth_dic_gt,
            death_dic_lh,
            death_dic_gt,
            colormap_lh,
            colormap_gt,
        )

    def plot_representative_cycles(
        self,
        dim=1,
        matches=6,
        threshold_lh=0.5,
        threshold_gt=0.5,
        plot_birth_and_death=False,
        app=False,
    ):
        assert dim in [0, 1]

        if dim == 0:
            (
                birth_dic_lh,
                birth_dic_gt,
                death_vertices_lh,
                death_vertices_gt,
                colormap_lh,
                colormap_gt,
            ) = self.get_birth_dics_dim_0(
                matches=matches,
                threshold_lh=threshold_lh,
                threshold_gt=threshold_gt,
                app=app,
            )
            CycleMap_lh, births_lh, deaths_lh = self.CP_lh.get_CycleMap_dim_0(
                birth_dic_lh, death_vertices_lh, threshold=threshold_lh, app=app
            )
            CycleMap_gt, births_gt, deaths_gt = self.CP_gt.get_CycleMap_dim_0(
                birth_dic_gt, death_vertices_gt, threshold=threshold_gt, app=app
            )
        elif dim == 1:
            (
                birth_dic_lh,
                birth_dic_gt,
                death_dic_lh,
                death_dic_gt,
                colormap_lh,
                colormap_gt,
            ) = self.get_birth_dics_dim_1(
                matches=matches,
                threshold_lh=threshold_lh,
                threshold_gt=threshold_gt,
                app=app,
            )
            CycleMap_lh, births_lh, deaths_lh = self.CP_lh.get_CycleMap_dim_1(
                birth_dic_lh, death_dic_lh
            )
            CycleMap_gt, births_gt, deaths_gt = self.CP_gt.get_CycleMap_dim_1(
                birth_dic_gt, death_dic_gt
            )

        if app == False:
            CycleMap_lh_masked = np.ma.masked_where(CycleMap_lh == 0, CycleMap_lh)
            CycleMap_gt_masked = np.ma.masked_where(CycleMap_gt == 0, CycleMap_gt)
        else:
            if self.filtration == "sublevel":
                CycleMap_lh_masked = np.ma.masked_where(
                    (CycleMap_lh == 0) | (self.CP_lh.PixelMap > threshold_lh),
                    CycleMap_lh,
                )
                CycleMap_gt_masked = np.ma.masked_where(
                    (CycleMap_gt == 0) | (self.CP_gt.PixelMap > threshold_gt),
                    CycleMap_gt,
                )
            else:
                CycleMap_lh_masked = np.ma.masked_where(
                    (CycleMap_lh == 0) | (self.CP_lh.PixelMap < threshold_lh),
                    CycleMap_lh,
                )
                CycleMap_gt_masked = np.ma.masked_where(
                    (CycleMap_gt == 0) | (self.CP_gt.PixelMap < threshold_gt),
                    CycleMap_gt,
                )

        fig = plt.figure(figsize=(10, 5))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        if app == False:
            if self.construction == "V":
                plt.imshow(self.CP_lh.PixelMap, cmap="gray")
            else:
                plt.imshow(self.CP_lh.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(
                            self.CP_lh.PixelMap <= threshold_lh, self.CP_lh.PixelMap, 0
                        ),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(
                            self.CP_lh.PixelMap >= threshold_lh, self.CP_lh.PixelMap, 0
                        ),
                        cmap="gray",
                    )
        plt.imshow(CycleMap_lh_masked, cmap=colormap_lh, interpolation="none")
        if plot_birth_and_death:
            plt.scatter(births_lh[1], births_lh[0], 50, c="g", marker="*")
            plt.scatter(deaths_lh[1], deaths_lh[0], 50, c="r", marker="x")
        plt.axis("off")
        fig.add_subplot(rows, columns, 2)
        if app == False:
            if self.construction == "V":
                plt.imshow(self.CP_gt.PixelMap, cmap="gray")
            else:
                plt.imshow(self.CP_gt.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(
                            self.CP_gt.PixelMap <= threshold_gt, self.CP_gt.PixelMap, 0
                        ),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(
                            self.CP_gt.PixelMap >= threshold_gt, self.CP_gt.PixelMap, 0
                        ),
                        cmap="gray",
                    )
        plt.imshow(CycleMap_gt_masked, cmap=colormap_gt, interpolation="none")
        if plot_birth_and_death:
            plt.scatter(births_gt[1], births_gt[0], 50, c="g", marker="*")
            plt.scatter(deaths_gt[1], deaths_gt[0], 50, c="r", marker="x")
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)
        plt.close(fig)
        return fig

    def plot_representative_cycles_app(self, dim=1, plot_birth_and_death=False):
        assert dim in [0, 1]

        def plot(match, threshold_lh, threshold_gt):
            return self.plot_representative_cycles(
                dim=dim,
                matches=[match],
                threshold_lh=threshold_lh,
                threshold_gt=threshold_gt,
                plot_birth_and_death=plot_birth_and_death,
                app=True,
            )

        if len(self.matched[dim]) == 0:
            return "No matches found."

        elif len(self.matched[dim]) == 1:
            i = interact(
                plot,
                match=fixed(0),
                threshold_lh=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_lh"
                ),
                threshold_gt=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_gt"
                ),
            )
            return pn.Column(pn.Row(i[1]), pn.Row(i[0][0], i[0][1]))

        else:
            i = pn.interact(
                plot,
                match=widgets.IntSlider(
                    start=0, end=len(self.matched[dim]) - 1, value=0, name="match"
                ),
                threshold_lh=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_lh"
                ),
                threshold_gt=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_gt"
                ),
            )
            return pn.Column(pn.Row(i[0][0]), pn.Row(i[1]), pn.Row(i[0][1], i[0][2]))


class ComposedWassersteinMatching:
    def __init__(
        self,
        likelihood,
        ground_truth,
        relative=False,
        reduced=False,
        filtration="sublevel",
        construction="V",
        comparison="union",
        valid="positive",
        training=False,
    ):
        assert valid in ["all", "nonnegative", "positive"]
        assert filtration in ["sublevel", "superlevel"]
        self.filtration = filtration
        assert construction in ["V", "T"]
        self.construction = construction
        assert comparison in ["union", "intersection"]
        self.comparison = comparison
        self.reduced = reduced
        self.training = training

        if comparison == "union":
            if filtration == "sublevel":
                if type(likelihood) == torch.Tensor:
                    Picture_comp = torch.minimum(likelihood, ground_truth)
                else:
                    Picture_comp = np.minimum(likelihood, ground_truth)
            else:
                if type(likelihood) == torch.Tensor:
                    Picture_comp = torch.maximum(likelihood, ground_truth)
                else:
                    Picture_comp = np.maximum(likelihood, ground_truth)
            self.CP_lh = CubicalPersistence(
                likelihood,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                training=training,
            )
            self.CP_gt = CubicalPersistence(
                ground_truth,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                training=training,
            )
            self.CP_comp = CubicalPersistence(
                Picture_comp,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                training=training,
            )
        else:
            if filtration == "sublevel":
                if type(likelihood) == torch.Tensor:
                    Picture_comp = torch.maximum(likelihood, ground_truth)
                else:
                    Picture_comp = np.maximum(likelihood, ground_truth)
            else:
                if type(likelihood) == torch.Tensor:
                    Picture_comp = torch.minimum(likelihood, ground_truth)
                else:
                    Picture_comp = np.minimum(likelihood, ground_truth)
            self.CP_comp = CubicalPersistence(
                Picture_comp,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                training=training,
            )
            self.CP_lh = CubicalPersistence(
                likelihood,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                training=training,
            )
            self.CP_gt = CubicalPersistence(
                ground_truth,
                relative=relative,
                reduced=reduced,
                valid=valid,
                filtration=filtration,
                construction=construction,
                training=training,
            )

        self.unmatched_lh = self.CP_lh.get_intervals(refined=True)
        self.unmatched_comp = self.CP_comp.get_intervals(refined=True)
        self.unmatched_gt = self.CP_gt.get_intervals(refined=True)
        if not self.reduced:
            self.matched = [[((0, np.infty), (0, np.infty), (0, np.infty))], []]
            self.unmatched_lh[0].remove((0, np.infty))
            self.unmatched_comp[0].remove((0, np.infty))
            self.unmatched_gt[0].remove((0, np.infty))
        else:
            self.matched = [[], []]
        self.match()

    def match(self):
        intervals_lh = self.CP_lh.get_intervals(refined=False)
        intervals_comp = self.CP_comp.get_intervals(refined=False)
        intervals_gt = self.CP_gt.get_intervals(refined=False)
        if not self.reduced:
            intervals_lh[0].remove(self.CP_lh.fine_to_coarse((0, np.infty)))
            intervals_comp[0].remove(self.CP_comp.fine_to_coarse((0, np.infty)))
            intervals_gt[0].remove(self.CP_gt.fine_to_coarse((0, np.infty)))

        if self.filtration == "sublevel":
            if self.training:
                intervals_0_lh = np.array(
                    [
                        [tupel[0].detach().cpu(), tupel[1].detach().cpu()]
                        for tupel in intervals_lh[0]
                    ]
                )
                intervals_1_lh = np.array(
                    [
                        [tupel[0].detach().cpu(), tupel[1].detach().cpu()]
                        for tupel in intervals_lh[1]
                    ]
                )
                intervals_0_comp = np.array(
                    [
                        [tupel[0].detach().cpu(), tupel[1].detach().cpu()]
                        for tupel in intervals_comp[0]
                    ]
                )
                intervals_1_comp = np.array(
                    [
                        [tupel[0].detach().cpu(), tupel[1].detach().cpu()]
                        for tupel in intervals_comp[1]
                    ]
                )
                intervals_0_gt = np.array(
                    [
                        [tupel[0].detach().cpu(), tupel[1].detach().cpu()]
                        for tupel in intervals_gt[0]
                    ]
                )
                intervals_1_gt = np.array(
                    [
                        [tupel[0].detach().cpu(), tupel[1].detach().cpu()]
                        for tupel in intervals_gt[1]
                    ]
                )
            else:
                intervals_0_lh = np.array(
                    [[tupel[0], tupel[1]] for tupel in intervals_lh[0]]
                )
                intervals_1_lh = np.array(
                    [[tupel[0], tupel[1]] for tupel in intervals_lh[1]]
                )
                intervals_0_comp = np.array(
                    [[tupel[0], tupel[1]] for tupel in intervals_comp[0]]
                )
                intervals_1_comp = np.array(
                    [[tupel[0], tupel[1]] for tupel in intervals_comp[1]]
                )
                intervals_0_gt = np.array(
                    [[tupel[0], tupel[1]] for tupel in intervals_gt[0]]
                )
                intervals_1_gt = np.array(
                    [[tupel[0], tupel[1]] for tupel in intervals_gt[1]]
                )
        else:
            if self.training:
                intervals_0_lh = np.array(
                    [
                        [tupel[1].detach().cpu(), tupel[0].detach().cpu()]
                        for tupel in intervals_lh[0]
                    ]
                )
                intervals_1_lh = np.array(
                    [
                        [tupel[1].detach().cpu(), tupel[0].detach().cpu()]
                        for tupel in intervals_lh[1]
                    ]
                )
                intervals_0_comp = np.array(
                    [
                        [tupel[1].detach().cpu(), tupel[0].detach().cpu()]
                        for tupel in intervals_comp[0]
                    ]
                )
                intervals_1_comp = np.array(
                    [
                        [tupel[1].detach().cpu(), tupel[0].detach().cpu()]
                        for tupel in intervals_comp[1]
                    ]
                )
                intervals_0_gt = np.array(
                    [
                        [tupel[1].detach().cpu(), tupel[0].detach().cpu()]
                        for tupel in intervals_gt[0]
                    ]
                )
                intervals_1_gt = np.array(
                    [
                        [tupel[1].detach().cpu(), tupel[0].detach().cpu()]
                        for tupel in intervals_gt[1]
                    ]
                )
            else:
                intervals_0_lh = np.array(
                    [[tupel[1], tupel[0]] for tupel in intervals_lh[0]]
                )
                intervals_1_lh = np.array(
                    [[tupel[1], tupel[0]] for tupel in intervals_lh[1]]
                )
                intervals_0_comp = np.array(
                    [[tupel[1], tupel[0]] for tupel in intervals_comp[0]]
                )
                intervals_1_comp = np.array(
                    [[tupel[1], tupel[0]] for tupel in intervals_comp[1]]
                )
                intervals_0_gt = np.array(
                    [[tupel[1], tupel[0]] for tupel in intervals_gt[0]]
                )
                intervals_1_gt = np.array(
                    [[tupel[1], tupel[0]] for tupel in intervals_gt[1]]
                )
        _, matched_0_lh = gudhi.wasserstein.wasserstein_distance(
            intervals_0_lh, intervals_0_comp, matching=True, order=1, internal_p=2
        )
        _, matched_1_lh = gudhi.wasserstein.wasserstein_distance(
            intervals_1_lh, intervals_1_comp, matching=True, order=1, internal_p=2
        )
        _, matched_0_gt = gudhi.wasserstein.wasserstein_distance(
            intervals_0_gt, intervals_0_comp, matching=True, order=1, internal_p=2
        )
        _, matched_1_gt = gudhi.wasserstein.wasserstein_distance(
            intervals_1_gt, intervals_1_comp, matching=True, order=1, internal_p=2
        )
        matched_lh = [matched_0_lh.tolist(), matched_1_lh.tolist()]
        matched_gt = [matched_0_gt.tolist(), matched_1_gt.tolist()]

        for dim in range(2):
            remove_lh = []
            remove_comp = []
            remove_gt = []
            for match_lh in matched_lh[dim]:
                if match_lh[0] != -1 and match_lh[1] != -1:
                    for match_gt in matched_gt[dim]:
                        if match_gt[0] != -1:
                            if match_lh[1] == match_gt[1]:
                                self.matched[dim].append(
                                    (
                                        self.unmatched_lh[dim][match_lh[0]],
                                        self.unmatched_comp[dim][match_lh[1]],
                                        self.unmatched_gt[dim][match_gt[0]],
                                    )
                                )
                                remove_lh.append(self.unmatched_lh[dim][match_lh[0]])
                                remove_comp.append(
                                    self.unmatched_comp[dim][match_lh[1]]
                                )
                                remove_gt.append(self.unmatched_gt[dim][match_gt[0]])
                                break
            for interval in remove_lh:
                self.unmatched_lh[dim].remove(interval)
            for interval in remove_comp:
                self.unmatched_comp[dim].remove(interval)
            for interval in remove_gt:
                self.unmatched_gt[dim].remove(interval)
        return

    def get_matching(self):
        matched = [
            [
                (
                    self.CP_lh.fine_to_coarse(match[0]),
                    self.CP_gt.fine_to_coarse(match[2]),
                )
                for match in self.matched[dim]
            ]
            for dim in range(2)
        ]
        unmatched_lh = [
            [self.CP_lh.fine_to_coarse(interval) for interval in self.unmatched_lh[dim]]
            for dim in range(2)
        ]
        unmatched_gt = [
            [self.CP_gt.fine_to_coarse(interval) for interval in self.unmatched_gt[dim]]
            for dim in range(2)
        ]
        return matched, unmatched_lh, unmatched_gt

    def loss(self, dimensions=[0, 1]):
        loss = 0
        for dim in dimensions:
            for I_lh, I_comp, I_gt in self.matched[dim]:
                (a_lh, b_lh) = self.CP_lh.fine_to_coarse(I_lh)
                if b_lh == np.infty:
                    b_lh = 1
                elif b_lh == -np.infty:
                    b_lh = 0
                (a_gt, b_gt) = self.CP_gt.fine_to_coarse(I_gt)
                if b_gt == np.infty:
                    b_gt = 1
                elif b_gt == -np.infty:
                    b_gt = 0
                loss += (a_lh - a_gt) ** 2 + (b_lh - b_gt) ** 2
            for I in self.unmatched_lh[dim]:
                (a, b) = self.CP_lh.fine_to_coarse(I)
                if b == np.infty:
                    b = 1
                elif b == -np.infty:
                    b = 0
                loss += ((a - b) ** 2) / 2
            for I in self.unmatched_gt[dim]:
                (a, b) = self.CP_gt.fine_to_coarse(I)
                if b == np.infty:
                    b = 1
                elif b == -np.infty:
                    b = 0
                loss += ((a - b) ** 2) / 2
        return loss

    def Betti_error(self, threshold, dimensions=[0, 1]):
        betti_lh = self.CP_lh.get_Betti_numbers(threshold=threshold)
        betti_gt = self.CP_gt.get_Betti_numbers(threshold=threshold)
        betti_err = 0
        for dim in dimensions:
            betti_err += np.abs(betti_lh[dim] - betti_gt[dim])
        return betti_err

    def BarCode(self, plot_comparison=False, colors=["r", "b", "g"], ratio=1):
        w, h = matplotlib.figure.figaspect(ratio)
        plt.figure(figsize=(w, h))
        intervals_lh = [
            self.CP_lh.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.CP_lh.intervals[dim]
        ]
        intervals_gt = [
            self.CP_gt.fine_to_coarse(interval)
            for dim in range(2)
            for interval in self.CP_gt.intervals[dim]
        ]
        if self.filtration == "sublevel":
            max_val_lh = max(
                intervals_lh, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_lh = min(intervals_lh, key=lambda x: x[0])[0]
        else:
            max_val_lh = max(intervals_lh, key=lambda x: x[0])[0]
            min_val_lh = min(
                intervals_lh, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        if self.filtration == "sublevel":
            max_val_gt = max(
                intervals_gt, key=lambda x: x[1] if (x[1] != np.infty) else -np.infty
            )[1]
            min_val_gt = min(intervals_gt, key=lambda x: x[0])[0]
        else:
            max_val_gt = max(intervals_gt, key=lambda x: x[0])[0]
            min_val_gt = min(
                intervals_gt, key=lambda x: x[1] if (x[1] != -np.infty) else np.infty
            )[1]
        max_val = max(max_val_lh, max_val_gt)
        min_val = min(min_val_lh, min_val_gt)
        x_min = min_val - (max_val - min_val) * 0.1
        x_max = max_val + (max_val - min_val) * 0.1
        for dim in range(2):
            num_intervals = (
                len(self.unmatched_lh[dim])
                + len(self.unmatched_gt[dim])
                + len(self.matched[dim])
            )
            alpha = 1 / 4
            delta = 1 / (num_intervals + 1)
            height = dim + delta
            for i, j in self.unmatched_lh[dim]:
                if j == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_lh.index_to_value(i), x_max),
                            (height, height),
                            color=colors[0],
                        )
                    else:
                        plt.plot(
                            (self.CP_lh.index_to_value(i), x_min),
                            (height, height),
                            color=colors[0],
                        )
                else:
                    plt.plot(
                        self.CP_lh.fine_to_coarse((i, j)),
                        (height, height),
                        color=colors[0],
                    )
                height += delta
            for i, j in self.unmatched_gt[dim]:
                if j == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_gt.index_to_value(i), x_max),
                            (height, height),
                            color=colors[1],
                        )
                    else:
                        plt.plot(
                            (self.CP_gt.index_to_value(i), x_min),
                            (height, height),
                            color=colors[1],
                        )
                else:
                    plt.plot(
                        self.CP_gt.fine_to_coarse((i, j)),
                        (height, height),
                        color=colors[1],
                    )
                height += delta
            for (i_lh, j_lh), (i_comp, j_comp), (i_gt, j_gt) in self.matched[dim]:
                if j_lh == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_lh.index_to_value(i_lh), x_max),
                            (height - delta * alpha, height - delta * alpha),
                            color=colors[0],
                        )
                    else:
                        plt.plot(
                            (self.CP_lh.index_to_value(i_lh), x_min),
                            (height - delta * alpha, height - delta * alpha),
                            color=colors[0],
                        )
                else:
                    plt.plot(
                        self.CP_lh.fine_to_coarse((i_lh, j_lh)),
                        (height - delta * alpha, height - delta * alpha),
                        color=colors[0],
                    )
                if plot_comparison == True:
                    if j_comp == np.infty:
                        if self.filtration == "sublevel":
                            plt.plot(
                                (self.CP_comp.index_to_value(i_comp), x_max),
                                (height, height),
                                color=colors[2],
                            )
                        else:
                            plt.plot(
                                (self.CP_comp.index_to_value(i_comp), x_min),
                                (height, height),
                                color=colors[2],
                            )
                    else:
                        plt.plot(
                            self.CP_comp.fine_to_coarse((i_comp, j_comp)),
                            (height, height),
                            color=colors[2],
                        )
                if j_gt == np.infty:
                    if self.filtration == "sublevel":
                        plt.plot(
                            (self.CP_gt.index_to_value(i_gt), x_max),
                            (height + delta * alpha, height + delta * alpha),
                            color=colors[1],
                        )
                    else:
                        plt.plot(
                            (self.CP_gt.index_to_value(i_gt), x_min),
                            (height + delta * alpha, height + delta * alpha),
                            color=colors[1],
                        )
                else:
                    plt.plot(
                        self.CP_gt.fine_to_coarse((i_gt, j_gt)),
                        (height + delta * alpha, height + delta * alpha),
                        color=colors[1],
                    )
                height += delta
        plt.plot((x_min, x_max), (1, 1), color="k", linewidth=0.8)
        plt.ylabel("Dimension")
        plt.xlim(x_min, x_max)
        plt.ylim(0, 2)
        plt.yticks([0.5, 1.5], [0, 1])
        return

    def get_birth_dics_dim_0(
        self,
        matches=6,
        threshold_lh=0.5,
        threshold_comp=0.5,
        threshold_gt=0.5,
        app=False,
    ):
        if len(self.matched[0]) == 0:
            raise ValueError("No matches in dimension 0.")
        assert (type(matches) == list and len(matches) <= 6) or (
            0 <= matches and matches <= 6
        )

        birth_dic_lh = {}
        birth_dic_comp = {}
        birth_dic_gt = {}
        death_vertices_lh = []
        death_vertices_comp = []
        death_vertices_gt = []
        counter = 1

        if type(matches) != list:
            num_matches = min(matches, len(self.matched[0]))
            matches = rd.sample(range(0, len(self.matched[0])), num_matches)

        if len(matches) == 1:
            (I_lh, I_comp, I_gt) = self.matched[0][matches[0]]
            birth_dic_lh[I_lh[0]] = counter
            birth_dic_comp[I_comp[0]] = counter
            birth_dic_gt[I_gt[0]] = counter
            if I_lh[1] != np.infty:
                death_vertices_lh.append(self.CP_lh.get_generating_vertex(I_lh[1]))
            if I_comp[1] != np.infty:
                death_vertices_comp.append(
                    self.CP_comp.get_generating_vertex(I_comp[1])
                )
            if I_gt[1] != np.infty:
                death_vertices_gt.append(self.CP_gt.get_generating_vertex(I_gt[1]))
        else:
            for I_lh, I_comp, I_gt in itemgetter(*matches)(self.matched[0]):
                birth_dic_lh[I_lh[0]] = counter
                birth_dic_comp[I_comp[0]] = counter
                birth_dic_gt[I_gt[0]] = counter
                if I_lh[1] != np.infty:
                    death_vertices_lh.append(self.CP_lh.get_generating_vertex(I_lh[1]))
                if I_comp[1] != np.infty:
                    death_vertices_comp.append(
                        self.CP_comp.get_generating_vertex(I_comp[1])
                    )
                if I_gt[1] != np.infty:
                    death_vertices_gt.append(self.CP_gt.get_generating_vertex(I_gt[1]))
                counter += 1

        if app == False:
            colormap_lh = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_comp = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_gt = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a_lh, b_lh) = self.CP_lh.fine_to_coarse(I_lh)
            (a_comp, b_comp) = self.CP_comp.fine_to_coarse(I_comp)
            (a_gt, b_gt) = self.CP_gt.fine_to_coarse(I_gt)
            if self.filtration == "sublevel":
                if a_lh <= threshold_lh and threshold_lh < b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_comp <= threshold_comp and threshold_comp < b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_gt <= threshold_gt and threshold_gt < b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
            elif self.filtration == "superlevel":
                if a_lh >= threshold_lh and threshold_lh > b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_comp >= threshold_comp and threshold_comp > b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_gt >= threshold_gt and threshold_gt > b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
        return (
            birth_dic_lh,
            birth_dic_comp,
            birth_dic_gt,
            death_vertices_lh,
            death_vertices_comp,
            death_vertices_gt,
            colormap_lh,
            colormap_comp,
            colormap_gt,
        )

    def get_birth_dics_dim_1(
        self, matches, threshold_lh=0.5, threshold_comp=0.5, threshold_gt=0.5, app=False
    ):
        if len(self.matched[1]) == 0:
            raise ValueError("No matches in dimension 1.")
        assert (type(matches) == list and len(matches) <= 6) or (
            0 <= matches and matches <= 6
        )

        birth_dic_lh = {}
        birth_dic_comp = {}
        birth_dic_gt = {}
        death_dic_lh = {}
        death_dic_comp = {}
        death_dic_gt = {}
        counter = 1

        if len(self.matched[1]) == 0:
            return (
                birth_dic_lh,
                birth_dic_comp,
                birth_dic_gt,
                death_dic_lh,
                death_dic_comp,
                death_dic_gt,
            )

        if type(matches) != list:
            num_matches = min(matches, len(self.matched[1]))
            matches = rd.sample(range(0, len(self.matched[1])), num_matches)

        if len(matches) == 1:
            (I_lh, I_comp, I_gt) = self.matched[1][matches[0]]
            birth_dic_lh[I_lh[1]] = counter
            death_dic_lh[I_lh[0]] = I_lh[1]
            birth_dic_comp[I_comp[1]] = counter
            death_dic_comp[I_comp[0]] = I_comp[1]
            birth_dic_gt[I_gt[1]] = counter
            death_dic_gt[I_gt[0]] = I_gt[1]
        else:
            for I_lh, I_comp, I_gt in itemgetter(*matches)(self.matched[1]):
                birth_dic_lh[I_lh[1]] = counter
                death_dic_lh[I_lh[0]] = I_lh[1]
                birth_dic_comp[I_comp[1]] = counter
                death_dic_comp[I_comp[0]] = I_comp[1]
                birth_dic_gt[I_gt[1]] = counter
                death_dic_gt[I_gt[0]] = I_gt[1]
                counter += 1

        if app == False:
            colormap_lh = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_comp = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
            colormap_gt = ListedColormap(
                [
                    (1, 0, 0, 1),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (1, 1, 0, 1),
                    (1, 0, 1, 1),
                    (0, 1, 1, 1),
                ]
            )
        else:
            (a_lh, b_lh) = self.CP_lh.fine_to_coarse(I_lh)
            (a_comp, b_comp) = self.CP_comp.fine_to_coarse(I_comp)
            (a_gt, b_gt) = self.CP_gt.fine_to_coarse(I_gt)
            if self.filtration == "sublevel":
                if a_lh <= threshold_lh and threshold_lh < b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_comp <= threshold_comp and threshold_comp < b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_gt <= threshold_gt and threshold_gt < b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
            elif self.filtration == "superlevel":
                if a_lh >= threshold_lh and threshold_lh > b_lh:
                    colormap_lh = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_lh = ListedColormap([(1, 0, 0, 1)])
                if a_comp >= threshold_comp and threshold_comp > b_comp:
                    colormap_comp = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_comp = ListedColormap([(1, 0, 0, 1)])
                if a_gt >= threshold_gt and threshold_gt > b_gt:
                    colormap_gt = ListedColormap([(0, 1, 0, 1)])
                else:
                    colormap_gt = ListedColormap([(1, 0, 0, 1)])
        return (
            birth_dic_lh,
            birth_dic_comp,
            birth_dic_gt,
            death_dic_lh,
            death_dic_comp,
            death_dic_gt,
            colormap_lh,
            colormap_comp,
            colormap_gt,
        )

    def plot_representative_cycles(
        self,
        dim=1,
        matches=6,
        threshold_lh=0.5,
        threshold_comp=0.5,
        threshold_gt=0.5,
        plot_birth_and_death=False,
        plot_comparison=False,
        app=False,
    ):
        assert dim in [0, 1]

        if dim == 0:
            (
                birth_dic_lh,
                birth_dic_comp,
                birth_dic_gt,
                death_vertices_lh,
                death_vertices_comp,
                death_vertices_gt,
                colormap_lh,
                colormap_comp,
                colormap_gt,
            ) = self.get_birth_dics_dim_0(
                matches=matches,
                threshold_lh=threshold_lh,
                threshold_comp=threshold_comp,
                threshold_gt=threshold_gt,
                app=app,
            )

            CycleMap_lh, births_lh, deaths_lh = self.CP_lh.get_CycleMap_dim_0(
                birth_dic_lh, death_vertices_lh, threshold=threshold_lh, app=app
            )
            if plot_comparison:
                CycleMap_comp, births_comp, deaths_comp = (
                    self.CP_comp.get_CycleMap_dim_0(
                        birth_dic_comp,
                        death_vertices_comp,
                        threshold=threshold_comp,
                        app=app,
                    )
                )
            CycleMap_gt, births_gt, deaths_gt = self.CP_gt.get_CycleMap_dim_0(
                birth_dic_gt, death_vertices_gt, threshold=threshold_gt, app=app
            )

        elif dim == 1:
            (
                birth_dic_lh,
                birth_dic_comp,
                birth_dic_gt,
                death_dic_lh,
                death_dic_comp,
                death_dic_gt,
                colormap_lh,
                colormap_comp,
                colormap_gt,
            ) = self.get_birth_dics_dim_1(
                matches=matches,
                threshold_lh=threshold_lh,
                threshold_comp=threshold_comp,
                threshold_gt=threshold_gt,
                app=app,
            )

            CycleMap_lh, births_lh, deaths_lh = self.CP_lh.get_CycleMap_dim_1(
                birth_dic_lh, death_dic_lh
            )
            if plot_comparison:
                CycleMap_comp, births_comp, deaths_comp = (
                    self.CP_comp.get_CycleMap_dim_1(birth_dic_comp, death_dic_comp)
                )
            CycleMap_gt, births_gt, deaths_gt = self.CP_gt.get_CycleMap_dim_1(
                birth_dic_gt, death_dic_gt
            )

        if app == False:
            CycleMap_lh_masked = np.ma.masked_where(CycleMap_lh == 0, CycleMap_lh)
            if plot_comparison:
                CycleMap_comp_masked = np.ma.masked_where(
                    CycleMap_comp == 0, CycleMap_comp
                )
            CycleMap_gt_masked = np.ma.masked_where(CycleMap_gt == 0, CycleMap_gt)
        else:
            if self.filtration == "sublevel":
                CycleMap_lh_masked = np.ma.masked_where(
                    (CycleMap_lh == 0) | (self.CP_lh.PixelMap > threshold_lh),
                    CycleMap_lh,
                )
                if plot_comparison:
                    CycleMap_comp_masked = np.ma.masked_where(
                        (CycleMap_comp == 0) | (self.CP_comp.PixelMap > threshold_comp),
                        CycleMap_comp,
                    )
                CycleMap_gt_masked = np.ma.masked_where(
                    (CycleMap_gt == 0) | (self.CP_gt.PixelMap > threshold_gt),
                    CycleMap_gt,
                )
            else:
                CycleMap_lh_masked = np.ma.masked_where(
                    (CycleMap_lh == 0) | (self.CP_lh.PixelMap < threshold_lh),
                    CycleMap_lh,
                )
                if plot_comparison:
                    CycleMap_comp_masked = np.ma.masked_where(
                        (CycleMap_comp == 0) | (self.CP_comp.PixelMap < threshold_comp),
                        CycleMap_comp,
                    )
                CycleMap_gt_masked = np.ma.masked_where(
                    (CycleMap_gt == 0) | (self.CP_gt.PixelMap < threshold_gt),
                    CycleMap_gt,
                )

        rows = 1
        if plot_comparison:
            fig = plt.figure(figsize=(15, 5))
            columns = 3
        else:
            fig = plt.figure(figsize=(10, 5))
            columns = 2
        fig.add_subplot(rows, columns, 1)
        if app == False:
            if self.construction == "V":
                plt.imshow(self.CP_lh.PixelMap, cmap="gray")
            else:
                plt.imshow(self.CP_lh.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(
                            self.CP_lh.PixelMap <= threshold_lh, self.CP_lh.PixelMap, 0
                        ),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(
                            self.CP_lh.PixelMap >= threshold_lh, self.CP_lh.PixelMap, 0
                        ),
                        cmap="gray",
                    )
        plt.imshow(CycleMap_lh_masked, cmap=colormap_lh, interpolation="none")
        if plot_birth_and_death:
            plt.scatter(births_lh[1], births_lh[0], 300, c="g", marker="*")
            plt.scatter(deaths_lh[1], deaths_lh[0], 300, c="r", marker="x")
        plt.axis("off")
        if plot_comparison:
            fig.add_subplot(rows, columns, 2)
            if app == False:
                if self.construction == "V":
                    plt.imshow(self.CP_comp.PixelMap, cmap="gray")
                else:
                    plt.imshow(self.CP_comp.ValueMap, cmap="gray")
            else:
                if self.construction == "V":
                    if self.filtration == "sublevel":
                        plt.imshow(
                            np.where(
                                self.CP_comp.PixelMap <= threshold_comp,
                                self.CP_comp.PixelMap,
                                0,
                            ),
                            cmap="gray",
                        )
                    else:
                        plt.imshow(
                            np.where(
                                self.CP_comp.PixelMap >= threshold_comp,
                                self.CP_comp.PixelMap,
                                0,
                            ),
                            cmap="gray",
                        )
            plt.imshow(CycleMap_comp_masked, cmap=colormap_comp, interpolation="none")
            if plot_birth_and_death:
                plt.scatter(births_comp[1], births_comp[0], 300, c="g", marker="*")
                plt.scatter(deaths_comp[1], deaths_comp[0], 300, c="r", marker="x")
            plt.axis("off")
            fig.add_subplot(rows, columns, 3)
        else:
            fig.add_subplot(rows, columns, 2)
        if app == False:
            if self.construction == "V":
                plt.imshow(self.CP_gt.PixelMap, cmap="gray")
            else:
                plt.imshow(self.CP_gt.ValueMap, cmap="gray")
        else:
            if self.construction == "V":
                if self.filtration == "sublevel":
                    plt.imshow(
                        np.where(
                            self.CP_gt.PixelMap <= threshold_gt, self.CP_gt.PixelMap, 0
                        ),
                        cmap="gray",
                    )
                else:
                    plt.imshow(
                        np.where(
                            self.CP_gt.PixelMap >= threshold_gt, self.CP_gt.PixelMap, 0
                        ),
                        cmap="gray",
                    )
        plt.imshow(CycleMap_gt_masked, cmap=colormap_gt, interpolation="none")
        if plot_birth_and_death:
            plt.scatter(births_gt[1], births_gt[0], 300, c="g", marker="*")
            plt.scatter(deaths_gt[1], deaths_gt[0], 300, c="r", marker="x")
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)
        plt.close(fig)
        return fig

    def plot_representative_cycles_app(
        self, dim=1, plot_birth_and_death=False, plot_comparison=False
    ):
        assert dim in [0, 1]

        def plot(match, threshold_lh, threshold_comp, threshold_gt):
            return self.plot_representative_cycles(
                dim=dim,
                matches=[match],
                threshold_lh=threshold_lh,
                threshold_comp=threshold_comp,
                threshold_gt=threshold_gt,
                plot_birth_and_death=plot_birth_and_death,
                plot_comparison=plot_comparison,
                app=True,
            )

        if len(self.matched[dim]) == 0:
            return "No matches found."

        elif len(self.matched[dim]) == 1:
            i = interact(
                plot,
                match=fixed(0),
                threshold_lh=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_lh"
                ),
                threshold_comp=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_comp"
                ),
                threshold_gt=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_gt"
                ),
            )
            if plot_comparison:
                return pn.Column(pn.Row(i[1]), pn.Row(i[0][0], i[0][1], i[0][2]))

            else:
                return pn.Column(pn.Row(i[1]), pn.Row(i[0][0], i[0][2]))

        else:
            i = pn.interact(
                plot,
                match=widgets.IntSlider(
                    start=0, end=len(self.matched[dim]) - 1, value=0, name="match"
                ),
                threshold_lh=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_lh"
                ),
                threshold_comp=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_comp"
                ),
                threshold_gt=widgets.FloatSlider(
                    start=0.0, end=1.0, step=0.001, value=0.5, name="threshold_gt"
                ),
            )
            if plot_comparison:
                return pn.Column(
                    pn.Row(i[0][0]), pn.Row(i[1]), pn.Row(i[0][1], i[0][2], i[0][3])
                )

            else:
                return pn.Column(
                    pn.Row(i[0][0]), pn.Row(i[1]), pn.Row(i[0][1], i[0][3])
                )
