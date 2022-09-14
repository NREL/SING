from shapely.geometry import LineString, Point, MultiPoint
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import geopy.distance as Dist
import networkx as nx
import pandas as pd
import numpy as np
import os

class PrimaryModel:
    def __init__(self, rG, substations):
        self.rG = rG
        self.base_path = os.path.abspath(__file__).lower()
        self.substations = pd.DataFrame(substations)
        # print("Substations", substations)
        # substation_file = self.base_path.replace("primary.py", substations)
        # self.substations = pd.read_csv(substation_file, index_col=None, header=None)
        return

    def redistribute_vertices(self, line, distance):
        # Cuts a line in two at a distance from its starting point
        # This is taken from shapely manual
        if distance <= 0.0 or distance >= line.length:
            return [LineString(line)]
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [
                    LineString(coords[:i + 1]),
                    LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])
                ]

    def get_linesegments(self, line, n):
        points = MultiPoint([line.interpolate(i / n, normalized=True) for i in range(1, n)])
        return line.difference(points.buffer(1e-13))

    def build(self, offset_ft, dPole, length_thresh):
        self.G = nx.Graph()
        nline = 0
        edge_cnt = 0
        nodes = {"substation": (self.substations[0][0], self.substations[1][0])}

        for u, v in self.rG.edges():
            edge = self.rG[u][v]
            for k, v in edge.items():
                if 'geometry' in v:
                    length_m = v['length']
                    geometry = v['geometry']

                    segments = int(length_m / dPole) + 1
                    distances = np.linspace(0, 1, segments)
                    points = [geometry.interpolate(distance, normalized=True) for distance in distances]
                    names = [f"Line_{n+nline}" for n in range(len(points))]
                    nline += len(points)
                    edge_cnt += 1

                    for n, p in zip(names, points):
                        nodes[n] = (p.x, p.y)

        dist_matrix = self.distance_matrix(nodes)
        C = list(nodes.keys())
        graph = csr_matrix(dist_matrix)
        Tcsr = minimum_spanning_tree(graph)
        Mc = Tcsr.tocoo()
        Tcsr = Tcsr.toarray()
        for r, c in zip(Mc.row, Mc.col):
            d = Tcsr[r, c]
            u = C[r]
            v = C[c]
            settings = {'Type': 'Primary', 'pos': nodes[u], 'col': 'black'}
            self.G.add_node(u, **settings)
            settings = {'Type': 'Primary', 'pos': nodes[v], 'col': 'black'}
            self.G.add_node(v, **settings)
            attrs = {"length": Dist.distance(np.flip(nodes[u]), np.flip(nodes[v])).ft}
            self.G.add_edge(u,  v, **attrs)

        primary = self.reduce_graph(length_thresh)
        return primary

    def distance_matrix(self, nodes):
        import mpu
        d = np.zeros((len(nodes), len(nodes)))
        v = list(nodes.values())
        for i, p1 in enumerate(v):
            for j, p2 in enumerate(v):
                if i < j:
                    d[i, j] = mpu.haversine_distance(np.flip(p1), np.flip(p2))
                    d[j, i] = d[i, j]
        return d

    def reduce_graph(self, length_thresh):
        for u, v in self.G.edges():
            try:
                attrs = self.G[u][v]
                if attrs["length"] <= length_thresh:
                    self.G = nx.contracted_nodes(self.G, u, v, copy=True, self_loops=False)
            except:
                pass


        pos = nx.get_node_attributes(self.G, 'pos')

        cols = nx.get_node_attributes(self.G, 'col')
        nodes = pos
        D = self.distance_matrix(pos)
        C = list(pos.keys())
        Tcsr = minimum_spanning_tree(D)
        Mc = Tcsr.tocoo()
        Tcsr = Tcsr.toarray()

        self.G = nx.Graph()
        for r, c in zip(Mc.row, Mc.col):
            d = Tcsr[r, c]
            u = C[r]
            v = C[c]
            settings = {'Type': 'Primary', 'pos': nodes[u], 'col': 'black'}
            self.G.add_node(u, **settings)
            settings = {'Type': 'Primary', 'pos': nodes[v], 'col': 'black'}
            self.G.add_node(v, **settings)
            attrs = {"length": Dist.distance(np.flip(nodes[u]), np.flip(nodes[v])).ft}
            self.G.add_edge(u, v, **attrs)

        return self.G

# base_path = os.getcwd()
# G = nx.read_gpickle(os.path.join(base_path, 'data', "road.gpickle"))
#
# a = PrimaryModel(G, 'substations.txt')
# primary = a.build(1, 50.0, 170)
#
