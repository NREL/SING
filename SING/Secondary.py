from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn import preprocessing
from shapely.ops import nearest_points
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopy.distance as Dist
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import osmnx as ox
import pickle
import math
import os

# os.environ["OMP_NUM_THREADS"] = "1"
pd.options.mode.chained_assignment = None

class SecondaryModel:
    def __init__(self, buildings, substations):

        substations = pd.DataFrame(substations)

        substations["X"] = substations[0]
        substations["Y"] = substations[1]

        for idx, row in substations.iterrows():
            X, Y = self.LatLonToMeters(row[1], row[0])
            substations.loc[idx]["X"] = X
            substations.loc[idx]["Y"] = Y

        self.buildings = buildings
        self.substations = substations
        self.nSubstation = len(substations)

        return

    def LatLonToMeters(self, lat, lon):
        originShift = 2 * math.pi * 6378137 / 2.0
        mx = lon * originShift / 180.0
        my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        my = my * originShift / 180.0
        return mx, my


    def MetersToLatLon(self, mx, my):
        originShift = 2 * math.pi * 6378137 / 2.0
        lon = (mx / originShift) * 180.0
        lat = (my / originShift) * 180.0
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
        return (lon, lat)

    def build(self, buildingsPerCluster = [3], HousesPerPole=[2]):
        if self.nSubstation > 1:
            print(self.nSubstation)
            pass
        else:
            coordinates = {0: self.buildings}

        results = {}

        for subIndex, bData in coordinates.items():
            nBuildings = len(bData)
            print(f"Number of buildings: {nBuildings}")
            XY = bData[["X", "Y"]].values
            for housePerPole in HousesPerPole:
                for buildingPerCluster in buildingsPerCluster:
                    nClusters = int(nBuildings / buildingPerCluster)
                    print(f"Number of clusters: {nClusters}")
                    min_max_scaler = preprocessing.MinMaxScaler()
                    x_scaled = min_max_scaler.fit_transform(XY)

                    clusters = KMeans(n_clusters=nClusters, random_state=0).fit(x_scaled)
                    bData["Cluster"] = clusters.labels_
                    bData["xfmrCluster"] = clusters.labels_
                    ax = bData.plot.scatter(x='X', y='Y', c='black')

                    self.infrastructure = pd.DataFrame(columns=["X", "Y", "Type", "TypeID", "PoleCluster", "xfmrCluster"])
                    for i in range(nClusters):
                        filteredData = bData[bData["Cluster"] == i]
                        filteredDataMean = filteredData[["X", "Y"]].mean()

                        row = pd.DataFrame(
                            [[filteredDataMean["X"], filteredDataMean["Y"], "Transformer", 0, None, i]],
                            columns=["X", "Y", "Type", "TypeID", "PoleCluster", "xfmrCluster"]
                        )
                        self.infrastructure = self.infrastructure.append(row, ignore_index=True)

                        fXY = filteredData[["X", "Y"]].values
                        nfilteredData = len(filteredData)
                        nPoleClusters = int(nfilteredData / housePerPole)
                        if nPoleClusters:

                            scaler = preprocessing.MinMaxScaler()
                            fXY_scaled = scaler.fit_transform(fXY)
                            pClusters = KMeans(n_clusters=nPoleClusters, random_state=0).fit(fXY_scaled)

                            filteredData.loc[:,"PoleCluster"] = pClusters.labels_
                            for j in range(nPoleClusters):
                                PoleData = filteredData[filteredData["PoleCluster"] == j]
                                indices = PoleData.index
                                bData.loc[indices, "PoleCluster"] = [j for n in range(len(PoleData))]
                                PoleDataMean = PoleData[["X", "Y"]].mean()

                                row = pd.DataFrame(
                                    [[PoleDataMean["X"], PoleDataMean["Y"], "Pole", 1, j, i]],
                                    columns=["X", "Y", "Type", "TypeID", "PoleCluster", "xfmrCluster"]
                                )
                                self.infrastructure = self.infrastructure.append(row, ignore_index=True)
                        else:
                            pass

                    self.save_df(self.infrastructure, f"infrastructure_{nClusters}_{housePerPole}.csv")
                    self.save_df(bData, f"building_data_{nClusters}_{housePerPole}.csv")

                    results[f"{nClusters}_{housePerPole}"] = {
                        "buildings":  bData.copy(),
                        "infrastructure": self.infrastructure.copy(),
                    }

                    # self.infrastructure.plot.scatter(ax=ax, x='X', y='Y', c='TypeID', colormap="Spectral")
                    # plt.show()
                    # quit()
        return results


    def save_df(self, DF, fname):
        xLat = []
        xLong = []
        for X, Y in DF[['X', 'Y']].values:
            lat, long = self.MetersToLatLon(X, Y)
            xLat.append(lat)
            xLong.append(long)
        DF["Lat"] = xLat
        DF["Long"] = xLong
        #DF.to_csv(fname)
        #print(f"File saved: {fname}")



    def move_point(self, P1, P2):
        thresh = 10
        D = Dist.distance((P1.y, P1.x), (P2.y, P2.x)).km * 1000
        a = thresh / D
        x = a * P2.x + (1 - a) * P1.x
        y = a * P2.y + (1 - a) * P1.y
        Dnew = Dist.distance((y, x), (P1.y, P1.x)).km * 1000
        return x, y

    def get_new_point(self, edge, i, p):
        P1, P2 = nearest_points(edge[i]['geometry'], p)
        x, y = self.move_point(P1, P2)
        xm, ym = self.LatLonToMeters(y, x)
        return x, y, xm, ym


    def centroid(self, infrastructure, roadGraph, alpha):
        G_proj = ox.project_graph(roadGraph)
        geom = gpd.points_from_xy(infrastructure['Lat'], infrastructure['Long'])
        gdf = gpd.GeoDataFrame(infrastructure, geometry=geom, crs='epsg:4326').to_crs(G_proj.graph['crs'])
        ne = ox.nearest_edges(G_proj, X=gdf['geometry'].x, Y=gdf['geometry'].y)
        nearest_edge = pd.DataFrame(ne, columns=["u", "v", "x"])

        infrastructure["u"] = nearest_edge["u"]
        infrastructure["v"] = nearest_edge["v"]
        infrastructure["x"] = nearest_edge["x"]

        infrastructure["mX"] = nearest_edge["u"]
        infrastructure["mY"] = nearest_edge["v"]
        infrastructure["mLat"] = 0.0
        infrastructure["mLong"] = 0.0

        for c in list(set(infrastructure["xfmrCluster"].tolist())):
            fInfra = infrastructure[infrastructure["xfmrCluster"] == c]

            trData = fInfra[fInfra["Type"] == "Transformer"]

            xmXFMR = trData["X"].values[0]
            ymXFMR = trData["Y"].values[0]
            longXFMR = trData["Long"].values[0]
            latXFMR = trData["Lat"].values[0]
            index = trData.index[0]

            infrastructure.at[index, "mX"] = xmXFMR
            infrastructure.at[index, "mY"] = ymXFMR
            infrastructure.at[index, "mLat"] = latXFMR
            infrastructure.at[index, "mLong"] = longXFMR

            Poles = fInfra[fInfra["Type"] == "Pole"]

            for idx, row in Poles.iterrows():
                xmPole = row["X"]
                ymPole = row["Y"]
                latPole = row["Lat"]
                longPole = row["Long"]

                r = [
                    idx,
                    alpha * xmXFMR + (1 - alpha) * xmPole,
                    alpha * ymXFMR + (1 - alpha) * ymPole,
                    alpha * latXFMR + (1 - alpha) * latPole,
                    alpha * longXFMR + (1 - alpha) * longPole
                ]

                infrastructure.at[idx, "mX"] = alpha * xmXFMR + (1 - alpha) * xmPole
                infrastructure.at[idx, "mY"] = alpha * ymXFMR + (1 - alpha) * ymPole
                infrastructure.at[idx, "mLat"] = alpha * latXFMR + (1 - alpha) * latPole
                infrastructure.at[idx, "mLong"] = alpha * longXFMR + (1 - alpha) * longPole

        return infrastructure



    def allign_infrastructure_to_road(self, infrastructure, roadGraph):

        G_proj = ox.project_graph(roadGraph)
        geom = gpd.points_from_xy(infrastructure['Lat'], infrastructure['Long'])
        gdf = gpd.GeoDataFrame(infrastructure, geometry=geom, crs='epsg:4326').to_crs(G_proj.graph['crs'])
        ne = ox.nearest_edges(G_proj, X=gdf['geometry'].x, Y=gdf['geometry'].y)
        nearest_edge = pd.DataFrame(ne, columns=["u", "v", "x"])


        infrastructure["u"] = nearest_edge["u"]
        infrastructure["v"] = nearest_edge["v"]
        infrastructure["x"] = nearest_edge["x"]

        infrastructure["mX"] = nearest_edge["u"]
        infrastructure["mY"] = nearest_edge["v"]
        infrastructure["mLat"] = nearest_edge["u"]
        infrastructure["mLong"] = nearest_edge["v"]

        r = []
        for idx, row in infrastructure.iterrows():
            X, Y, u, v = row[["X", "Y", "u", "v"]].tolist()
            lat, long = self.MetersToLatLon(X, Y)
            p = Point(lat, long)
            edge = roadGraph.get_edge_data(u, v)
            if 'geometry' in edge[0]:
                result = self.get_new_point(edge, 0, p)
            elif 1 in edge and 'geometry' in edge[1]:
                result = self.get_new_point(edge, 1, p)
            else:
                result = [None, None, None, None]
            r.append(result)

        R = pd.DataFrame(r, columns=["mLat", "mLong", "mX", "mY",])

        infrastructure["mLat"] = R["mLat"]
        infrastructure["mLong"] = R["mLong"]
        infrastructure["mX"] = R["mX"]
        infrastructure["mY"] = R["mY"]

        return infrastructure

    def distance_matrix(self, points):
        d = np.zeros((len(points), len(points)))
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                d[i, j] = Dist.distance(np.flip(p1), np.flip(p2)).ft
        return d

    def create_secondaries(self, infrastructure, buildings, B, H):
        G = nx.Graph()
        clusters = list(set(buildings["Cluster"].tolist()))
        tr_mapping = {}
        for c in clusters:
            print(f"Percentage complete: {clusters.index(c)/len(clusters) * 100}")
            data = pd.DataFrame(columns=["Long", "Lat", "Name", "Type"])
            fInfra = infrastructure[infrastructure["xfmrCluster"] == c]
            fBuild = buildings[buildings["xfmrCluster"] == c]

            for idx, row in fInfra.iterrows():
                dRow = pd.DataFrame([[
                    row["mLat"] if not pd.isnull(row["mLat"]) else row["Lat"],
                    row["mLong"] if not pd.isnull(row["mLong"]) else row["Long"],
                    idx,
                    row["Type"],
                    row["xfmrCluster"],
                    row["PoleCluster"],
                ]],columns=["Long", "Lat", "Name", "Type", "T", "P"])
                data = data.append(dRow)

            for idx, row in fBuild.iterrows():
                dRow = pd.DataFrame([[
                    row["Lat"],
                    row["Long"],
                    idx,
                    "Load",
                    row["xfmrCluster"],
                    row["PoleCluster"],
                ]], columns=["Long", "Lat", "Name", "Type", "T", "P"])
                data = data.append(dRow)

            data = data.drop_duplicates(subset=["Long", "Lat"], keep='last')
            data.index = range(len(data))

            poleClusters = set(data["P"].tolist())
            # secondary edges
            sec_nodes = data[data['Type'] != 'Load']
            tr_exists = "Transformer" in sec_nodes.values
            pole_exists = "Pole" in sec_nodes.values
            if tr_exists:
                points = sec_nodes[["Long", "Lat"]].values
                distances = self.distance_matrix(points)
                C = sec_nodes["Type"] + "_" + sec_nodes["Name"].astype(str)
                C = C.tolist()
                graph = csr_matrix(distances)
                Tcsr = minimum_spanning_tree(graph)
                Mc = Tcsr.tocoo()
                Tcsr = Tcsr.toarray()
                for r, c in zip(Mc.row, Mc.col):
                    d = Tcsr[r, c]
                    u = C[r]
                    v = C[c]
                    for x in [u, v]:
                        t, n = x.split("_")
                        node = sec_nodes[(sec_nodes['Type'] == t) & (sec_nodes['Name'] == int(n))]

                        if node['Type'].values[0] == "Load":
                            color = "blue"
                        elif node['Type'].values[0] == "Pole":
                            color = "green"
                        else:
                            color = "red"

                        settings = {
                            'Type': node['Type'].values[0],
                            'pos': (node['Long'].values[0], node['Lat'].values[0]),
                            "col": color
                        }
                        G.add_node(x, **settings)
                    attrs = {"length": d}
                    G.add_edge(u, v, **attrs)

            elif pole_exists:
                data["Type"][0] = "Transformer"

            trx = data[data["Type"] == "Transformer"]
            ldx = data[data["Type"] == "Load"]

            if not trx.empty:
                tr_name = trx["Name"].values[0]
                ld_names = ldx["Name"].tolist()
                tr_mapping[tr_name] = ld_names


            # build service drop edges
            for P in poleClusters:
                if not pd.isnull(P):
                    pole_data = data[data["P"] == P]
                    loads = pole_data[pole_data['Type'] == 'Load']
                    poles = pole_data[pole_data['Type'] != 'Load']
                    if not poles.empty:
                        #creating nodes
                        for idx, row in pole_data.iterrows():
                            settings = row[['Type']].to_dict()
                            settings["pos"] = (row["Long"], row["Lat"])
                            if row["Type"] == "Load":
                                settings["col"] = "blue"
                            elif row["Type"] == "Pole":
                                settings["col"] = "green"
                            else:
                                settings["col"] = "red"
                            G.add_node(f"{row['Type']}_{row['Name']}", **settings)

                        u = f"{poles['Type'].values[0]}_{poles['Name'].values[0]}"
                        p1 = (poles['Lat'].values[0], poles['Long'].values[0])
                        for idx, row in loads.iterrows():
                            v = f"{row['Type']}_{row['Name']}"
                            p2 = (row['Lat'], row['Long'])
                            attrs = {"length": Dist.distance(p1, p2).ft}
                            G.add_edge(u, v, **attrs)

        xfmrs = {}
        for n, data in G.nodes(data=True):
            if data["Type"] == "Transformer":
                xfmrs[n] = data["pos"]


        return G, xfmrs, tr_mapping

