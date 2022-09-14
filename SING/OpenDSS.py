from common import Conductors, xfmr_info
import scipy.stats as stats
from fitter import Fitter
import networkx as nx
import pandas as pd
import os

class Model:
    def __init__(self, Buildings, Circuit, Model, File, HV, MV, PC, SC):
        self.Buildings = Buildings
        self.Circuit = Circuit
        self.Model = Model
        self.File = File
        self.HV = HV
        self.MV = MV
        self.PC = PC
        self.SC = SC

        print("Circuit : ", Circuit)
        print("Model : ", Model)
        print("File : ", File)
        print("HV : ", HV)
        print("MV : ", MV)
        print("Primary conductor : ", PC)
        print("Secondary conductor : ", SC)
        return

    def write_geometry(self, conductor):
        c = conductor.split("  ")[0]
        self.f.write("\n")
        line = f"New LineGeometry.{c} Nconds=4 Nphases=3 Units=m\n"
        line += f"~ Cond=1 Wire={c}  X=0.5   H=7.0\n"
        line += f"~ Cond=2 Wire={c}  X=0.5   H=6.5\n"
        line += f"~ Cond=3 Wire={c}  X=0.5   H=6.0\n"
        line += f"~ Cond=4 Wire={c}  X=0.5   H=5.5\n"
        line += f"~ Reduce=y\n"
        self.f.write(line)
        self.f.write("\n")
        return

    def write_conductor(self, conductor):
        self.f.write("\n")
        c = conductor.split("  ")[0]
        DIAM = Conductors[c]['DIAM']
        GMR = Conductors[c]['GMR']
        RES = Conductors[c]['RES']
        CAP = Conductors[c]['CAP']
        line = f"New wiredata.{c} GMRac={GMR} GMRunits=ft rac={RES}  runits=mi normamps={CAP} diam={DIAM} radunits=in"
        self.f.write(line)
        self.f.write("\n")
        return

    def write_lines(self):
        for u, v, a in self.Model.edges(data=True):
            N1 = self.nodes[u]['Type']
            if N1 == 'Primary':
                c = self.PC.split("  ")[0]
            elif N1 == 'Pole':
                c = self.SC.split("  ")[0]
            else:
                c = self.SC.split("  ")[0]
                u = u + "_lv"
            Camp = Conductors[c]['CAP']
            line = f"new line.{u}_{v} bus1={u} bus2={v} phases=3 length={a['length']} normamps={Camp} units=ft geometry={c}\n"
            self.f.write(line)
        return

    def get_samples(self, dist, params, nSamples):
        dist = getattr(stats, dist)
        return dist.rvs(*params, size=nSamples)

    def write_loads(self):
        loads = self.loadAMIdataframe
        load_peak = loads.sum(axis=1)
        idx = load_peak.idxmax()
        loads_KW = loads.loc[idx]

        f = Fitter(loads_KW.values)#, distributions=['gamma', 'rayleigh', 'uniform'])
        f.fit()
        fitting_results = f.summary()
        print(fitting_results)
        best_fit = fitting_results.index[0]
        best_fit_params = f.fitted_param[best_fit]

        self.f.write("\n")
        self.kW = 0
        loads_x = {}
        for u, d in self.Model.nodes(data=True):
            if d['Type'] == "Load":
                kW = self.get_samples(best_fit, best_fit_params, 1)[0]
                loads_x[int(u.replace("Load_", ""))] = kW
                self.f.write(f"New Load.{u} Bus1={u} Phases=3 kw={kW} kv=0.207 pf=0.95\n")
                self.kW += kW
        return loads_x

    def write_nodes(self):
        
        nodes_path = os.path.join(self.base_path, f"coordinates.dss")
        self.g = open(nodes_path, "w")
        nodes = {}
        for u, d in self.Model.nodes(data=True):
            nodes[u] = d
            self.g.write(f"{u} {d['pos'][0]} {d['pos'][1]}\n")
            if d['Type'] == 'Transformer':
                self.g.write(f"{u}_lv {d['pos'][0]} {d['pos'][1]}\n")
        self.g.close()
        return nodes

    def write_transformers(self, loads):
        self.f.write(f"\n")

        cluster_tr = {}
        for tr_num, load_list in self.Buildings.items():
            kW = 0
            for i in load_list:
                print(type(i))
                if i in loads:
                    kW += loads[i]
            cluster_tr[tr_num] = kW

            tr_size = 1
            for k in xfmr_info:
                if k > kW:
                    tr_size = k
                    break
            r = xfmr_info[tr_size]["R"]
            x = xfmr_info[tr_size]["Xhl"]
            nLd = xfmr_info[tr_size]["nLd"]

            tr = f"New Transformer.Transformer_{tr_num} Phases=3 Windings=2 Xhl={x}\n"
            tr += f"~ wdg=1 bus=Transformer_{tr_num}  conn=Delta kv={self.MV}  kva={tr_size} %r={r/2} %noloadloss={nLd}\n"
            tr += f"~ wdg=2 bus=Transformer_{tr_num}_lv  conn=Wye kv=0.207  kva={tr_size} %r={r/2}  %noloadloss={nLd}\n"
            self.f.write(tr)
        self.f.write(f"\n")
        return

    def Write(self, base_path, loadAMIdataframe):
        self.base_path = base_path
        self.loadAMIdataframe = loadAMIdataframe
        self.nodes = self.write_nodes()
        file_path = os.path.join(base_path, f"{self.File}.dss")
        self.f = open(file_path, "w")

        self.f.write(f"clear\n")
        self.f.write(f"New Circuit.{self.Circuit} bus1=node_sub pu=1.0 basekV={self.HV} phases=3\n\n")

        self.write_conductor(self.PC)
        self.write_conductor(self.SC)

        self.write_geometry(self.PC)
        self.write_geometry(self.SC)


        self.write_lines()
        loads = self.write_loads()
        self.write_transformers(loads)

        kW = sum(loads.values())
        tr_size = 1
        for k in xfmr_info:
            if k > kW:
                tr_size = k
                break
        r = xfmr_info[tr_size]["R"]
        x = xfmr_info[tr_size]["Xhl"]
        nLd = xfmr_info[tr_size]["nLd"]

        self.f.write(
            f"\nNew Transformer.xfmr1 Buses=[node_sub, substation] Conns=[Delta Wye] kVs=[{self.HV} {self.MV}] kVAs=[{tr_size} {tr_size}] XHL={x} %Rs=[{r/2} {r/2}] %noloadloss={nLd}\n")

        self.f.write(f"\nSet Voltagebases = [{self.HV}, {self.MV}, 0.207]\n\n")
        self.f.write(f"Calcvoltagebases\n\n")
        self.f.write(f"Buscoords coordinates.dss\n\n")
        self.f.write(f"solve\n\n")
        self.f.close()


        return