"""Build the fixed PP/VPP computation DAG used by iterative_critical_path.py.

This module is imported automatically; there is no standalone command to run.
Input event timestamps/durations are in microseconds; graph times are milliseconds.
"""
import collections
import heapq
import re

PAT = re.compile(r"(forward|backward)_step_mb(\d+)_vpp(\d+)$")


class Graph:
    def __init__(self, data, wrap=True, activation=True):
        self.data = data
        self.nodes = []
        self.by_rank = collections.defaultdict(list)
        self.key = {}
        for e in data["events"]:
            m = PAT.fullmatch(e["name"])
            if not m:
                continue
            i = len(self.nodes)
            n = dict(e, ts=e["ts"] / 1000, dur=e["dur"] / 1000,
                     direction=m[1], mb=int(m[2]), vpp=int(m[3]))
            self.nodes.append(n)
            self.by_rank[n["rank"]].append(i)
            k = (n["rank"], m[1], int(m[2]), int(m[3]))
            assert k not in self.key, ("duplicate", k)
            self.key[k] = i
        self.ranks = sorted(self.by_rank)
        self.pred = [dict() for _ in self.nodes]
        self.succ = [dict() for _ in self.nodes]
        def edge(u, v, kind):
            self.pred[v][u] = kind
            self.succ[u][v] = kind
        for rank, ids in self.by_rank.items():
            ids.sort(key=lambda i: (self.nodes[i]["ts"], i))
            for a, b in zip(ids, ids[1:]):
                edge(a, b, "local")
        for i, n in enumerate(self.nodes):
            r, d, m, v = n["rank"], n["direction"], n["mb"], n["vpp"]
            s = self.ranks.index(r)
            if d == "forward":
                if s:
                    edge(self.key[(self.ranks[s-1], d, m, v)], i, "forward")
                elif wrap and v > 1:
                    edge(self.key[(self.ranks[-1], d, m, v-1)], i, "forward_wrap")
            else:
                if s < len(self.ranks)-1:
                    edge(self.key[(self.ranks[s+1], d, m, v)], i, "backward")
                elif wrap and v < max(x["vpp"] for x in self.nodes):
                    edge(self.key[(self.ranks[0], d, m, v+1)], i, "backward_wrap")
                if activation:
                    edge(self.key[(r, "forward", m, v)], i, "activation")
        deg = [len(x) for x in self.pred]
        q = [i for i, d in enumerate(deg) if not d]
        heapq.heapify(q)
        self.order = []
        while q:
            u = heapq.heappop(q)
            self.order.append(u)
            for v in self.succ[u]:
                deg[v] -= 1
                if not deg[v]:
                    heapq.heappush(q, v)
        assert len(self.order) == len(self.nodes), ("cycle", len(self.order))

    def violations(self):
        out = []
        for v, preds in enumerate(self.pred):
            for u, kind in preds.items():
                gap = self.nodes[v]["ts"] - self.nodes[u]["ts"] - self.nodes[u]["dur"]
                if gap < -1e-6:
                    out.append(dict(u=u,v=v,kind=kind,gap_ms=gap))
        return out
