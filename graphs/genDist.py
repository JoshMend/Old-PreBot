#!/usr/bin/env python
import numpy as np
import networkx as nx
from respirnet import dist_prebot


n = 80
pI = .3
gE= 2.5
gI = 2.5
pTypes = [0,0.25,.45,.3]
g = dist_prebot(n,pTypes,pI,gE,gI)
nx.write_gml(g, 'distestGauss.gml')
