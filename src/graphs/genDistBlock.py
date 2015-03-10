#!/usr/bin/env python
import numpy as np
import networkx as nx
from respirnet import dist_block_prebot

n = 40 
pI = .3
gE= 2.5
gI = 2.5
pTypes = [0,0.25,.45,.3]

g = dist_block_prebot(n,pTypes,pI,gE,gI)
nx.write_gml(g, 'distblockprebot.gml')
