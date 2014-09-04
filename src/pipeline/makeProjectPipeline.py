#!/usr/bin/env python
import numpy as np
import os
from itertools import product

#### modify these:
n = 150
idegree = np.arange(1.5, 3.0, .5)

##is this the right pI? or should I go higher? 
pIs = np.arange(0, .4, 0.05)
gEs = np.arange(1.5,3.0,.5)
gIs = (1.5,3.0,.5)
reps = range(3)
tf = 100000
projName = "blockrandom"
####

print "setting up project '" + projName + "'"
cwd = os.getcwd()
dataDir = os.path.join(os.environ['HOME'], 'Prebotc-graph-model', 'data', projName)
graphDir = os.path.join(dataDir, 'graphs')
oputDir = os.path.join(dataDir, 'output')
postDir = os.path.join(dataDir, 'post')
plotDir = os.path.join(dataDir, 'plots')
srcDir = os.path.join(os.environ['HOME'], 'Prebotc-graph-model', 'src')
paramFn = os.path.join(srcDir, 'model', 'param_files', 'BPR_syn.json')
cmdFn0 = os.path.join(srcDir, 'pipeline', projName + "_graphs")
cmdFn1 = os.path.join(srcDir, 'pipeline', projName + "_sweepfile")
cmdFn2 = os.path.join(srcDir, 'pipeline', projName + "_post")
cmdFn3 = os.path.join(srcDir, 'pipeline', projName + "_plot")
modelOpts = "-S -q"
#postOpts = "-f 60 --bin 40"

try:
    os.makedirs(graphDir)
    os.makedirs(oputDir)
    os.makedirs(postDir)
    os.makedirs(plotDir)
except OSError:
    pass
cmd0 = open(cmdFn0, 'w')
cmd1 = open(cmdFn1, 'w')
cmd2 = open(cmdFn2, 'w')
cmd3 = open(cmdFn3, 'w')

## loop through parameters
for (rep, idegree, pI, gE, gI) in product(reps, idegree, pIs, gEs, gIs):
    baseName = "er_n%0.1f_idegree%.2f_pI%0.2f_rep%d_gE%0.1f_gI%0.1f" % (n, idegree, pI, 
                                                                    rep, gE, gI)
    graphFn = os.path.join(graphDir, baseName + ".gml")
    simOutFn = os.path.join(oputDir, baseName + ".mat")
    postOutFn = os.path.join(postDir, baseName + "_post.mat")
    plotFn = os.path.join(plotDir, baseName)
    cmd = ' '.join(
        [os.path.join(srcDir, 'graphs', 'genPreBotBot.py'),
         str(n), str(idegree), graphFn, '-pI', str(pI), '-gE', str(gE),
         '-gI', str(gI)]) + '\n'
    cmd0.write(cmd)
    cmd = ' '.join(
        [os.path.join(srcDir, 'model', "runmodel.py"),
         paramFn, graphFn, simOutFn, '-tf', str(tf), modelOpts]) + "\n"
    cmd1.write(cmd)

    #####Stopped HERE
    cmd = ' '.join(
        [os.path.join(srcDir, 'joshpost', "doPost.py"),
         simOutFn, postOutFn]) + "\n"
    cmd2.write(cmd)
    cmd = " ".join(
	[os.path.join(srcDir, 'joshpost', 'doPlots.py'),
	postOutFn, plotFn]) + "\n"	
    cmd3.write(cmd)
cmd0.close()
cmd1.close()
cmd2.close()
cmd3.close()


