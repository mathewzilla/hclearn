import numpy as np
from location import *

def plotPlaceCells(hist, iCell, gridDict):
    
    T = len(hist.ecs)

    visits  = 0.00001+np.zeros((7,7))   #how many times agent has been here
    firings = np.zeros((7,7))   #how many times cell has fired   (avoid div by zero)

    #NB although we use a std CA3 class container for the CA3s, it has no real  semantics
    #the container was just build from a flat vector, whose weights were wake-sleep learned

    for t in range(0,T):
        loc = Location()
        loc.setGrids( hist.ecs[t].grids, gridDict )
        (x,y)=loc.getXY()
        
        visits[x][y] += 1
        
        if hist.ca3s[t].toVector()[iCell]:
            firings[x][y] += 1

    return (firings/visits, firings, visits)
            
    
