import numpy as np
import pygimli as pg
import jointtools.meshinfo as mi 
from functools import partial
from jointtools.spatialgradient import spatialgradient as sG
from pygimli.frameworks import MeshModelling

class JItools(MeshModelling):
    """Tool Box for not directly fwi related things"""
    
    def __init__(self, fopList, dataList, errorList, meshList=None, **kwargs):
        """Create an instance for toolbox.
        """
        super().__init__()
        self.fops = fopList
        self.dataList = dataList
        self.errorList = errorList
        self.meshs = meshList
        if meshList:
            self.setMesh(self.meshs)
            
            
        self.modelTrans = pg.trans.TransLogLU()
        self.fops[0].regionManager()
        self.setRegionManager(self.fops[1].regionManagerRef())
    
    
    def setMesh(self, meshList):
        self.meshs = meshList
        self.lens=[]
        for i,f in enumerate(self.fops):
            f.setMesh(meshList[i])
            self.lens.append(len(f.paraDomain.cells()))
        self.mesh = self.fops[0].paraDomain
        
        
    def setData(self, dataList):
        for i,f in enumerate(self.fops):
            f.setData(dataList[i])
            
            
    def response(self, model):
        resp = np.array([])
        resps = []
        for i,f in enumerate(self.fops):
            r = np.array(f.response(np.split(model, 2)[i]))
            resp = np.concatenate((resp, r))
            resps.append(r)
                
        return resp
    
    
    def createStartModel(self):
        mod = np.array([])
        for i,f in enumerate(self.fops):
            mod = np.concatenate((mod, np.array(f.createStartModel(self.dataList[i]))))
        return mod
    
    @property
    def parameterCount(self):
        pc = 0
        for i,f in enumerate(self.fops):
            pc += f.regionManager().parameterCount()
        self.model = np.zeros(pc)
        return pc
    
    def constraints(self):
        for i,f in enumerate(self.fops):
            f.createConstraints()
        cs = pg.matrix.BlockMatrix()
        cs.addMatrix(self.fops[0].constraints(), 0, 0)
        cs.addMatrix(
            self.fops[1].constraints(), 
            self.fops[0].constraints().shape[0], 
            self.fops[0].constraints().shape[1]
            )
        return cs
    
    def createJacobian(self, model):
        self.model = model
        for i,f in enumerate(self.fops):
            f.createJacobian(np.split(model, 2)[i])
            
    def jacobian(self):
        j = pg.matrix.BlockMatrix()
        j.addMatrix(self.fops[0].jacobian(), 0, 0)
        j.addMatrix(
            self.fops[1].jacobian(), 
            self.fops[0].jacobian().shape[0], 
            self.fops[0].jacobian().shape[1]
            )
        return j
        
    
    
class crossGradient(JItools):
    """Cross gradient manager"""
    
    def __init__(self, alpha, dist=5, smooth=False, **kwargs):
        super().__init__(**kwargs)
        
        self.alpha = alpha
        self.dist = dist
        self.smooth = smooth
        
    def cross_gradient_classic(self, m1, m2):
        """Calculates the magnitudes of the cross-gradient vectors at cell centers."""
        node1 = pg.meshtools.cellDataToNodeData(self.mesh, m1)
        node2 = pg.meshtools.cellDataToNodeData(self.mesh, m2)

        g1 = -pg.solver.grad(self.mesh, node1)
        g2 = -pg.solver.grad(self.mesh, node2)

        
        return np.cross(g1, g2)[:,2]
        
    def cross_gradient_smooth(self, m1, m2, dist=5, TO=1):
        """Calculates the magnitudes of the cross-gradient vectors at cell centers."""
        
        self.m1 = m1
        self.m2 = m2
        neighbour_function = partial(
            mi.distance_to_neighbour_list_for_cell, 
            dist=dist, mesh=self.mesh
            ) #important to define the neighbour function for specific cells
        mesh_mi = mi.MeshInfo(
            mesh=self.mesh, 
            initialise_gn2=True, neighbour_function=neighbour_function)
        g1 = sG.calculate_spatial_gradient(
            model=self.m1, mesh_info=mesh_mi, taylor_order=TO)
        g2 = sG.calculate_spatial_gradient(
            model=self.m2, mesh_info=mesh_mi, taylor_order=TO)
        
        return np.cross(g1, g2)
    
    def constraints(self):
        C_np= pg.utils.gmat2numpy(super().constraints())
        if self.smooth == True:
            cg = self.cross_gradient_smooth(
                self.fops[0].modelTrans(np.split(self.model,2)[0]), 
                self.fops[1].modelTrans(np.split(self.model,2)[1]), 
                self.dist, TO=1
                )
        else:
            cg = self.cross_gradient_classic(                
                self.fops[0].modelTrans(np.split(self.model,2)[0]), 
                self.fops[1].modelTrans(np.split(self.model,2)[1])
                )
        cg = np.concatenate([cg, -cg])
        cg_mat = pg.utils.numpy2gmat(C_np@np.diag(self.alpha*cg))
        cs = pg.matrix.BlockMatrix()
        cs.addMatrix(super().constraints(), 0, 0)
        cs.addMatrix(
            cg_mat, 
            super().constraints().shape[0], 
            0
            )
        return cs