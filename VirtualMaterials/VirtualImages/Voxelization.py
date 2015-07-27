# -*- coding: utf-8 -*-

import numpy as np
import vtk
import math
import VirtualMaterials as vmat
#from vtk.util import numpy_support


#--------------------------------------------------------------------
#      Voxelization
#--------------------------------------------------------------------

    
#--------------------------------------------------------------------
def Voxelize(vtkPolyDataObject,gridX,gridY,gridZ,raydirection='xz'):
    #Voxelize the object on a window adapted to its bounds. The windows will 
    #be chosed to be a subsample or an extention of the whole image voxel 
    #gridX,Y,Z. This function uses VTK VoxelModel to voxelize the surface, 
    #then the FillInside function to fill the inside.

    #Prepare a subwindows zooming on the object
    subWindowBound=vtkPolyDataObject.GetBounds()    
    nVoxSubImage,boundSubgrid,gridRelativePosition = GetSubWindowInformation(
                                                        subWindowBound,gridX,gridY,gridZ)
    
    
    subImage=VoxelizeRayTracing(vtkPolyDataObject,nVoxSubImage,boundSubgrid,raydirection)
    

    #Get back to the original window
    nVoxImage = (len(gridX)-1,len(gridY)-1,len(gridZ)-1)    
    wholeImage = InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition)
    
    return wholeImage.astype(np.bool)


#-----------------------------------------------------------------------------
def VoxelizeRayTracing(vtkPolyDataObject,nVoxSubImage,boundSubgrid,raydirection='z'):

    #Count the number of voxels in each direction:

#    points = numpy_support.vtk_to_numpy(vtkPolyDataObject.GetPoints().GetData())
#    polys = numpy_support.vtk_to_numpy(vtkPolyDataObject.GetPolys().GetData())
#    
#    nTriangle=len(polys)/4
#    meshXYZ=np.zeros((nTriangle,3,3),dtype=np.float)
#    for iTri in range(nTriangle):
#        for iVert in range(3):
#                assert polys[4*iTri]==3
#                meshXYZ[iTri,:,iVert]=points[polys[4*iTri+iVert+1]]
    
    gridCOx=np.linspace(boundSubgrid[0],boundSubgrid[1],nVoxSubImage[0])
    gridCOy=np.linspace(boundSubgrid[2],boundSubgrid[3],nVoxSubImage[1])
    gridCOz=np.linspace(boundSubgrid[4],boundSubgrid[5],nVoxSubImage[2])

    voxcountX,voxcountY,voxcountZ = nVoxSubImage[0],nVoxSubImage[1],nVoxSubImage[2]

    
    # Prepare logical array to hold the voxelised data:
    gridOUTPUT = np.zeros((voxcountX,voxcountY,voxcountZ,len(raydirection)),dtype=np.bool)
    countdirections = 0;
    
    if raydirection.find('x')>-1:
      countdirections = countdirections + 1
      #rotatedMesh = meshXYZ[:,[1,2,0],:]
      rotationCenter=(0,0,0)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateY(vtkPolyDataObject,rotationCenter,-90)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateZ(rotatedMesh,rotationCenter,-90)
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( 
          VoxelizeRayTracingZDirectionVTK(rotatedMesh,gridCOy,gridCOz,gridCOx),
                                  axes=[2,0,1] )
    
    #TODO : debug y direction
    if raydirection.find('y')>-1:    
      countdirections = countdirections + 1
      #rotatedMesh = meshXYZ[:,[2,0,1],:]
      rotationCenter=(0,0,0)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateZ(vtkPolyDataObject,rotationCenter,-90)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateX(rotatedMesh,rotationCenter,-90)
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( 
          VoxelizeRayTracingZDirectionVTK(rotatedMesh,gridCOz,gridCOx,gridCOy),
                                  axes=[1,2,0] )
    
    if raydirection.find('z')>-1:
      countdirections = countdirections + 1
      gridOUTPUT[:,:,:,countdirections-1] = VoxelizeRayTracingZDirectionVTK(
                                  vtkPolyDataObject,gridCOx,gridCOy,gridCOz)
    
    # Combine the results of each ray-tracing direction:
    if len(raydirection)>1:
        gridOUTPUT = (np.sum(gridOUTPUT,axis=3)== len(raydirection)) # >=len(raydirection)/2.0)
    else:
        gridOUTPUT = gridOUTPUT[:,:,:,0]
        
    return gridOUTPUT
    


#-----------------------------------------------------------------------------
def VoxelizeRayTracingZDirectionVTK(polydata,gridCOx,gridCOy,gridCOz):
    
    #Identify the min and max x,y coordinates (cm) of the mesh:
#    meshXmin = meshXYZ[:,0,:].min()
#    meshXmax = meshXYZ[:,0,:].max()
#    meshYmin = meshXYZ[:,1,:].min()
#    meshYmax = meshXYZ[:,1,:].max()
#    meshZmin = meshXYZ[:,2,:].min()
#    meshZmax = meshXYZ[:,2,:].max()
    
    meshXmin,meshXmax,meshYmin,meshYmax,meshZmin,meshZmax = polydata.GetBounds()
    assert (meshXmin<meshXmax and meshYmin<meshYmax and meshZmin<meshZmax)
    
    #Identify the min and max x,y coordinates (pixels) of the mesh:
    meshXminp = np.nonzero(abs(gridCOx-meshXmin)==min(abs(gridCOx-meshXmin)))[0][0]
    meshXmaxp = np.nonzero(abs(gridCOx-meshXmax)==min(abs(gridCOx-meshXmax)))[0][0]
    meshYminp = np.nonzero(abs(gridCOy-meshYmin)==min(abs(gridCOy-meshYmin)))[0][0]
    meshYmaxp = np.nonzero(abs(gridCOy-meshYmax)==min(abs(gridCOy-meshYmax)))[0][0]
        
    #meshXYZmin = np.min(meshXYZ,axis=2)
    #meshXYZmax = np.max(meshXYZ,axis=2)    
        
    sampleDimensions=(len(gridCOx),len(gridCOy),len(gridCOz))
    image=np.zeros(sampleDimensions,dtype=np.bool)
    #Identify the min and max x,y,z coordinates of each facet:
    
    
    correctionLIST = []    
    
    #Construct octree of the mesh for fast access to its geometry
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(polydata)
    obbTree.BuildLocator()
    
    for loopY in xrange(meshYminp,meshYmaxp+1):
        for loopX in xrange(meshXminp,meshXmaxp+1):
            
            epsilon=(meshZmax-meshZmin)/10
            pSource=[gridCOx[loopX],gridCOy[loopY],meshZmax+epsilon]
            pTarget=[gridCOx[loopX],gridCOy[loopY],meshZmin-epsilon]
            pointsIntersection=VTKRayCasting(obbTree,pSource,pTarget)
            
            
            if len(pointsIntersection)>0: 

              gridCOzCROSS=np.asarray([pointsIntersection[i][2] for i in range(len(pointsIntersection))])
                
              #Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
              gridCOzCROSS = gridCOzCROSS[ np.logical_and(
                  np.greater_equal(gridCOzCROSS,(meshZmin-1e-12)*np.ones(gridCOzCROSS.shape)),
                  np.less_equal(gridCOzCROSS,(meshZmax+1e-12)*np.ones(gridCOzCROSS.shape))) ]
          
              #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
              gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
              gridCOzCROSS = np.unique(gridCOzCROSS)
    
              if gridCOzCROSS.size%2 == 0: 
                  for loopASSIGN in range(gridCOzCROSS.size/2):
                      voxelsINSIDE = np.logical_and(
                                    np.greater(gridCOz,gridCOzCROSS[2*loopASSIGN]), 
                                    np.less(gridCOz,gridCOzCROSS[2*loopASSIGN+1]))
                                    
                      image[loopX,loopY,voxelsINSIDE] = 1
              elif len(gridCOzCROSS)>0:
                  correctionLIST.append([loopX,loopY])
        
    
    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
    #For rays where the voxelisation did not give a clear result, the ray is
    #computed by interpolating from the surrounding rays.    

    image=InterpolateRemainingVoxels(correctionLIST,sampleDimensions,image)
    
    return image





#-----------------------------------------------------------------------------
def VTKRayCasting(polydataObbTree,pSource,pTarget):
#https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/


#    pSource = [loopX, loopY, +10]
#    pTarget = [loopX, loopY, -10]
    
    pointsVTKintersection = vtk.vtkPoints()
    code = polydataObbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, None)

    #check code
#    if code==0:
#        1
#    elif code==-1:
#        1
#    elif code==1:
#        1
   #"Take the passed line segment and intersect it with the data set. This method 
   #assumes that the data set is a vtkPolyData that describes a closed surface, 
   #and the intersection points that are returned in ‘points’ alternate between 
   #entrance points and exit points. The return value of the function is 0 if no 
   #intersections were found, –1 if point ‘a0’ lies inside the closed surface, or 
   #+1 if point ‘a0’ lies outside the closed surface. Either ‘points’ or ‘cellIds’ 
   #can be set to NULL if you don’t want to receive that information."

    pointsVTKIntersectionData = pointsVTKintersection.GetData()
    nPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
#   pointsIntersection = []
#    for idx in range(nPointsVTKIntersection):
#        _tup = pointsVTKIntersectionData.GetTuple3(idx)
#        pointsIntersection.append(_tup)

    pointsIntersection=[pointsVTKIntersectionData.GetTuple3(idx) 
                                for idx in range(nPointsVTKIntersection)]
    
    return pointsIntersection


#-----------------------------------------------------------------------------
def InterpolateRemainingVoxels(correctionLIST,sampleDimensions,image):
    
    countCORRECTIONLIST = len(correctionLIST)
    
    if countCORRECTIONLIST>0:
        
        #If necessary, add a one-pixel border around the x and y edges of the
        #array.  This prevents an error if the code tries to interpolate a ray at
        #the edge of the x,y grid.
        cond0 = min([correctionLIST[i][0] for i in range(len(correctionLIST))])==0
        cond1 = max([correctionLIST[i][0] for i in range(len(correctionLIST))])==sampleDimensions[0]-1
        cond2 = min([correctionLIST[i][1] for i in range(len(correctionLIST))])==0
        cond3 = max([correctionLIST[i][1] for i in range(len(correctionLIST))])==sampleDimensions[1]-1
    
        if cond0 or cond1 or cond2 or cond3:
            image = np.hstack( (np.zeros((sampleDimensions[0],1,sampleDimensions[2])),
                                image,np.zeros((sampleDimensions[0],1,sampleDimensions[2]))))
            image = np.vstack( (np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2])),
                                image,np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2]))))
            correctionLIST = [ [correctionLIST[i][0]+1,correctionLIST[i][1]+1] 
                                            for i in range(len(correctionLIST)) ]
        
        for loopC in range(countCORRECTIONLIST):
            voxelsforcorrection = np.squeeze( np.sum( [ 
                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]-1,:],
                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1],:],
                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]+1,:],
                image[correctionLIST[loopC][0],correctionLIST[loopC][1]-1,:],
                image[correctionLIST[loopC][0],correctionLIST[loopC][1]+1,:],
                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]-1,:],
                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1],:],
                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]+1,:],
                ], axis=0 ) )
            voxelsforcorrection = (voxelsforcorrection>=4)
            image[correctionLIST[loopC][0],correctionLIST[loopC][1],voxelsforcorrection] = 1
        
    #Remove the one-pixel border surrounding the array, if this was added
    #previously.
    if image.shape[0]>sampleDimensions[0] or image.shape[1]>sampleDimensions[1]:
        image = image[1:-1,1:-1,:]
    return image





#
#
#
##-----------------------------------------------------------------------------
#def VoxelizeRayTracingZDirection(meshXYZ,gridCOx,gridCOy,gridCOz):
#
#    #% Loop through each x,y pixel.
#    #% The mesh will be voxelised by passing rays in the z-direction through
#    #% each x,y pixel, and finding the locations where the rays cross the mesh.
#    #This function is inspired by some parts of the Matlab file exchange function
#    #VOXELISE (AUTHOR Adam H. Aitkenhead, The Christie NHS Foundation Trust) 
#    
#    
#    #Identify the min and max x,y coordinates (cm) of the mesh:
#    meshXmin = meshXYZ[:,0,:].min()
#    meshXmax = meshXYZ[:,0,:].max()
#    meshYmin = meshXYZ[:,1,:].min()
#    meshYmax = meshXYZ[:,1,:].max()
#    meshZmin = meshXYZ[:,2,:].min()
#    meshZmax = meshXYZ[:,2,:].max()
#    
#    #Identify the min and max x,y coordinates (pixels) of the mesh:
#    meshXminp = np.nonzero(abs(gridCOx-meshXmin)==min(abs(gridCOx-meshXmin)))[0][0]
#    meshXmaxp = np.nonzero(abs(gridCOx-meshXmax)==min(abs(gridCOx-meshXmax)))[0][0]
#    meshYminp = np.nonzero(abs(gridCOy-meshYmin)==min(abs(gridCOy-meshYmin)))[0][0]
#    meshYmaxp = np.nonzero(abs(gridCOy-meshYmax)==min(abs(gridCOy-meshYmax)))[0][0]
#        
#    meshXYZmin = np.min(meshXYZ,axis=2)
#    meshXYZmax = np.max(meshXYZ,axis=2)    
#        
#    sampleDimensions=(len(gridCOx),len(gridCOy),len(gridCOz))
#    image=np.zeros(sampleDimensions,dtype=np.bool)
#    #Identify the min and max x,y,z coordinates of each facet:
#    
#    
#    correctionLIST = []    
#    
#    
#    for loopY in range(meshYminp,meshYmaxp+1):
#    
#        #- 1a - Find which mesh facets could possibly be crossed by the ray:
#        possibleCROSSLISTy = np.nonzero(np.logical_and(
#                                      np.less_equal(meshXYZmin[:,1],gridCOy[loopY]), 
#                                      np.greater_equal(meshXYZmax[:,1],gridCOy[loopY]) ))[0]
#        
#        for loopX in range(meshXminp,meshXmaxp+1):
#        
#            #- 1b - Find which mesh facets could possibly be crossed by the ray:
#            possibleCROSSLIST = possibleCROSSLISTy[ np.logical_and(
#                       np.less_equal(meshXYZmin[possibleCROSSLISTy,0],gridCOx[loopX]),
#                       np.greater_equal(meshXYZmax[possibleCROSSLISTy,0],gridCOx[loopX]))]
#        
#            if len(possibleCROSSLIST)>0:  #Only continue the analysis if some nearby 
#                                          #facets were actually identified
#                  
#        #      % - 2 - For each facet, check if the ray really does cross the facet rather than just passing it close-by:
#                  
#        #      % GENERAL METHOD:
#        #      % A. Take each edge of the facet in turn.
#        #      % B. Find the position of the opposing vertex to that edge.
#        #      % C. Find the position of the ray relative to that edge.
#        #      % D. Check if ray is on the same side of the edge as the opposing vertex.
#        #      % E. If this is true for all three edges, then the ray definitely passes through the facet.
#        #      %
#        #      % NOTES:
#        #      % A. If a ray crosses exactly on a vertex:
#        #      %    a. If the surrounding facets have normal components pointing in the same (or opposite) direction as the ray then the face IS crossed.
#        #      %    b. Otherwise, add the ray to the correctionlist.
#              
#              facetCROSSLIST = []   #Prepare to record all facets which are crossed by the ray.
#                                    #This array is built on-the-fly, but since
#                                    #it ought to be relatively small (typically a list of <10) 
#                                    #should not incur too much of a speed penalty.
#              
#        #      %----------
#        #      % - 1 - Check for crossed vertices:
#        #      %----------
#              possibleCROSSLIST, correctionLIST,facetCROSSLIST = CheckForCrossedVertices(
#                                  meshXYZ,
#                                  possibleCROSSLIST,correctionLIST,facetCROSSLIST,
#                                  gridCOx,gridCOy,loopX,loopY)
#                                        
#        #      %----------
#        #      % - 2 - Check for crossed facets:
#        #      %----------
#              facetCROSSLIST= CheckForCrossedFacets(meshXYZ,
#                                              possibleCROSSLIST,facetCROSSLIST,
#                                              gridCOx,gridCOy,loopX,loopY)
#              
#        #        %----------
#        #        % - 3 - Find the z coordinate of the locations where the ray crosses each facet or vertex:
#        #        %----------
#              gridCOzCROSS=FindCrossingLocation(meshXYZ,facetCROSSLIST,
#                                                gridCOx,gridCOy,loopX,loopY)  
#                
#                
#              #%Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
#              gridCOzCROSS = gridCOzCROSS[ np.logical_and(
#                  np.greater_equal(gridCOzCROSS,(meshZmin-1e-12)*np.ones(gridCOzCROSS.shape)),
#                  np.less_equal(gridCOzCROSS,(meshZmax+1e-12)*np.ones(gridCOzCROSS.shape))) ]
#          
#              #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
#              gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
#              gridCOzCROSS = np.unique(gridCOzCROSS)
#    
#              if gridCOzCROSS.size%2 == 0: 
#                  for loopASSIGN in range(gridCOzCROSS.size/2):
#                      voxelsINSIDE = np.logical_and(
#                                    np.greater(gridCOz,gridCOzCROSS[2*loopASSIGN]), 
#                                    np.less(gridCOz,gridCOzCROSS[2*loopASSIGN+1]))
#                                    
#                      image[loopX,loopY,voxelsINSIDE] = 1
#              elif len(gridCOzCROSS)>0:
#                  correctionLIST.append([loopX,loopY])
#        
#    
#    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
#    #For rays where the voxelisation did not give a clear result, the ray is
#    #computed by interpolating from the surrounding rays.    
#
#    image=InterpolateRemainingVoxels(correctionLIST,sampleDimensions,image)
#    
#    return image  
#
#
#
#def CheckForCrossedVertices(meshXYZ,possibleCROSSLIST,correctionLIST,facetCROSSLIST,
#                            gridCOx,gridCOy,loopX,loopY):
##      % Find which mesh facets contain a vertex which is crossed by the ray:
#  vertexCROSSLIST = possibleCROSSLIST[np.logical_or(np.logical_or(
#                   np.logical_and(meshXYZ[possibleCROSSLIST,0,0]==gridCOx[loopX], 
#                           meshXYZ[possibleCROSSLIST,1,0]==gridCOy[loopY]),
#                   np.logical_and(meshXYZ[possibleCROSSLIST,0,1]==gridCOx[loopX], 
#                           meshXYZ[possibleCROSSLIST,1,1]==gridCOy[loopY])),
#                   np.logical_and(meshXYZ[possibleCROSSLIST,0,2]==gridCOx[loopX], 
#                           meshXYZ[possibleCROSSLIST,1,2]==gridCOy[loopY]))
#                          ]
#  
#  if len(vertexCROSSLIST)>0:  #Only continue the analysis if potential 
#                              #vertices were actually identified
#
#      checkindex = np.zeros(vertexCROSSLIST.size)
#
#      while min(checkindex) == 0:
#      
#          vertexindex             = np.nonzero(checkindex==0)[0][0]
#          checkindex[vertexindex] = 1
#        
#          tempfaces,tempvertices = CONVERT_meshformat(meshXYZ[vertexCROSSLIST,:,:])
#          adjacentindex  = np.concatenate((
#              np.reshape(np.asarray([ np.sum(a[0] == tempfaces[vertexindex,0]) for a in tempfaces ],dtype=np.bool),(len(tempfaces),1)),
#              np.reshape(np.asarray([ np.sum(a[0] == tempfaces[vertexindex,0]) for a in tempfaces ],dtype=np.bool),(len(tempfaces),1))),
#              axis=1)
#          adjacentindex  = np.max(adjacentindex,axis=1)
#          checkindex[adjacentindex]  = 1;
#        
#          coN = COMPUTE_mesh_normals(meshXYZ[vertexCROSSLIST[adjacentindex],:,:])
#          #vtkPolyDataNormals
#          if max(coN[:,2])<0 or min(coN[:,2])>0:
#              facetCROSSLIST.append(vertexCROSSLIST[vertexindex])
#          else:
#              possibleCROSSLIST = []
#              correctionLIST.append([loopX,loopY])
#              checkindex[:]     = 1
#  
#  return possibleCROSSLIST, correctionLIST,facetCROSSLIST
#
#def CheckForCrossedFacets(meshXYZ,possibleCROSSLIST,facetCROSSLIST,gridCOx,gridCOy,loopX,loopY):
#  
#  if len(possibleCROSSLIST)>0:  #Only continue the analysis if some nearby 
#                                #facets were actually identified
#      
#      for loopCHECKFACET in np.transpose(possibleCROSSLIST):
#  
#      #Check if ray crosses the facet. Taking each edge of the facet in turn, 
#      #check if the ray is on the same side as the opposing vertex.
#    
#          Y1predicted = meshXYZ[loopCHECKFACET,1,1] - ((
#              meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2])*(
#              meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,0])/(
#              meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
#          YRpredicted = meshXYZ[loopCHECKFACET,1,1] - ((
#              meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2])*(
#              meshXYZ[loopCHECKFACET,0,1]-gridCOx[loopX])/(
#              meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
#        
#          if ((Y1predicted > meshXYZ[loopCHECKFACET,1,0] and
#              YRpredicted > gridCOy[loopY]) or 
#              (Y1predicted < meshXYZ[loopCHECKFACET,1,0] and
#              YRpredicted < gridCOy[loopY])):
#            #The ray is on the same side of the 2-3 edge as the 1st vertex.
#
#              Y2predicted = meshXYZ[loopCHECKFACET,1,2] - ((
#                  meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0])*(
#                  meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,1])/(
#                  meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
#              YRpredicted = meshXYZ[loopCHECKFACET,1,2] - ((
#                  meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0])*(
#                  meshXYZ[loopCHECKFACET,0,2]-gridCOx[loopX])/(
#                  meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
#              
#              if ((Y2predicted > meshXYZ[loopCHECKFACET,1,1] and 
#                  YRpredicted > gridCOy[loopY]) or
#                  (Y2predicted < meshXYZ[loopCHECKFACET,1,1] and
#                  YRpredicted < gridCOy[loopY])):
#                  #The ray is on the same side of the 3-1 edge as the 2nd vertex.
#    
#                  Y3predicted = meshXYZ[loopCHECKFACET,1,0] - ((
#                      meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1])*(
#                      meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,2])/(
#                      meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
#                  YRpredicted = meshXYZ[loopCHECKFACET,1,0] - ((
#                      meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1])*(
#                      meshXYZ[loopCHECKFACET,0,0]-gridCOx[loopX])/(
#                      meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
#                
#                  if ((Y3predicted > meshXYZ[loopCHECKFACET,1,2] and
#                      YRpredicted > gridCOy[loopY]) or 
#                      (Y3predicted < meshXYZ[loopCHECKFACET,1,2] and
#                      YRpredicted < gridCOy[loopY])):
#                    #The ray is on the same side of the 1-2 edge as the 3rd vertex.
#    
#                    #The ray passes through the facet since it is on the correct 
#                    #side of all 3 edges
#                      facetCROSSLIST.append(loopCHECKFACET)
#                      
#  return facetCROSSLIST    
#
#def FindCrossingLocation(meshXYZ,facetCROSSLIST,gridCOx,gridCOy,loopX,loopY):              
#          
#  facetCROSSLIST=np.asarray(facetCROSSLIST)
#  gridCOzCROSS = np.zeros(facetCROSSLIST.shape)
#  for loopFINDZ in facetCROSSLIST:
#
##          % METHOD:
##          % 1. Define the equation describing the plane of the facet.  For a
##          % more detailed outline of the maths, see:
##          % http://local.wasp.uwa.edu.au/~pbourke/geometry/planeeq/
##          %    Ax + By + Cz + D = 0
##          %    where  A = y1 (z2 - z3) + y2 (z3 - z1) + y3 (z1 - z2)
##          %           B = z1 (x2 - x3) + z2 (x3 - x1) + z3 (x1 - x2)
##          %           C = x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)
##          %           D = - x1 (y2 z3 - y3 z2) - x2 (y3 z1 - y1 z3) - x3 (y1 z2 - y2 z1)
##          % 2. For the x and y coordinates of the ray, solve these equations to find the z coordinate in this plane.
#
#      planecoA=(meshXYZ[loopFINDZ,1,0]*(meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,2,2])+ 
#            meshXYZ[loopFINDZ,1,1]*(meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,2,0])+
#            meshXYZ[loopFINDZ,1,2]*(meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,2,1]))
#      
#      planecoB = (meshXYZ[loopFINDZ,2,0]*(meshXYZ[loopFINDZ,0,1]-meshXYZ[loopFINDZ,0,2])+
#            meshXYZ[loopFINDZ,2,1]*(meshXYZ[loopFINDZ,0,2]-meshXYZ[loopFINDZ,0,0])+
#            meshXYZ[loopFINDZ,2,2]*(meshXYZ[loopFINDZ,0,0]-meshXYZ[loopFINDZ,0,1])) 
#      
#      planecoC = (meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]-meshXYZ[loopFINDZ,1,2])+
#            meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]-meshXYZ[loopFINDZ,1,0]) + 
#            meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]-meshXYZ[loopFINDZ,1,1]))
#      
#      planecoD = (- meshXYZ[loopFINDZ,0,0]*(
#                meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,2]-
#                meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,1])-
#            meshXYZ[loopFINDZ,0,1]*(
#                meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,0]-
#                meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,2]) -
#            meshXYZ[loopFINDZ,0,2]*(
#                meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,1]-
#                meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,0]))
#
#      if abs(planecoC) < 1e-14 :
#          planecoC=0
#      
#      gridCOzCROSS[facetCROSSLIST==loopFINDZ] = (- planecoD - 
#            planecoA*gridCOx[loopX] - planecoB*gridCOy[loopY]) / planecoC
#    
#  return gridCOzCROSS 
#
#

#
#
#
##------------------------------------------------------------------------------
#def  CONVERT_meshformat(meshXYZ):
##%CONVERT_meshformat  Convert mesh data from array to faces,vertices format or vice versa
##%==========================================================================
##% AUTHOR        Adam H. Aitkenhead
##% CONTACT       adam.aitkenhead@christie.nhs.uk
##% INSTITUTION   The Christie NHS Foundation Trust
##%
##% USAGE         [faces,vertices] = CONVERT_meshformat(meshXYZ)
##%         or... [meshXYZ]        = CONVERT_meshformat(faces,vertices)
##%
##% IN/OUTPUTS    meshXYZ  - Nx3x3 array - An array defining the vertex
##%                          positions for each of the N facets, with: 
##%                            1 row for each facet
##%                            3 cols for the x,y,z coordinates
##%                            3 pages for the three vertices
##%
##%               vertices - Nx3 array   - A list of the x,y,z coordinates of
##%                          each vertex in the mesh.
##%
##%               faces    - Nx3 array   - A list of the vertices used in
##%                          each facet of the mesh, identified using the row
##%                          number in the array vertices.
##%==========================================================================
##
#
#    vertices = np.concatenate((meshXYZ[:,:,0],meshXYZ[:,:,1],meshXYZ[:,:,2]),axis=0) 
#    vertices = np.asarray(list(set([tuple(vertices[i,:])
#                                    for i in range(vertices.shape[0]) ])))
#
#    faces = np.zeros((meshXYZ.shape[0],2))
#
#    for loopF in range( 0,meshXYZ.shape[0]):
#        for loopV in range(0,2):
#
#            vertref = np.nonzero(vertices[:,0]==meshXYZ[loopF,0,loopV])[0]
#            vertref = vertref[vertices[vertref,1]==meshXYZ[loopF,1,loopV]]
#            vertref = vertref[vertices[vertref,2]==meshXYZ[loopF,2,loopV]]
#      
#            faces[loopF,loopV] = vertref
#
#    return faces,vertices
#
#
##------------------------------------------------------------------------------
#def COMPUTE_mesh_normals(meshdataIN):
##% COMPUTE_mesh_normals  Calculate the normals for each facet of a triangular mesh
##%==========================================================================
##% AUTHOR        Adam H. Aitkenhead
#
#    coordVERTICES = meshdataIN;
#
#    
#    #%======================
#    #% Initialise array to hold the normal vectors
#    #%======================
#    
#    facetCOUNT   = coordVERTICES.shape[0]
#    coordNORMALS = np.zeros((facetCOUNT,3))
#    
#   
#    
#    #%======================
#    #% Compute the normal vector for each facet
#    #%======================
#    
#    for loopFACE in range(0,facetCOUNT):
#      
#      #Find the coordinates for each vertex.
#        cornerA = coordVERTICES[loopFACE,range(3),0]
#        cornerB = coordVERTICES[loopFACE,range(3),1]
#        cornerC = coordVERTICES[loopFACE,range(3),2]
#      
#      #Compute the vectors AB and AC
#        AB = cornerB-cornerA
#        AC = cornerC-cornerA
#        
#      #Determine the cross product AB x AC
#        ABxAC = np.cross(AB,AC) 
#        
#      #Normalise to give a unit vector
#        ABxAC = ABxAC / np.linalg.norm(ABxAC)
#        coordNORMALS[loopFACE,range(3)] = ABxAC
#      
#    
#
#
#    return coordNORMALS
#




#--------------------------------------------------------------------    
def GetSubWindowInformation(subWindowBounds,gridX,gridY,gridZ): 

    nVoxGridX = len(gridX)-1  
    nVoxGridY = len(gridY)-1
    nVoxGridZ = len(gridZ)-1    
    bounds=subWindowBounds    
    
    
    Xmin = min(gridX)
    deltaX = (max(gridX)-Xmin)/float(nVoxGridX)    
    subNxMin = int(math.floor((bounds[0]-Xmin)/deltaX)-1)   
    subNxMax = int(math.ceil((bounds[1]-Xmin)/deltaX)+1)
    subgridXmin = Xmin+deltaX*subNxMin
    subgridXmax = Xmin+deltaX*subNxMax
    nVoxSubgridX = subNxMax - subNxMin
    
    Ymin = min(gridY)
    deltaY = (max(gridY)-Ymin)/float(nVoxGridY)
    subNyMin = int(math.floor((bounds[2]-Ymin)/deltaY)-1)
    subNyMax = int(math.ceil((bounds[3]-Ymin)/deltaY)+1)
    subgridYmin = Ymin+deltaY*subNyMin
    subgridYmax = Ymin+deltaY*subNyMax
    nVoxSubgridY = subNyMax - subNyMin
    
    Zmin = min(gridZ)
    deltaZ = (max(gridZ)-Zmin)/float(nVoxGridZ)
    subNzMin = int(math.floor((bounds[4]-Zmin)/deltaZ)-1)
    subNzMax = int(math.ceil((bounds[5]-Zmin)/deltaZ)+1)
    subgridZmin = Zmin+deltaZ*subNzMin
    subgridZmax = Zmin+deltaZ*subNzMax
    nVoxSubgridZ = subNzMax - subNzMin

    nVoxSubImage = (nVoxSubgridX,nVoxSubgridY,nVoxSubgridZ)
    
    boundSubgrid = (subgridXmin,subgridXmax,
                    subgridYmin,subgridYmax,
                    subgridZmin,subgridZmax,)
                    
    gridRelativePosition=(subNxMin,subNxMax,
                          subNyMin,subNyMax,
                          subNzMin,subNzMax,)
    
    return nVoxSubImage,boundSubgrid,gridRelativePosition

#--------------------------------------------------------------------    
def InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition): 
    
    subNxMin,subNxMax,subNyMin,subNyMax,subNzMin,subNzMax = gridRelativePosition   
    
    wXmin = max(0,subNxMin)
    wXmax = min(nVoxImage[0],subNxMax)
    subwXmin = max(0,-subNxMin)
    subwXmax = min(nVoxImage[0]-subNxMin,subNxMax-subNxMin)
    #print subwXmax, subwXmin , wXmax,wXmin
    assert subwXmax>=subwXmin &  (wXmax>=wXmin)    
    
    wYmin = max(0,subNyMin)
    wYmax = min(nVoxImage[1],subNyMax)
    subwYmin = max(0,-subNyMin)
    subwYmax = min(nVoxImage[1]-subNyMin,subNyMax-subNyMin)
    assert subwYmax>=subwYmin &  (wYmax>=wYmin)    
    
    wZmin = max(0,subNzMin)
    wZmax = min(nVoxImage[2],subNzMax)
    subwZmin = max(0,-subNzMin)
    subwZmax = min(nVoxImage[2]-subNzMin,subNzMax-subNzMin)
    assert subwZmax>=subwZmin &  (wZmax>=wZmin)
    
    objectImage = np.zeros(nVoxImage,dtype=np.uint8)
    objectImage[wXmin:wXmax,wYmin:wYmax,wZmin:wZmax] = subImage[subwXmin:subwXmax,
                                            subwYmin:subwYmax,subwZmin:subwZmax]

    return objectImage
    
