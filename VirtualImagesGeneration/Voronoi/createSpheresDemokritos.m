function createSpheresDemokritos( folderName )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    parentFolder=pwd;
    cd(folderName)
    
    path(path,parentFolder)
    
    points=dlmread('points.txt');
    vertices=dlmread('vertices.txt');
    
    radius=dlmread('radius.txt');
    radiusMax=max(radius);
    
    nVertice=length(vertices(:,1));
    
    Xmin=(min(points(:,1)-2*radiusMax));
    Xmax=(max(points(:,1)+2*radiusMax));
    Ymin=(min(points(:,2)-2*radiusMax));
    Ymax=(max(points(:,2)+2*radiusMax));
    Zmin=(min(points(:,3)-2*radiusMax));
    Zmax=(max(points(:,3)+2*radiusMax));
    
    
    voxelEdgeSize=1e-6;
    
    gridX=Xmin:voxelEdgeSize:Xmax;
    gridY=Ymin:voxelEdgeSize:Ymax;
    gridZ=Zmin:voxelEdgeSize:Zmax;
    
    
    sphereGrid=false( length(gridX),length(gridY),length(gridZ) );
    
    [x,y,z] = ndgrid(-4:4);
    sphere = sqrt(x.^2 + y.^2 + z.^2) <=4;
    
    
    for iVertice=1:nVertice
        vertCoord=points((vertices(iVertice,2)),:);
        
        ix=round((vertCoord(1)-Xmin)/voxelEdgeSize)+1;
        iy=round((vertCoord(2)-Ymin)/voxelEdgeSize)+1;
        iz=round((vertCoord(3)-Zmin)/voxelEdgeSize)+1;
        
        sphereGrid(ix-4:ix+4,iy-4:iy+4,iz-4:iz+4)=or(sphere,sphereGrid(ix-4:ix+4,iy-4:iy+4,iz-4:iz+4));
        
    end
    
    
      
    
    load('imageMatlab.mat') %mygrid
    
    cylinderGrid=logical(mygrid);
    
    wholeImage=+or(sphereGrid,cylinderGrid);
    
    
    save(strcat(folderName,'.mat'),'wholeImage')
    
    
    WriteTiffStack(wholeImage,strcat(folderName,'.tif'))
    
    
    cd(parentFolder)
end

