function gridOUTPUT = FastVoxelise(gridX,gridY,gridZ,filename)
%FASTVOXELISE Summary of this function goes here
%   Detailed explanation goes here
    
    coordVERTICES = READ_stl(filename);
    [~,vertices] = CONVERT_meshformat(coordVERTICES);
    
    meshXmin=min(vertices(:,1));
    meshXmax=max(vertices(:,1));
    meshYmin=min(vertices(:,2));
    meshYmax=max(vertices(:,2));
    meshZmin=min(vertices(:,3));
    meshZmax=max(vertices(:,3));
    
    
    xfind=find(and( gridX>meshXmin , gridX<meshXmax));
    ixMin=max(xfind(1)-1,1);
    ixMax=min(xfind(end)+1,length(gridX));
    localGridX=gridX(ixMin:ixMax);
    
    yfind=find(and( gridY>meshYmin , gridY<meshYmax));
    iyMin=max(yfind(1)-1,1);
    iyMax=min(yfind(end)+1,length(gridY));
    localGridY=gridY(iyMin:iyMax);
    
    zfind=find(and( gridZ>meshZmin , gridZ<meshZmax));
    izMin=max(zfind(1)-1,1);
    izMax=min(zfind(end)+1,length(gridZ));
    localGridZ=gridZ(izMin:izMax);
    
    localGridOUTPUT= VOXELISE(localGridX,localGridY,localGridZ,coordVERTICES);
    
    
    gridOUTPUT=zeros(length(gridX),length(gridY),length(gridZ));
    
    gridOUTPUT(ixMin:ixMax,iyMin:iyMax,izMin:izMax)=localGridOUTPUT;
end

