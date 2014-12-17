function myVoxeliseFolder(folderName)
    
    
    
    parentFolder=pwd;
    cd(folderName)

    path(path,'/home/greentoto/Documents/MATLAB/Mesh_voxelisation/Mesh_voxelisation')
    path(path,parentFolder)
    
    points=dlmread('points.txt');
    radius=dlmread('radius.txt');
    radiusMax=max(radius);

    %attention toutes les longueurs ici et dans les stl sont 100 plus grande que les grandeurs r�elles
    Xmin=100*(min(points(:,1)-2*radiusMax));
    Xmax=100*(max(points(:,1)+2*radiusMax));
    Ymin=100*(min(points(:,2)-2*radiusMax));
    Ymax=100*(max(points(:,2)+2*radiusMax));
    Zmin=100*(min(points(:,3)-2*radiusMax));
    Zmax=100*(max(points(:,3)+2*radiusMax));

    


    cd ./fichiers_stl
    %%

    %Find nSphere and nFibre
    name=pwd;
    listing = dir(name);
    allNames = { listing.name };
    nFile=length(allNames);

    foo = strfind(allNames, 'fibre');
    nFibre=nFile-sum(cellfun(@isempty,foo));

    foo = strfind(allNames, 'sphere');
    nSphere=nFile-sum(cellfun(@isempty,foo));

    %Create voxel grid
    %nVoxelPerDirection=500;   %nombre approximatif
    %voxelEdgeSize=min([(Xmax-Xmin)/nVoxelPerDirection,(Ymax-Ymin)/nVoxelPerDirection,(Zmax-Zmin)/nVoxelPerDirection]);
    voxelEdgeSize=1e-4;
    
    gridX=Xmin:voxelEdgeSize:Xmax;
    gridY=Ymin:voxelEdgeSize:Ymax;
    gridZ=Zmin:voxelEdgeSize:Zmax;

    %Perform voxelisation of each stl file

    disp(0)

    filename=sprintf('fibre%d.stl',0);

    mygrid= VOXELISE(gridX,gridY,gridZ,filename);

    for i=0:nSphere-1

        filename=sprintf('sphere%d.stl',i);
        disp(filename)
        try
            gridOUTPUT= FastVoxelise(gridX,gridY,gridZ,filename);

            mygrid=or(mygrid,gridOUTPUT);
        catch
            disp(strcat('pas de fichier ',filename))
        end
    end

    for i=0:nFibre-1

        filename=sprintf('fibre%d.stl',i);
        disp(filename)
        try
            %gridOUTPUT= VOXELISE(gridX,gridY,gridZ,filename);
            gridOUTPUT= FastVoxelise(gridX,gridY,gridZ,filename);
            mygrid=or(mygrid,gridOUTPUT);
        catch
            disp(strcat('pas de fichier ',filename))
        end
    end


    mygrid=+mygrid;

    cd ..
    save('imageMatlab','mygrid')



    %Write tif stack file
    stack=mygrid;
    write_stack
    
    cd(parentFolder)
end