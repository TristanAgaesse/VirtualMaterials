contactAngle=110;
network.RemoveLinkData('ContactAngle');
theta=contactAngle*pi/180*ones(1,network.GetNumberOfLinks);
network.AddNewLinkData(theta,'ContactAngle');
[cluster,breakthroughPressure,invasionPressureList]=ComputeInvasionPercolation(network,inletLink,outletLink,'currentWettability');
network.AddNewPoreData(cluster.GetInvadedPoresBooleans,'InvadedPores_110');


ordreInvasion=cluster.InvadedPores;
ordreInvasion(1:end)=0;
for i=1:length(find(cluster.InvadedPores))
ordreInvasion(cluster.InvadedPores(i))=i;
end

network.AddNewPoreData(ordreInvasion,'OrdreInvasion_110');

invasionPressureList(invasionPressureList==0)=ones;

invasionPressure=zeros(1,network.GetNumberOfPores);
for i=1:length(invasionPressureList)
invasionPressure(cluster.InvadedPores(i))=invasionPressureList(i);
end
network.AddNewPoreData(invasionPressure,'InvasionPressure_110');
network.ExportToParaview('test_pseudo_JRC')