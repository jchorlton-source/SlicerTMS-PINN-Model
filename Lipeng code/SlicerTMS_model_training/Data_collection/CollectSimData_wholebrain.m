function CollectSimData_wholebrain(CoilID,subjectID)
% function CollectSimData(subjectID,CoilID,IsoFlag)
% simulate E-field using SimNIBS for a selected subject, coil and type of
% conductivity
%CoilID: 1:18
%subjectID: 1:20
%IsoFlag: 0 aniso, 1 iso (default)

addpath('/data/pnl/lipeng/software/SimNIBS/bin/');
addpath('/data/pnl/lipeng/software/SimNIBS/matlab/');
addpath('/data/pnl/lipeng/software/SimNIBS/simnibs/');
addpath('/data/pnl/lipeng/Efield/ForJax/');
soft_path = '/data/pnl/lipeng/software/';
DataPath = '/data/pnl/lipeng/Efield/ForJax/';


IsoFlag = 1;


load('/data/pnl/lipeng/Efield/ForJax/SampleCoilPositions/SampleInfo.mat','SampleInfo');
load('/data/pnl/lipeng/Efield/ForJax/ResampleInfo.mat','SampleInfo_train','SampleInfo_test','SampleInfo_valid');


sampleinfo = SampleInfo{CoilID,subjectID};


coilName = sampleinfo.coil; %'Magstim_70mm_Fig8.nii.gz';
centerName = sampleinfo.center{10};
subjid = sampleinfo.subj;

Ydir = sampleinfo.ydir(10,:,10);
N_center = 1;%length(centerName);
N_ydir = size(Ydir,1);

%% creat simulation structure

s = sim_struct('SESSION');
s.fields = 'EDs'; % E filed and norm(E), dA/dt and conductivity
s.date = '';
s.poslist = '';
s.poslist{1} = sim_struct('TMSLIST');
s.poslist{1}.fnamecoil = coilName;
s.open_in_gmsh=false; % Open result in gmsh when done
s.map_to_surf=false; % map results on individual surface (read out in middle of GM sheet)
s.map_to_fsavg=false; % map results further onto fsaverage template
s.map_to_vol=false; % write fields as nifti
s.map_to_MNI=false; % write fields as nifti in MNI space



s.fnamehead = [DataPath  'anat/' subjid '/' subjid '.msh'];
s.subpath = [DataPath  'anat/' subjid '/m2m_' subjid];



for c = 1:N_center

    Output_folder = tempname([DataPath 'output/']);
    mkdir(Output_folder);
    s.pathfem = Output_folder;
    
    for y = 1:N_ydir        
        s.poslist{1}.pos(y).centre = centerName;
        s.poslist{1}.pos(y).pos_ydir = Ydir;%(y,:,c);
    end
    
    dir_ID = 1;%setID(Ydir_all(:,:,c),Ydir(:,:,c));

    
    name_mat =[Output_folder '/sim_config.mat'];
    save(name_mat, 's');
% Write the 'simnibs' executable during the postinstall process
    path_to_simnibs = fullfile(soft_path,'SimNIBS/bin', 'simnibs');
    result = system([path_to_simnibs ' ' name_mat]);
    
    
        % copy surface data
        disp('Simulation Completed!')
        
        disp('Begin transfering data!')
        % copy msh data
        Coilid = ['Coil' num2str(CoilID)];
        Prefix =['Coil' num2str(CoilID) '_' subjid '_' centerName];
       
        
        vol_dir = [Output_folder '/vol/'];
        mkdir(vol_dir);
        vol_outputprefix = [vol_dir '/Coil' num2str(CoilID) '_' subjid '_' centerName];
        
        
        fn_mesh = dir([Output_folder '/*.msh']);
        
        
        msh2nii_dir = '/data/pnl/lipeng/software/SimNIBS/bin/msh2nii';

        for i = 1:length(fn_mesh)
            fn_in = [Output_folder '/' fn_mesh(i).name];
            fn_out = [vol_outputprefix '_ydir' num2str(dir_ID(i))];
            system([msh2nii_dir ' ' fn_in ' ' s.subpath ' ' fn_out]);
            
        end

        % transfer data

     
         cp_nas = ['scp ' vol_dir '/*_ydir' num2str(dir_ID(i))  '_*.nii.gz pnl-maxwell.partners.org:/rfanfs/pnl-zorro/home/lipeng/NAS/Synology/'];
          result = system(cp_nas); 
            

        
        
        
        
       % remove data 
       system(['rm -r ' Output_folder]);
        

    
end


