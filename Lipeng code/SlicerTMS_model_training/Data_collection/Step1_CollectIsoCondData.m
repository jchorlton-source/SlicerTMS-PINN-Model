dir_data = '/NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Training/';
dir_out = '/NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Code/training/';

fn = dir(['/NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/*_conductivity.nii.gz']);

%parpool(10)
%CoilID = [1 2 5 6 7 8];%1 2
CoilID = [18]% 13 15 17
for i = 1:length(CoilID) %length(fn):-1:1
    fn_cond = dir([dir_data '/Coil' num2str(CoilID(i)) '_*conductivity.nii.gz']);
    %fn_cond = dir([dir_data '/Coil1_130922_F2_ydir17_conductivity.nii.gz']);
    L = length(fn_cond);
    for j = 1:L
        fn_dadt = [dir_data '/' fn_cond(j).name(1:end-19) 'D.nii.gz' ];
        fn_E = [dir_data '/' fn_cond(j).name(1:end-19) 'E.nii.gz' ];
        fn_cond_j = [dir_data '/' fn_cond(j).name ];
        fn_out_E = [dir_out '/' fn_cond(j).name(1:end-19) 'E.nii.gz' ];
        fn_out_Ds = [dir_out '/' fn_cond(j).name(1:end-19) 'Ds.nii.gz' ];
        
        if(exist(fn_dadt,'file')&&exist(fn_E,'file')&&exist(fn_cond_j,'file')&&(~exist(fn_out_E,'file'))&&(~exist(fn_out_Ds,'file')))
            try
                nii_dadt = MRIread(fn_dadt);
                nii_E = MRIread(fn_E);
                nii_cond = MRIread(fn_cond_j);

                nii_out = nii_E;
                nii_out.vol = single(nii_E.vol*100);
                MRIwrite(nii_out,fn_out_E);


                S = single(cat(4,nii_cond.vol, nii_dadt.vol*100));
                nii_out.vol = S;
                MRIwrite(nii_out,fn_out_Ds);
            catch
                fn_cond_j
            end
        end
    end
end
!rm training/*_Fpz*
!rm training/*_Fp1*
!rm training/*_Fp2*