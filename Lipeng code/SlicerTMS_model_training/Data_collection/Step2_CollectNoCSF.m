CoilID = [13 15 17 18];
for i = 3%1:4

    
    fn_Ds = dir(['training/Coil' num2str(CoilID(i)) '*_Ds.nii.gz']);
    fn_E = dir(['training/Coil' num2str(CoilID(i)) '*_E.nii.gz']);
    L_Ds = length(fn_Ds);
    L_E = length(fn_E);
   % if(L_Ds~=L_E)
   %     ['CoilID =' num2str(CoilID(i))]
   % else
        L = length(fn_Ds);
        
        for i = 1:L
            fn_Ds_out = ['training_R01/' fn_Ds(i).name];
            fn_E_out = ['training_R01/' fn_E(i).name];
            fn_Ds_in = ['training/' fn_Ds(i).name];
            fn_E_in = ['training/' fn_E(i).name];
            if(exist(fn_Ds_in,'file')&&exist(fn_E_in,'file'))
                if((~exist(fn_Ds_out,'file'))||(~exist(fn_E_out,'file')))
                    try
                        nii_Ds = MRIread(['training/' fn_Ds(i).name]);
                        cond = nii_Ds.vol(:,:,:,1);
                        [nx,ny,nz] = size(cond);
                        mask = zeros(size(cond));
                        mask(cond>0.125&cond<0.127) = 0.126;
                        mask(cond>0.274&cond<0.276) = 0.275;
                        mag = nii_Ds.vol(:,:,:,2:end)/100;
                        mag = reshape(mag,nx*ny*nz,3);
                        mag(mask==0,:) = 0;
                        mag = reshape(mag,nx,ny,nz,3);
                        mag = cat(4,mask,mag);
                        nii_Ds.vol = mag;
                        MRIwrite(nii_Ds,['training_R01/' fn_Ds(i).name]);
                        %%
                        nii_E = MRIread(['training/' fn_E(i).name]);
                        E = nii_E.vol/100;
                        E = reshape(E,nx*ny*nz,3);
                        E(mask==0,:) = 0;
                        E = reshape(E,nx,ny,nz,3);
                        nii_E.vol = E;
                        MRIwrite(nii_E,['training_R01/' fn_E(i).name]);
                    catch
                        fn_Ds(i).name
                    end
                end
            end
        end
    %end
end
        
            