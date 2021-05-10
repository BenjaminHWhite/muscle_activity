% This script can be used to aggregate the movements predicted by multiple
% models.

% 9 movements as defined in the paper.
movements = {'AntComp','BotSwing','Brace', 'Crunch','Dgap','Lift','Pgap','Rest','Swing'};

% 5 csvs generated using test.py using 5 models generated by 5 atlases    
subj='test_data/1218_L_L2G6s-1.h5';  % test data
B=rdir(['test_data/prediction/1218*.csv']);  % movements corresponding to each atlas
f=readtable(B(1).name);
X=zeros(size(f,1),length(B));

for j=1:length(B)
    fprintf('.');
    f=readtable(B(j).name);
    for l=1:size(f,1)
        s=f.Var2{l};
        for u=1:length(movements)
            if strcmp(movements{u},s)
                break;
            end
        end
        X(l,j)=u;
    end
end
fprintf('\n');
X1=mode(X,2);
X2=X1;
for l=6:length(X1)-5
    X2(l)=mode(X1(l-5:l+5));
end
% Using ensembling of the atlases, the final movement is the following.
s=['test_data/prediction/1218_L_L2G6s-1_ensemble.csv'];  
fprintf('Writing %s\n',s);
fp=fopen(s,'w');
for l=1:length(X1)
    
    fprintf(fp,'%d,%s\n',l,movements{X2(l)});
end
fclose(fp);
