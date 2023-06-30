clear

%% read clinical
[num_exp,txt,raw] = xlsread('PCA_EXP.xlsx');
[num_cnv,txt,raw] = xlsread('PCA_CNV.xlsx');
[num_mt,txt,raw] = xlsread('PCA_MT.xlsx');

[num_reorder,txt,raw] = xlsread('reorder.xlsx');

n=1;
order=num_reorder(:,n);

new_exp=[];
new_cnv=[];
new_mt=[];

% for i=1:length(order)
%     id=order(i);
%     temp_exp=num_exp(:, (id-1)*3+1:(id-1)*3+3);
%     new_exp=[new_exp temp_exp];
%     
%     temp_cnv=num_cnv(:, (id-1)*3+1:(id-1)*3+3);
%     new_cnv=[new_cnv temp_cnv];
%     
%     temp_mt=num_mt(:, (id-1)*3+1:(id-1)*3+3);
%     new_mt=[new_mt temp_mt];
% end
for i=1:length(order)
    id=order(i);
    temp_exp=num_exp(:, (id-1)*5+1:(id-1)*5+5);
    new_exp=[new_exp temp_exp];
    
    temp_cnv=num_cnv(:, (id-1)*5+1:(id-1)*5+5);
    new_cnv=[new_cnv temp_cnv];
    
    temp_mt=num_mt(:, (id-1)*5+1:(id-1)*5+5);
    new_mt=[new_mt temp_mt];
end

[num_mt,txt,raw] = xlsread('pathway_146.xlsx'); 
ordered_pathway=raw(order);% only for n=2 for 2pc

csvwrite('PCA_EXP_reorder_3pc_lgg.csv',new_exp);
csvwrite('PCA_CNV_reorder_3pc_lgg.csv',new_cnv);
csvwrite('PCA_MT_reorder_3pc_lgg.csv',new_mt);

