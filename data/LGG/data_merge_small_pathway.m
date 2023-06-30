clear

%% read KEGG
% [num,txt,raw] = xlsread('c2.cp.kegg.v5.2.symbols.xlsx');
% 
% for i=1:size(raw,1)
%    KEGG_Pathway(i).name=raw{i,1};
%    
%    genes={};
%    for j=3:size(raw,2)
%       gene=raw{i, j};
%       
%       if isnan(gene)==1
%          break;
%       else
%          genes=[genes;gene];
%       end
%    end
%    KEGG_Pathway(i).genes=genes;
% end
% 
% %% read clinical variables
% [num,txt,raw] = xlsread('data_bcr_clinical_data_patient.xlsx');
% 
% TCGA_ID               = raw(6:end,2);
% GENDER                = raw(6:end,11);
% AGE                   = cell2mat(raw(6:end,53));
% OS_STATUS	          = raw(6:end,64);
% OS_MONTHS             = cell2mat(raw(6:end,65));
% 
% 
% %% read expression data
% [num,txt,raw] = xlsread('data_RNA_Seq_v2_mRNA_median_Zscores.xlsx');
% EXP_ID=raw(1,3:end);
% EXP_Data=num(:,2:end);
% EXP_Gene=raw(2:end,1);
% 
% %% read CNV data
% [num,txt,raw] = xlsread('data_CNA.xlsx');
% CNV_ID=raw(1,3:end);
% CNV_Data=num(:,2:end);
% CNV_Gene=raw(2:end,1);
% 
% %% read methylation data
% [num,txt,raw] = xlsread('data_methylation_hm450.xlsx');
% MT_ID_450=raw(1,3:end);
% MT_Data_450=num(:,2:end);
% MT_Gene_450=raw(2:end,1);
% 
% save all_data KEGG_Pathway TCGA_ID GENDER    AGE OS_STATUS OS_MONTHS EXP_ID EXP_Data EXP_Gene CNV_ID CNV_Data CNV_Gene    MT_ID_450 MT_Data_450 MT_Gene_450

%% loading
load all_data


%% unique KEGG pathway genes
genes={};
for i=1:length(KEGG_Pathway)
   genes= [genes;KEGG_Pathway(i).genes];
end
uni_genes=unique(genes);

MT_ID=MT_ID_450;
MT_Data=MT_Data_450;
MT_Gene=MT_Gene_450;


EXP_Data=EXP_Data';
CNV_Data=CNV_Data';
MT_Data=MT_Data';

%% so far so good
%% sample overlapping checking based on TCGA
for i=1:length(EXP_ID)
    id1=EXP_ID{i};
    EXP_ID{i}=id1(1:12);
end
[C,ia,ib] = intersect(TCGA_ID,EXP_ID);

for i=1:length(CNV_ID)
    id1=CNV_ID{i};
    CNV_ID{i}=id1(1:12);
end
[C1,ia,ib] = intersect(CNV_ID,C);

for i=1:length(MT_ID)
    id1=MT_ID{i};
    MT_ID{i}=id1(1:12);
end
[C2,ia,ib] = intersect(MT_ID,C1);

Common_samples_EXP_CNV_MT=C2;

%% extract overlapping data
STATUS=[];
for i=1:length(OS_STATUS)
   stat=OS_STATUS{i};
   stat=stat(1);
   
   STATUS=[STATUS;str2num(stat)];
   
end

SEX=[];
for i=1:length(GENDER)
   stat=GENDER(i);
   if strcmp(upper(stat),'FEMALE')==1
       SEX=[SEX;1];
   elseif strcmp(upper(stat),'MALE')==1
       SEX=[SEX;0];
   end
end

%% Clinical data
Clinical_data=[SEX    AGE STATUS OS_MONTHS];
[D,ia,ib] = intersect(TCGA_ID,Common_samples_EXP_CNV_MT);

valid_Clinical_data=Clinical_data(ia,:);

%% Expression
[D1,ia,ib] = intersect(EXP_ID,Common_samples_EXP_CNV_MT);
valid_EXP_Data=EXP_Data(ia,:);

%% CNV
[D2,ia,ib] = intersect(CNV_ID,Common_samples_EXP_CNV_MT);
valid_CNV_Data=CNV_Data(ia,:);

%% MT
[D3,ia,ib] = intersect(MT_ID,Common_samples_EXP_CNV_MT);
valid_MT_Data=MT_Data(ia,:);


%% PCA per Pathway
PCA_EXP=[];
PCA_CNV=[];
PCA_MT=[];
missing_EXP=[];
missing_CNV=[];
missing_MT=[];
all_pathway_genes={};
PC=5;
for i=1:146 % 146 pathway
    i
   pathway_genes=KEGG_Pathway(i).genes;
   all_pathway_genes=[all_pathway_genes;pathway_genes];
   %% expression
   [E1,ia,ib] = intersect(EXP_Gene,pathway_genes);
   exp=valid_EXP_Data(:,ia);
   idx=find(sum(isnan(exp))==0);
   [coeff,score,latent] = pca(exp(:,idx));
   PCA_EXP=[PCA_EXP  score(:,1:PC)];
   missing_EXP=[missing_EXP;length(pathway_genes)-length(E1)];
   
   %% CNV
   [E2,ia,ib] = intersect(CNV_Gene,pathway_genes);
   cnv=valid_CNV_Data(:,ia);
   idx=find(sum(isnan(cnv))==0);
   [coeff,score,latent] = pca(cnv(:,idx));
   PCA_CNV=[PCA_CNV  score(:,1:PC)];
   missing_CNV=[missing_CNV;length(pathway_genes)-length(E2)];
   
   %% MT
   [E3,ia,ib] = intersect(MT_Gene,pathway_genes);
   mt=valid_MT_Data(:,ia);
   idx=find(sum(isnan(mt))==0);
   [coeff,score,latent] = pca(mt(:,idx));
   
   if i==7
        PCA_MT=[PCA_MT  [score(:,1:4) score(:,4)]]; % due to there are only 4, 4th colume was duplicated.
   else
        PCA_MT=[PCA_MT  score(:,1:PC)];
   end
   missing_MT=[missing_MT;length(pathway_genes)-length(E3)];
end

save PCA_results PCA_EXP PCA_CNV PCA_MT PC valid_Clinical_data Common_samples_EXP_CNV_MT


