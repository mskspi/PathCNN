# PathCNN: Interpretable convolutional neural networks for survival prediction and pathway analysis applied to glioblastoma

Jung Hun Oh <sup>1,†,∗</sup>, Wookjin Choi <sup>2,†</sup>, Euiseong Ko <sup>3</sup>, Mingon Kang <sup>3,∗</sup>, Allen Tannenbaum <sup>4</sup> and Joseph O. Deasy <sup>1</sup>  

<sup>1</sup>Department of Medical Physics, Memorial Sloan Kettering Cancer Center, New York, USA, 
<sup>2</sup>Department of Computer Science, Virginia State University, Petersburg, USA, 
<sup>3</sup>Department of Computer Science, University of Nevada, Las Vegas, USA and 
<sup>4</sup>Departments of Computer Science and Applied Mathematics & Statistics, Stony Brook University, New York, USA  

<sup>*</sup>To whom correspondence should be addressed.  
<sup>†</sup>The authors wish it to be known that, in their opinion, the first two authors should be regarded as Joint First Authors.  

Contact: <ohj@mskcc.org> or <mingon.kang@unlv.edu>

![PathCNN](img/pathcnn.png)

1. Model Building  
   - PathCNN.py  

2. GradCAM  
   - PathCNN_GradCAM_modeling.py: to generate a model for GradCAM (PathCNN_model.h5)
   - PathCNN_GradCAM.py: to generate GradCAM images and a resultant file (pathcnn_gradcam.csv)

3. Multi-omics data
   - GBM multi-omics data including mRNA expression, CNV, and DNA methylation were downloaded from the CBioPortal database.
   - Pathway information was downloaded from the KEGG database.
   - PCA was performed for each pathway in individual omics types.
   
   Five PCs in each omics type are in the following files:
   - PCA_EXP.xlsx, PCA_CNV.xlsx, PCA_MT.xlsx
   
   Clinival variables are in the following file:
   - Clinical.xlsx
