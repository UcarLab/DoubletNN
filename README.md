

%%%%%%%%% README for DoubletNN %%%%%%%%%%%%%%%%

Using one of the many bncmrk-dblts outs:

-First, obtain CLR normalized UMI matrices with common genes among different HTO data. For example, when GMM negatives excluded, CZI.PBMC had around 300k cells, 18k genes. PBMC.8.HTO around 16k cells, 18k genes. Four.Cell.12.HTO around 20k, 18k genes. Make sure you saved these common genes, to, e.g, a file named genes.nn.csv, and your barcodes too, e.g, CZI.PBMC.1.brcds. Make sure your matrices are sorted column wise by alphabetic order of barcodes, and your rows by alphabetic order of genes.

You can use the processed Seurat files from DF pipeline, stored as bncmrk-dblts/DF/input/pr_name/pr_name + preprocessed.seurat.for.DF.output.Rds. For example, the Four.Cell.12.HTO.preprocessed.seurat.for.DF.output.Rds in "bncmrk-dblts/DF/input/Four.Cell.12.HTO" folder is one of them. Within that Seurat file you can find "CLR" normalized data matrix. If not, and you have a small dataset, you can just easily normalize it in Seurat and get the assay data from it. Once obtained, write this sparse (clr normalized) UMI matrix onto scipy's sparse matrix format using R's "writeMM" function. For example, you will have sparse matrices saved in scipy mtx format such as "CZI.PBMC.1.umis.mtx". These are going to be your data matrix : X.

-Second, use cells/barcodes identified as Singlets, and Doublets by HTO-GMM method as your 0 and 1 for binary classification. These ground truths should have been saved in bncmrk-dblts/GroundTruths folder. If you didn't already, you can easily follow the first two steps of bncmrk-dblts pipeline. There are necessary functions such as HTO.GMM.analysis.R, which outputs Singlet-Doublet-Negative classifications and CellType1-CellType2-Celltpype3 kind of annotations if the organize_data_script.R pipeline of bncmrk-dblts is followed. If you root out the HTO-GMM identified Negatives (Empty cells), than you have your ground truths as a binary vector: y .

-Third, import the common genes of data sets which was previously saved to "genes.nn.csv", import cell barcodes and import the sparse data matrix in ".mtx" format. The "data_prep.py" stored in DoubletNN/Python folder was reading these genes and matrices for CZI.PBMC data having 10 samples, concatenating them into one large matrix, and saving. If you run:

import numpy as np
import scipy.io
import pandas as pd

genes=pd.read_csv("genes.nn.csv").x[1:]
brcds=pd.read_csv(input_dir+'pr_name'+.brcds').x
counts_matrix=scipy.io.mmread(input_dir + 'pr_name'+'.umis.mtx').T.todense()
df=pd.DataFrame(counts_matrix,index=brcds,columns=genes)

you import your genes and barcodes. You import your "mtx" formatted sparse umi matrices, transpose it, and convert it onto a dense numpy array. You create a pandas dataframe using dense numpy array of counts matrix, barcodes and genes. This is your full data matrix X. You may double check whether rows and columns are alphabetically sorted. If so, save them in very fast and sturdy feather format:

import feather

write_dataframe(df,'pr_name'+'full.X.nn.sorted.cleaned.feather') 

For example, CZI.PBMC.full.X.nn.sorted.cleaned.feather file may be an example.

-Fourth, import your ground truth vector, y. And, split it onto 80% training, 10% validation/dev, 10% train sets. The following code gives you locations for that

a=np.arange(0,len(y))
np.random.shuffle(a)
train_locs, validate_locs, test_locs = np.split(a, [int(.8 * len(a)), int(.9 * len(a))])

such that you can split them by:

X_train=X[train_locs,:] in numpy or X_train=X.ilocs[train_locs,:] in pandas. You may save X_train, y_train, X_validate, y_validate, X_test, y_test onto feather formatted data frames for easier and faster access. e.g, Save them as 'CZI.PBMC.y.train.nn.feather' etc.

-Fifth, by going through the pipelines given in Jupyter notebooks, either use the notebook itself or command line interactive session or batch slurm jobs (I provided codes for it in python and Slurm folders) train your neural network. I am leaving tutorial Sonar and Iris datasets, as well as their pipelines in github. But, keras tutorials are pretty much straight forward. It's more about trial-error and good practices.



