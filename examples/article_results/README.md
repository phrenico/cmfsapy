# Instructions to reproduce the results in "Manifold-adaptive dimension estimation revisited" article

Here I write down the step-by-step instructions to reproduce the results in the aforementioned article.
The instructions assume that you have a proper installation of cmfsapy in your working environment.
Each figure and the table has a script, which generates it.
Some figures have precomputed data, which can be found in a folder or data-generating scripts are available 
in this folder to obtain the required data.

In the followings I write down how to reproduce the results:

## Generate Figure 1
Run the  fig01_FSA_pdfs.py file to generate the first figure about the
empirical and theoretical probability density functions:
```
python fig01_FSA_pdfs.py
```
The output is <a href="./Figure1.pdf">Figure1.pdf.</a>

<embed src="./Figure1.pdf" type="application/pdf">



## Generate Figure 2
Run the fig02medianFSA.py file to generate the second figure:

```
python fig02medianFSA.py
```
The figure contains the sampling distribution of the median for different sample sizes and embedding dimensions.

The output is <a href="./Figure2.pdf">Figure2.pdf.</a>

<embed src="./Figure2.pdf" type="application/pdf">


## Generate Figure 3
Run fig03_dimdep.py to plot the embedding dimension dependence of the medianFSA estimator.
```
python fig03_dimdep.py
```
To speed up the plot-generation process, a supplementary data file: Figure03_data.pkl is available 
with previously measured dimension values.
If you choose to redo the dimension measurements, you can do it by setting the value of gen_and_measure 
boolean parameter to True in the script.

The output is <a href="./Figure3.pdf">Figure3.pdf.</a>

<embed src="./Figure3.pdf" type="application/pdf">


## Generate Figure 4
Run the fig04_convergence.py file to generate Figure 4. 
```
python fig04_convergence.py
```
Data for the plot is stored in the Figure04_data.pkl file.
Fresh data can be generated by setting the gen_and_measure boolean parameter to True,
same as for Fig. 3.

The output is <a href="./Figure4.pdf">Figure4.pdf.</a>


## Generate Figure 5

Run the fig04_convergence.py file to generate Figure 5., which shows the corrigated-mFSA values.
```
python fig05_correction.py
```
Data for the plot is stored in the Figure05_data.pkl file.
Fresh data can be generated by setting the gen_and_measure boolean parameter to True.

The output is <a href="./Figure5.pdf">Figure5.pdf.</a>


## Generate Figure 6
Run the fig06_mpe.py file to generate Figure 6, which shows the comparison of cmFSA and DANCo
on the synthetic benchmark datasets.
```
python fig06_mpe.py
```
The corresponding data files are in the ./benchmark_result folder:
the danco_matlab_benchmark_res.mat matlab file for the results of DANCo
and the cmfsa_benchmark_res.npy npy file for the results of cmFSA.

These data-files can be generated from the scripts in the repository,
for details see the "Generate benchmark data-files and results" section below.

The output is <a href="./Figure6.pdf">Figure6.pdf.</a>

## Generate Figure 7
Run the fig07_epi.py file to generate Figure 7., which shows mean intrinsic dimension values 
at seizures and interictal periods of intracranial EEG recordings.
```
python fig07_epi.py
```
The corresponding data file is in the ./epi_data/Figure07_data.pkl file,
this data is a generated by the ./epi_data/gen_epi_dimvals.py script from 
the raw data file.
The raw data file is a bz2 compressed pickle file with control and seizure
Local Field Potential (LFP) segments. It will be available in the near future.
To the genertaing scripts to work properly, the raw data file should be put
on the ./epi_data/raw_data_dicts.pkl path.

The output is <a href="./Figure7.pdf">Figure7.pdf.</a>


## Generate Figure S1
Run the figS01_calibration.py file to generate Figure S1, which shows the calibration-procedure
of the cmFSA estimator.
```
python figS01_calibration.py
```
Data is in the ./calibration_result/calibration_data_krange20_n2500_d80.npz file,
which is generated by the gen_calibration_data.py script in the same folder.

The data file contains medianFSA values computed for random d-Dimensional cube datasets
with n=2500 sample size. The simulations were carried out in the 2-80 embedding dimension 
and 1-20 neighborhood range, 100 realizations were generated for each embedding dimension value.

The output is <a href="./FigureS1.pdf">FigureS1.pdf.</a>


## Generate Figure S2
Run the figS02_epi_embed.py file to generate Figure S2, which shows additional information for the 
time delay embedding of the EEG signals.
```
python figS02_epi_embed.py
```
The data file is at ./epi_data/epi_embed/embedding_results in numpy binary format
(check the script to see how to load it).

The output is <a href="./FigureS2.pdf">FigureS2.pdf.</a>


## Generate Table 1
To generate Table 1, run the table.py file.
```
python table.py
```
The corresponding data files are in the ./benchmark_result folder:
- danco_matlab_benchmark_res.mat: matlab file for the results of DANCo matlab implementation
- danco_r_benchmark_res.npy: DANCo R implementation
- ml_benchmark_res.npy: Maximum likelihood estimator (Levina-Bickel) results
- fsa_krange20_benchmark_res.npy: mFSA results
- cmfsa_benchmark_res.npy: npy file for the results of cmFSA.

These data-files can be generated from the scripts in the repository,
for details see the "Generate benchmark data-files and results" section below.

## Generate benchmark data-files and results
To generate the data and reproduce the results
you will need GNU octave, matlab and R beyond pure python. 

### Generate benchmark data
To generate benchmark dataset, first run the ./benchmark_data/gen_benchmark_data.m file 
from the benchmark data folder:
```
octave gen_benchmark_data.m
```
This will generate the data files into the ./manifold_data/ folder.

### Generate benchmark results

To generate mFSA results run:
```
python gen_fsa_krange_benchmark.py 
```

To generate cmFSA results run:
```
python gen_cmfsa_benchmark.py 
```
Here the script uses the saved results of a calibration procedure (coefs.npy and powers.npy),
you can recalibrate the method by running the calibration.py file.

To generate Maximum-Likelihood (Levina-Bickel) results run:
```
python gen_ML_benchmark.py 
```

To generate DANCo results enter to the benchmark_result/DANCo/ folder,
then chose the implementation to generate result-data.

To generate the DANCo results with the idEstimation matlab implementation enter 
to the ./benchmark_result/DANCo/matlab/ subfolder and run gen_danco_results.m with matlab.
Unfortunately the package at the time is not compatible with GNU octave or Scilab.


To generate the DANCo results with the intrinsicDimension R package 
you can run the gen_DANCo_R_benchmark_res.py python file, which uses the rpy2 package.
This package unfortunately is not compatible with python3, so you should run the script e.g. in a
python2.7 virtual environment:
```
python gen_DANCo_R_benchmark_res.py 
```
This operation can take a while...