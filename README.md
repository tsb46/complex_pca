# Complex Principal Component Analysis for fMRI
This repository contains a Python command line script for performing complex principal component analysis (CPCA) of functional MRI (fMRI) time courses, as described in Bolt et al. (2022; Nature Neuroscience). CPCA allows for the extraction of traveling wave patterns via the the eigendecomposition of complex-valued time courses (see more details below). We refer to this analysis as complex PCA, but in other fields, it has been referred to as complex hilbert empirical orthogonal functions (Hannachi et al., 2007) or complex orthogonal decomposition.

The Python script performs group or subject-level CPCA on variety of file formats including nifti, cifti or text files containing 2d arrays (e.g. ROI time courses). The individual file(s) are passed to the script with a text file containing their absolute or relative file paths. The script performs group-wise temporal concatenation (if multiple files are provided), followed by a Hilbert transform of the time courses before application of CPCA. This Python script is meant to be performed after fMRI preprocessing, and expects individual fMRI scans to be in the same standard space (unless analysis is performed on a single scan). To isolate the temporal frequency of a given CPCA component it is highly recommended to bandpass filter the signals to a restricted frequency range (e.g. 0.01 - 0.1 Hz). The script provides an option to bandpass filter the time courses before CPCA. This package also provides functionality for varimax or promax rotation of complex principal component loadings via code from the [xmca](https://github.com/nicrie/xmca) package.

# Implementation
CPCA is implemented with a randomized singular value decomposition (SVD) algorithm developed by Facebook (https://github.com/facebookarchive/fbpca).

# Installation
The code in this repo was run with Python 3.11, but should be compatible with other V3 versions. 

In the base directory of the repo, pip install all necessary packages:
```
pip install -r requirements.txt
```

# Usage
## Basic Usage
cpca.py is a command-line Python script that can be called from the base directory of the repository. For example, to run CPCA with 10 components on a batch of fMRI scans (w/ their file paths specified in 'input_files.txt', for example) in nifti format: 
```
 python cpca.py -i inpute_files.txt -n 10 -m mask
```
> [!NOTE]
> A filepath to a brain mask in nifti format must be provided to the cpca.py script via the '-m' argument if you pass nifti files

By default, cpca.py expects files in nifti format, but you also can pass cifti files (no mask is needed in this case). Just make sure you pass 'cifti' the '-f' argument:
```
 python cpca.py -i inpute_files.txt -n 10 -f cifti
```
You can also pass text files of 2-dimensional arrays arranged with observations in the rows and ROIs/voxels/vertices in the columns:
```
 python cpca.py -i inpute_files.txt -n 10 -f txt
```

## Standard PCA
You can also run standard (non-complex) PCA by passing 'real' to the '-t' argument:
```
 python cpca.py -i inpute_files.txt -n 10 -m mask -t real
```

## Specify Output Path
By default, the results of CPCA are written to the base directory with a standard label based on the parameters passed to the script. You can also pass a path (or file name) via the '-o' argument to override this behavior:
```
 python cpca.py -i inpute_files.txt -n 10 -m mask -o new_file_path
```

## Rotation of Principal Components
Principal components tend to lack 'simple structure', meaning a large number of time courses correlate strongly with every component (particularly, the first few components). One can attempt to 'sparsify' these components via orthogonal (e.g. varimax) or oblique (e.g. promax) rotation of the principal component loadings. We offer varimax and promax rotation of the complex principal components:

```
 python cpca.py -i inpute_files.txt -n 10 -m mask -r varimax
```

## Reconstruction of Complex Principal Components
In Bolt et al. (2022), we created 'movies' of each component by reconstructing their time courses from the phase time courses (i.e. the the phase representation of the complex-valued principal component time courses). We provide an option to perform reconstruction of each complex principal component via the '-recon' argument:

```
 python cpca.py -i inpute_files.txt -n 10 -m mask -recon
```
> [!WARNING]
> A separate nifti/cifti/txt file is created for each reconstructed complex component. This could write a lot of files to disc if the number of components is large (n > 10).

## Help
For further parameter options and their descriptions, write the following command into the terminal:
```
python cpca.py --help

usage: cpca.py [-h] -i INPUT -n N_COMPS [-m MASK] [-f {nifti,cifti,txt}]
               [-o OUTPUT_PREFIX] [-t {real,complex}] [-r {varimax,promax}]
               [-recon] [-norm {zscore,mean_center}] [-b]
               [-f_low BANDPASS_FILTER_LOW] [-f_high BANDPASS_FILTER_HIGH]
               [-tr SAMPLING_UNIT] [-n_bins N_RECON_BINS] [-v]

Run CPCA or PCA analysis

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        <Required> file path to .txt file containing the file
                        paths to individual fMRI scans in nifti, cifti or 2D
                        matrices represented in .txt. format. The 2D matrix
                        should be observations in rows and columns are
                        voxel/ROI/vertices
  -n N_COMPS, --n_comps N_COMPS
                        <Required> Number of components from PCA
  -m MASK, --mask MASK  path to brain mask in nifti format. Only needed if
                        file_format="nifti"
  -f {nifti,cifti,txt}, --file_format {nifti,cifti,txt}
                        the file format of the individual fMRI scans specified
                        in input
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        the output file name. Default will be to save to
                        current working directory with standard name
  -t {real,complex}, --pca_type {real,complex}
                        Calculate complex or real PCA
  -r {varimax,promax}, --rotate {varimax,promax}
                        Whether to rotate pca weights
  -recon, --recon       Whether to reconstruct time courses from complex PCA
  -norm {zscore,mean_center}, --normalize {zscore,mean_center}
                        Type of scan normalization before group concatenation.
                        It is recommend to z-score
  -b, --bandpass_filter
                        Whether to bandpass filter time course w/ a
                        butterworth filter
  -f_low BANDPASS_FILTER_LOW, --bandpass_filter_low BANDPASS_FILTER_LOW
                        Low cut frequency for bandpass filter in Hz
  -f_high BANDPASS_FILTER_HIGH, --bandpass_filter_high BANDPASS_FILTER_HIGH
                        High cut frequency for bandpass filter in Hz
  -tr SAMPLING_UNIT, --sampling_unit SAMPLING_UNIT
                        The sampling unit of the signal - i.e. the TR
  -n_bins N_RECON_BINS, --n_recon_bins N_RECON_BINS
                        Number of phase bins for reconstruction of CPCA
                        components. Higher number results in finer temporal
                        resolution
  -v, --verbose_off     turn off printing
```

# Inputs
The input text file must contain the file paths to each file on separate lines, for example:
```
path/to/file1.nii
path/to/file2.nii
path/to/file3.nii
.
.
.
```

# Output
cpca.py writes out two types of outputs upon completion - 1) spatial maps of the complex principal components and 2) output from the SVD and derivatives (e.g. left, right singular vectors, eigenvalues, loadings, pc scores).

Spatial maps for the complex principal components are written out in the format of the input files (nifti, cifti). They include:
* The real part of the complex principal components
* The imaginary part of the complex principal components (i.e. the real part shifted by $t = {\pi \over 2}$ radians)
* The phase of the complex principal component (this displays the time-delays of the time courses within the component).
* The magnitude map of the complex principal component.
  
Spatial maps are not available for text file inputs.

The SVD output and its derivatives are written to a MATLAB file (.mat) that may be loaded into MATLAB or Python for further analysis:
```
{
  'U': left singular vectors (2d array - ROIs/voxels by components),
  's' singular values (1d array),
  'Va': right singular vectors (2d array - components by time),
  'loadings': normalized right singular vectors b/w -1 and 1 (2d array - components by ROIs/voxels),
  'exp_var': eigenvalues (1d array),
  'pc_scores': complex principal component scores/time courses (2d array - time by components),
  'params': dictionary of input parameters
}
```

# CPCA Description
CPCA allows the representation of time-lag relationships between BOLD signals through the introduction of complex correlations between Hilbert transformed complex-valued BOLD signals. The original time courses and their Hilbert transforms are complex vectors with real and imaginary components, corresponding to the non-zero-lagged time course (t=0) and the time course phase shifted by 
$t = {\pi \over 2}$ radians (i.e. 90 degrees), respectively. The correlation between two complex signals is itself a complex number (composed of a real and imaginary part), and allows one to derive the phase offset (and magnitude) between the original time courses - i.e. the time-lag at which the correlation is maximum. CPCA applied to the complex-valued correlation matrix produces complex spatial weights for each principal component that can give information regarding the time-lags between time courses. In the same manner that a complex signal can be represented in terms of amplitude and phase components (via Euler’s transform), the real and imaginary components of the complex principal component can be represented in terms of amplitude and phase spatial weights. The principal components from the CPCA retain the same interpretive relevance as the original PCA - the first N principal components represent the top N dimensions of variance in the Hilbert transformed BOLD signals. 


# References
Bolt, T., Nomi, J. S., Bzdok, D., Salas, J. A., Chang, C., Thomas Yeo, B. T., Uddin, L. Q., & Keilholz, S. D. (2022). A parsimonious description of global functional brain organization in three spatiotemporal patterns. Nature Neuroscience, 25(8), Article 8. 

Feeny, B. F. (2008). A complex orthogonal decomposition for wave motion analysis. Journal of Sound and Vibration, 310(1), 77–90. https://doi.org/10.1016/j.jsv.2007.07.047

Hannachi, A., Jolliffe, I. T., & Stephenson, D. B. (2007). Empirical orthogonal functions and related techniques in atmospheric science: A review. International Journal of Climatology, 27(9), 1119–1152. https://doi.org/10.1002/joc.1499
https://doi.org/10.1038/s41593-022-01118-1



