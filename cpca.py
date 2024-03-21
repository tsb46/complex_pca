import argparse
import numpy as np
import fbpca
import warnings

from numpy.linalg import pinv
from scipy.io import savemat
from scipy.signal import hilbert
from scipy.stats import zscore
from utils.cpca_reconstruction import cpca_recon
from utils.load_write import load_data, write_out
from xmca.tools.rotation import varimax, promax


def hilbert_transform(input_data, verbose):
    if verbose:
        print('applying hilbert transform')
    # hilbert transform
    input_data = hilbert(input_data, axis=0)
    return input_data.conj()

def package_parameters(n_comps, mask_fp, file_format,
                       pca_type, rotate, normalize, bandpass, 
                       low_cut, high_cut, tr):
    # place input parameters into dictionary to write with results
    params = {
        'n_components': n_comps,
        'mask': mask_fp,
        'file_format': file_format,
        'pca_type': pca_type,
        'rotate': rotate,
        'normalize': normalize,
        'bandpass': bandpass,
        'lowcut': low_cut,
        'highcut': high_cut,
        'tr': tr
    }
    if mask_fp is None:
        params['mask'] = ''
    if rotate is None:
        params['rotate'] = ''
    if not bandpass:
        params['bandpass'] = ''
        params['lowcut'] = ''
        params['highcut'] = ''
    if tr is None:
        params['tr'] = ''
    return params


def pca(input_data, n_comps, verbose, n_iter=10):
    # compute pca
    if verbose:
        print('performing PCA/CPCA')
    # get number of observations
    n_samples = input_data.shape[0]
    # fbpca pca
    (U, s, Va) = fbpca.pca(input_data, k=n_comps, n_iter=n_iter)
    # calc explained variance
    explained_variance_ = ((s ** 2) / (n_samples - 1)) / input_data.shape[1]
    total_var = explained_variance_.sum()
    # compute PC scores
    pc_scores = input_data @ Va.T
    # get loadings from eigenvectors
    loadings =  Va.T @ np.diag(s) 
    loadings /= np.sqrt(input_data.shape[0]-1)
    # package outputs
    output_dict = {'U': U,
                   's': s,
                   'Va': Va,
                   'loadings': loadings.T,
                   'exp_var': explained_variance_,
                   'pc_scores': pc_scores
                   }   
    return output_dict


def rotation(pca_output, data, rotation, verbose):
    if verbose:
        print(f'applying {rotation} to PCA/CPCA loadings')
    # rotate PCA weights, if specified, and recompute pc scores
    if rotation == 'varimax':
        rotated_weights, r_mat = varimax(pca_output['loadings'].T)
        pca_output['r_mat'] = r_mat
    elif rotation == 'promax':
        rotated_weights, r_mat, phi_mat = promax(pca_output['loadings'].T)
        pca_output['r_mat'] = r_mat
        pca_output['phi_mat'] = phi_mat
    # https://stats.stackexchange.com/questions/59213/how-to-compute-varimax-rotated-principal-components-in-r
    # recompute pc scores
    projected_scores = data @ pinv(rotated_weights).T
    pca_output['loadings'] = rotated_weights.T
    pca_output['pc_scores'] = projected_scores
    return pca_output


def write_results(pca_output, pca_type, mask, file_format,
                  header, rotate, out_prefix):
    # write out results of pca analysis
    # create output name if out_prefix is None
    if out_prefix is None:
        if pca_type == 'complex':
            out_prefix = f'cpca'
        elif pca_type == f'real':
            out_prefix = 'pca'
        if rotate is not None:
            out_prefix += f'_{rotate}'
        out_prefix += '_results'

    # get loadings
    loadings = pca_output['loadings']
    # Write brain maps
    if file_format in ('nifti', 'cifti'):
        if pca_type == 'complex': 
            # if complex, write out real, imaginary comps, amplitude and angles
            write_out(
              np.abs(loadings), mask, header, 
              file_format, f'{out_prefix}_magnitude'
            )
            write_out(
              np.real(loadings), mask, header, 
              file_format, f'{out_prefix}_real'
            )
            write_out(
              np.imag(loadings), mask, header, 
              file_format, f'{out_prefix}_imag'
            )
            write_out(
              np.angle(loadings), mask, header, 
              file_format, f'{out_prefix}_phase'
            )
        elif pca_type == 'real':
            write_out(
              loadings, mask, header, file_format, out_prefix
            )
    # write out pca results dictionary to .mat file
    savemat(f'{out_prefix}.mat', pca_output)


def run_cpca(input_files, n_comps, mask_fp, file_format, out_prefix, 
             pca_type, rotate, recon, normalize, bandpass, 
             low_cut, high_cut, tr, n_bins, verbose):
    # load dataset
    func_data, mask, header = load_data(
        input_files, file_format, mask_fp, normalize, 
        bandpass, low_cut, high_cut, tr, verbose
    ) 
    # if pca_type is complex, compute hilbert transform
    if pca_type == 'complex':
        func_data = hilbert_transform(func_data, verbose)

    # compute pca
    pca_output = pca(func_data, n_comps, verbose)

    # rotate pca weights, if specified
    if rotate is not None:
        pca_output = rotation(pca_output, func_data, rotate, verbose)

    # if cpca, and recon=True, create reconstructed time courses of complex PC
    if recon & (pca_type == 'complex'):
        if verbose:
            print('performing CPCA component time series reconstruction')
        if n_comps > 10:
            warnings.warn(
              'the # of components estimated is large, CPCA reconstruction '
              'will create a separate file for each component. This may take '
              'a while.'
          )
        del func_data # free up memory
        cpca_recon(pca_output, rotate, file_format,
                   mask, header, out_prefix, n_bins)
    elif recon & (pca_type == 'real'):
        warnings.warn('Time series reconstruction only available for CPCA')

    # put input parameters into PCA results dicitonary
    pca_output['params'] = package_parameters(
        n_comps, mask_fp, file_format, pca_type, rotate, 
        normalize, bandpass, low_cut, high_cut, tr
    )
    # write out results
    if verbose:
        print('writing out results')
    write_results(pca_output, pca_type, mask, file_format, 
                  header, rotate, out_prefix)


if __name__ == '__main__':
    """Run complex or standard principal component analysis"""
    parser = argparse.ArgumentParser(description='Run CPCA or PCA analysis')
    parser.add_argument('-i', '--input',
                        help='<Required> file path to .txt file containing the file paths '
                        'to individual fMRI scans in nifti, cifti or 2D matrices represented '
                        'in .txt. format. The 2D matrix should be observations in rows and '
                        'columns are voxel/ROI/vertices',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-m', '--mask',
                        help='path to brain mask in nifti format. Only needed '
                        'if file_format="nifti"',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('-f', '--file_format',
                        help='the file format of the individual fMRI scans '
                        'specified in input',
                        required=False,
                        default='nifti',
                        choices=['nifti', 'cifti', 'txt'],
                        type=str)
    parser.add_argument('-o', '--output_prefix',
                        help='the output file name. Default will be to save '
                        'to current working directory with standard name',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('-t', '--pca_type',
                        help='Calculate complex or real PCA',
                        default='complex',
                        choices=['real', 'complex'],
                        type=str)
    parser.add_argument('-r', '--rotate',
                        help='Whether to rotate pca weights',
                        default=None,
                        required=False,
                        choices=['varimax', 'promax'],
                        type=str)
    parser.add_argument('-recon', '--recon',
                        help='Whether to reconstruct time courses from '
                        'complex PCA',
                        action='store_true',
                        required=False)
    parser.add_argument('-norm', '--normalize',
                        help='Type of scan normalization before group '
                        'concatenation. It is recommend to z-score',
                        default='zscore',
                        required=False,
                        choices=['zscore', 'mean_center'],
                        type=str)
    parser.add_argument('-b', '--bandpass_filter',
                        help='Whether to bandpass filter time course w/ '
                        'a butterworth filter',
                        action='store_true',
                        required=False)
    parser.add_argument('-f_low', '--bandpass_filter_low',
                        help='Low cut frequency for bandpass filter in Hz',
                        required=False,
                        default=0.01,
                        type=float
                        )
    parser.add_argument('-f_high', '--bandpass_filter_high',
                        help='High cut frequency for bandpass filter in Hz',
                        required=False,
                        default=0.1,
                        type=float
                        )
    parser.add_argument('-tr', '--sampling_unit',
                        help='The sampling unit of the signal - i.e. the TR',
                        required=False,
                        default=None,
                        type=float
                        )
    parser.add_argument('-n_bins', '--n_recon_bins',
                        help='Number of phase bins for reconstruction of CPCA '
                        'components. Higher number results in finer temporal '
                        'resolution',
                        required=False,
                        default=30,
                        type=int
                        )
    parser.add_argument('-v', '--verbose_off',
                        help='turn off printing',
                        action='store_false',
                        required=False)

    args_dict = vars(parser.parse_args())
    run_cpca(args_dict['input'], args_dict['n_comps'], args_dict['mask'],
            args_dict['file_format'], args_dict['output_prefix'], 
            args_dict['pca_type'], args_dict['rotate'], 
            args_dict['recon'], args_dict['normalize'], 
            args_dict['bandpass_filter'], args_dict['bandpass_filter_low'],
            args_dict['bandpass_filter_high'], args_dict['sampling_unit'],
            args_dict['n_recon_bins'], args_dict['verbose_off'])
