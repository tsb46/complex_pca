import numpy as np

from scipy.io import savemat
from scipy.stats import zscore
from utils.load_write import load_data, write_out       


def create_bins(phase_ts, n_bins): 
    freq, bins = np.histogram(phase_ts, n_bins)
    bin_indx = np.digitize(phase_ts, bins)
    bin_centers = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
    return bin_indx, bin_centers


def create_dynamic_phase_maps(recon_ts, bin_indx, n_bins):
    bin_timepoints = []
    for n in range(1, n_bins+1):
        ts_indx = np.where(bin_indx==n)[0]
        bin_timepoints.append(np.mean(recon_ts[ts_indx,:], axis=0))
    dynamic_phase_map = np.array(bin_timepoints)
    return dynamic_phase_map


def reconstruct_ts(pca_res, n, rotation, real=True):
    # reconstruct ts 
    if rotation:
        recon_ts = pca_res['pc_scores'][:, n] @ pca_res['loadings'][n, :].conj()
    else:
        U = pca_res['U'][:,n]
        s = np.atleast_2d(pca_res['s'][n])
        Va = pca_res['Va'][n,:].conj()
        recon_ts = U @ s @ Va
    if real:
        recon_ts = np.real(recon_ts)
    else:
        recon_ts = np.imag(recon_ts)
    return recon_ts


def write_results(phase_ts, rotation, n, mask, header, 
                  file_format, out_prefix):
    # write results of cpca reconstruction
    # create output name if out_prefix is None
    if out_prefix is None:
        out_prefix = f'cpca'
        if rotation is not None:
            out_prefix += f'_{rotate}'
    out_prefix += f'_recon_n{n+1}'
    if file_format in ('cifti', 'nifti'):
        write_out(phase_ts, mask, header, file_format, out_prefix)
    else:
        np.savetxt(f'{out_prefix}.txt', phase_ts)


def cpca_recon(cpca_res, rotation, file_format, mask, header, 
                out_prefix, n_bins):
    # reconstruct cpca component 'movies' from cpca results
    bin_indx_all = []
    bin_centers_all = []
    n_comps = cpca_res['pc_scores'].shape[1]
    for n in range(n_comps):
        recon_ts = reconstruct_ts(cpca_res, [n], rotation)
        phase_ts = np.angle(cpca_res['pc_scores'][:,n])
        # shift phase delay angles from -pi to pi -> 0 to 2*pi
        phase_ts = np.mod(phase_ts, 2*np.pi)
        # bin time courses into phase bins
        bin_indx, bin_centers = create_bins(phase_ts, n_bins)
        # average time courses within bins
        phase_ts = create_dynamic_phase_maps(recon_ts, bin_indx, n_bins)
        # write phase ts for component to file
        bin_indx_all.append(bin_indx); bin_centers_all.append(bin_centers)
        write_results(phase_ts, rotation, n, mask, header, 
                      file_format, out_prefix)
    # write metadata from reconstruction
    if out_prefix is None:
        out_prefix = f'cpca'
        if rotation is not None:
            out_prefix += f'_{rotate}'
    out_prefix += f'_recon_metadata'
    metadata = {
        'phase_bin_index': bin_indx_all,
        'phase_bin_centers': bin_centers_all
    }
    savemat(f'{out_prefix}.mat', metadata)
