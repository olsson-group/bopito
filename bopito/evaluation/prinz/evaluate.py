import numpy as np

md = np.load("storage/data/prinz/trajs.npy")
init_cond = md[:,999]
mean_0 = np.mean(init_cond)
factor = np.mean((init_cond-mean_0)**2)


def compute_norm_correlation(model_type, model, lag):
    model_samples = np.load(f'storage/results/prinz_potential/{model_type}/{model}/lag_{str(lag).zfill(3)}_ic_eq.npy')
    mean_t = np.mean(md[:,999+lag])
    model_corr = (init_cond[:len(model_samples)]-mean_0)*(model_samples-mean_t)/factor
    return np.mean(model_corr)

def compute_correlation_diff(model_type, model, lag):
    mean_t = np.mean(md[:,999+lag])
    md_corr = (init_cond-mean_0)*(md[:,999+lag]-mean_t)/factor
    md_corr = np.mean(md_corr)
    model_corr = compute_norm_correlation(model_type, model, lag)
    return np.abs(md_corr-model_corr)