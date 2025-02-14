import numpy as np
import mdtraj
from bopito.utils.utils import retrieve_data

psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
data_positions = np.load("storage/data/ala2/Ala2TSF300.npy")*10
init_indices = np.load("storage/data/ala2/indices/eq_indices.npy")
init_samples = data_positions[init_indices]
ref_traj = mdtraj.load("storage/data/ala2/alanine-dipeptide-nowater.pdb")

def compute_phi_vector(positions, ref_traj=ref_traj):
    traj = mdtraj.Trajectory(positions, ref_traj.topology)
    phi = mdtraj.compute_dihedrals(traj, [phi_indices])
    phi = phi[:,0] 
    phi_vec = np.transpose(np.array([np.sin(phi), np.cos(phi)]))
                                    
    return phi_vec

init_cond = compute_phi_vector(init_samples)
mean_0 = np.mean(init_cond, axis=0)
init_cond = init_cond - mean_0
factor = np.mean(np.array([ np.dot(init_cond[i], init_cond[i]) for i in range(init_cond.shape[0]) ]))

def compute_norm_correlation(model, lag):
    #md averages
    lagged_samples = data_positions[init_indices+lag]
    phi_lagged = compute_phi_vector(lagged_samples)
    mean_t = np.mean(phi_lagged, axis=0)

    #model
    lagged_samples = retrieve_data(model, lag)
    lagged_samples = lagged_samples[:,1,:,:]
    corr_lagged = compute_phi_vector(lagged_samples)
    corr_lagged = corr_lagged - mean_t
    return np.mean(np.array([ np.dot(init_cond[i], corr_lagged[i]) for i in range(corr_lagged.shape[0]) ]))/factor
    

def compute_correlation_diff(model, lag):
    #md
    lagged_samples = data_positions[init_indices+lag]
    corr_lagged = compute_phi_vector(lagged_samples)
    mean_t = np.mean(corr_lagged, axis=0)
    corr_lagged = corr_lagged - mean_t
    md_corr = np.mean(np.array([ np.dot(init_cond[i], corr_lagged[i]) for i in range(init_cond.shape[0]) ]))/factor
    
    #model
    model_corr = compute_norm_correlation(model, lag)
    return np.abs(md_corr-model_corr)
