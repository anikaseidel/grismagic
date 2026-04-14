from build_matrix import build_matrix
import os
from scipy.sparse import load_npz
from scipy.sparse.linalg import lsqr, lsmr
import numpy as np
from scipy.optimize import lsq_linear
from scipy.optimize import linprog

class recovery:
    def __init__(self):
        if os.path.exists("A_matrix_with_trace_count.npz"): #checks if file exists
            self.A_full = load_npz("A_matrix_with_trace_count.npz") #loads stored traces matrix
            self.trace_count = self.A_full[-1].toarray().ravel() #gives the amount of trace pixels per column of A. A1 gives 1D vector
            self.A=self.A_full[:-1] #keeps all rows except the last one, so A is the trace build matrix again
        else:
            self.A_full = None
            self.A = None
            self.trace_count = None
            
        if os.path.exists("H_matrix_flux_gaussian_all_orders.npz"): #checks if file exists
            self.H_full = load_npz("H_matrix_flux_gaussian_all_orders.npz") #loads stored traces matrix
           
        else:
            self.H_full = None
            
        if os.path.exists("A_matrix_with_trace_count_sensitivities_all_orders.npz"): #checks if file exists
            self.ASens_full = load_npz("A_matrix_with_trace_count_sensitivities_all_orders.npz") #loads stored traces matrix
            self.trace_countSens = self.ASens_full[-1].toarray().ravel() #gives the amount of trace pixels per column of A. A1 gives 1D vector
            self.ASens=self.ASens_full[:-1] #keeps all rows except the last one, so A is the trace build matrix again
        else:
            self.ASens_full = None
            self.ASens = None
            self.trace_countSens = None
                
        if os.path.exists("H_matrix_flux_1st_order_PCA.npz"): #checks if file exists
            self.H_PCA = load_npz("H_matrix_flux_1st_order_PCA.npz") #loads stored traces matrix
           
        else:
            self.H_PCA = None
        
        if os.path.exists("H_matrix_flux_1st_order_PCA_sensitivity.npz"): #checks if file exists
            self.H_PCA_sens = load_npz("H_matrix_flux_1st_order_PCA_sensitivity.npz") #loads stored traces matrix
           
        else:
            self.H_PCA_sens = None
        
        
    def recover_direct_from_traces_matrix(self, dispersed):
        """Function to recover direct image from SELF-COMPUTED dispersed. Uses the precomputed traces matrix A to recover the direct image from a dispersed image 
        via least squares."""
        m,n = dispersed.shape
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.A,f) #solves min_d ||Ad-f||^2
        d_recovered = result[0]
        
        d=d_recovered*self.trace_count #recovers total intensity for uniform ditribution
        
        Recovered = d.reshape(m, n) #transforms lsqr solution to matrix
        #Recovered[Recovered<0.05]=0 #small values are background error so ignore this
        
        return Recovered
    
    def recover_via_lsqr_bounds(self, dispersed):
        """uniform dist"""
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
       
        result= lsq_linear(self.A, f, bounds=(0, np.inf))
        d=result.x*self.trace_count #recovers total intensity for uniform ditribution
        

        A = build_matrix()
        Recovered = A.integrated_flux_image(d)
        return Recovered
    
    def recover_direct_from_traces_basis_matrix(self, dispersed):
        """Function to recover direct image from GIVEN IMAGE dispersed. Uses the precomputed traces matrix H to recover the direct image from a dispersed image 
        via least squares."""
        print(self.H_full.shape)
        print(dispersed.shape)
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.H_full,f) #solves min_d ||Ad-f||^2.
        d = result[0] #  lsqr stores result as final_solution, istop, itn.... So we use only [0]
        
        A = build_matrix()
        Recovered = A.integrated_flux_image(d)
     
        return Recovered
    
    def recover_direct_from_traces_sensitivities_matrix(self, dispersed):
        """Function to recover direct image from SELF-COMPUTED dispersed. Uses the precomputed traces matrix A to recover the direct image from a dispersed image 
        via least squares."""
        m,n = dispersed.shape
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.ASens,f) #solves min_d ||Ad-f||^2
        d_recovered = result[0]
        
        #d=d_recovered*self.trace_count #recovers total intensity for uniform ditribution
        
        Recovered = d_recovered.reshape(m, n) #transforms lsqr solution to matrix
        #Recovered[Recovered<0.05]=0 #small values are background error so ignore this
        
        return Recovered
    
    def recover_direct_from_traces_basis_matrix_PCA(self, dispersed):
        """Function to recover direct image from GIVEN IMAGE dispersed. Uses the precomputed traces matrix H to recover the direct image from a dispersed image 
        via least squares."""
   
        m,n=dispersed.shape
        f=dispersed.ravel() #flattens dispersion matrix to vector for matrix multiplication
        result = lsqr(self.H_PCA_sens,f, iter_lim=500, show=True) #solves min_d ||Ad-f||^2.
        d = result[0] #  lsqr stores result as final_solution, istop, itn.... So we use only [0]
       
        A = build_matrix()
        Recovered = A.integrated_flux_image_PCA(d)
     
        return Recovered
    
    from scipy.optimize import lsq_linear
from scipy.sparse import vstack, diags
import numpy as np

def recover_direct_from_traces_basis_matrix_PCA_thikonov_variance(self, dispersed):
    """
    Recover direct image from dispersed image using:

    min_{d >= 0} ||W(Hd - f)||^2 + lambda ||d||^2

    where:
        W = diag(1/sigma)
        sigma estimated from data (Poisson + read noise)
    """

    # =====================================================
    # 1. Flatten image
    # =====================================================
    f = dispersed.astype(float).ravel()
    H = self.H_PCA_sens

    # =====================================================
    # 2. Estimate sigma (noise model)
    #    sigma^2 = f + read_noise^2
    # =====================================================
    read_noise = 5.0  # reasonable default (can tune)

    variance = np.maximum(f, 0) + read_noise**2
    sigma = np.sqrt(variance)

    # numerical safety
    sigma = np.maximum(sigma, 1e-8)

    # =====================================================
    # 3. Whitening (diagonal covariance)
    # =====================================================
    W = 1.0 / sigma

    # efficient row scaling
    H_w = H.multiply(W[:, None])   # sparse-safe row scaling
    f_w = W * f

    # =====================================================
    # 4. Tikhonov regularization
    #    augment system:
    #    [H_w        ] d ≈ [f_w]
    #    [√λ I       ]     [0  ]
    # =====================================================
    lam_reg = 1e-3  # tune this

    n_unknowns = H.shape[1]

    reg_matrix = np.sqrt(lam_reg) * diags(np.ones(n_unknowns))

    H_aug = vstack([H_w, reg_matrix])
    f_aug = np.concatenate([f_w, np.zeros(n_unknowns)])

    # =====================================================
    # 5. Solve nonnegative least squares
    # =====================================================
    res = lsq_linear(
        H_aug,
        f_aug,
        bounds=(0, np.inf),
        method='trf',
        max_iter=200,
        lsmr_tol='auto',
        verbose=1
    )

    d = res.x

    # =====================================================
    # 6. Reconstruct image
    # =====================================================
    A = build_matrix()
    Recovered = A.integrated_flux_image_PCA(d)

    return Recovered
