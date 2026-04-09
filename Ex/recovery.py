from build_matrix import build_matrix
import os
from scipy.sparse import load_npz
from scipy.sparse.linalg import lsqr, lsmr
import numpy as np
from scipy.optimize import lsq_linear

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
    
    