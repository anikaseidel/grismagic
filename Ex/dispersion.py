import os
from scipy.sparse import load_npz, csr_matrix
import numpy as np

class dispersion:
    def __init__(self):
                
        if os.path.exists("A_matrix_with_trace_count.npz"): #checks if file exists
            self.A_full = load_npz("A_matrix_with_trace_count.npz") #loads stored traces matrix
            self.trace_count = self.A_full[-1].toarray().ravel() #gives the amount of trace pixels per column of A. A1 gives 1D vector
            self.A=self.A_full[:-1] #keeps all rows except the last one, so A is the trace build matrix again
        else:
            self.A_full = None
            self.A = None
            self.trace_count = None

        
    def compute_dispersed_linear(self, direct_image):
        """Computes the dispersed image for given direct image with the precomputed traces matrix H.
        Uses sparse matrix"""
        m,n = direct_image.shape

        direct_flattened = direct_image.ravel()# #flattens direct image matrix to vector for matrix multiplication. Ravel=Flatten but faster
        
        
        #divides each entry by amount of trace pixels such that the sum of the trace has the same value as its original object. 
        #This simulates intensity distribution. Here: Uniform distribution
        d = np.divide(direct_flattened,self.trace_count, out=direct_flattened.copy(), where = self.trace_count !=0) 
        
        # convert to sparse column vector
        d_sparse = csr_matrix(d).T
        
        #d=direct_distributed.transpose() #transposes vector for correct matrix vector multiplication
        f_sparse = self.A@d_sparse #forward matrix multiplication. Result ist dispersed image vector
        
        f = np.array(f_sparse.todense()).ravel()
        dispersed = f.reshape(m, n) #reconstructs matrix from solution
        return dispersed
       