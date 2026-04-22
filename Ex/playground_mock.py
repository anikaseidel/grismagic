from grismagic.traces import GrismTrace
from build_matrix import build_matrix
from dispersion import dispersion
from recovery import recovery
import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import time 
import asdf
from pathlib import Path
from astropy.stats import sigma_clip
import pandas as pd
from scipy.linalg import pinv
from scipy.sparse import load_npz, save_npz

#H = build_matrix("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F150W.220725.conf",filter_name="F150W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")
#H.build_and_save_trace_matrix_coefficients_PCA_sensitivity()

######################
# mock image. star consisting of 5 pixels centered at x,y= 50,490

a_star = np.array([1.2,-0.4,0.08,0.02,-0.01,0.005,0,0,0,0]) #manual spectral coefficients

x=10
y=250
a_tilde = np.zeros(500*20*10)
pixel_star = y*20+x
a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star    # all 5 pixels belong to the same star so same spectral coefficients
# pixel_star = 49150
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star
# pixel_star = 48950
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star
# pixel_star = 49051
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star
# pixel_star = 49049
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star

build = build_matrix()
mock_direct = build.integrated_flux_image_PCA(a_tilde) # make direct image visible
np.save("mock_20_500.npy",mock_direct)

disp = dispersion()
mock_dispersed = disp.dispersed_PCA(a_tilde) # compute dispersed image
np.save("mock_dispersed_20_500.npy", mock_dispersed)

recov = recovery()
d= recov.recover_direct_from_traces_basis_matrix_PCA(mock_dispersed, image=False) # recovers image. image=False to output the vector d and not the ready image
#mock_recovered= recov.recover_direct_from_traces_sensitivities_matrix(mock_dispersed)
########################## extracts spectrum of pixel
Phi = build.eigenspectra_basis()
x = 10
y= 250
k = y*20+x 
n = 10
a_k = d[k*n:(k+1)*n]
spectrum = Phi @ a_k #recovered spectrum
######################## original spectrum
spectrum_og = Phi @ a_star

###################### plot both spectra against each other
plt.subplot(1,2,1)
plt.plot(build.lambdas, spectrum_og)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("Original spectrum (x,y)= (50,250)")

plt.subplot(1,2,2)
plt.plot(build.lambdas, spectrum)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("Recovered spectrum (x,y)= (50,250)")
plt.show()

mock_recovered = build.integrated_flux_image_PCA(d) # converts recovered to visible image
np.save("mock_recovered_20_500.npy", mock_recovered)

# #################################################################
base = Path(__file__).resolve().parent

mock_direct = np.load(base / "mock_20_500.npy")
mock_dispersed = np.load(base / "mock_dispersed_20_500.npy")
mock_recovered = np.load(base / "mock_recovered_20_500.npy")
#################################################################
plt.subplot(1,4,1)
std1 = np.nanstd(mock_direct)
mean1 = np.nanmean(mock_direct)
plt.imshow(mock_direct, cmap="inferno", vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Mock direct")

plt.subplot(1,4,2)
std1 = np.nanstd(mock_dispersed)
mean1 = np.nanmean(mock_dispersed)
plt.imshow(mock_dispersed, cmap="inferno", vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Mock dispersed")

plt.subplot(1,4,3)
std1 = np.nanstd(mock_recovered)
mean1 = np.nanmean(mock_recovered)
#max = np.max(mock_recovered)
plt.imshow(mock_recovered, vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Mock recovered")

plt.subplot(1,4,4)
std1 = np.nanstd(mock_recovered-mock_direct)
mean1 = np.nanmean(mock_recovered -mock_direct)
plt.imshow(mock_recovered - mock_direct, vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Rediduals: Mock recovered - Mock direct")
plt.show()

# snippet for spectrum of a specific trace

one_trace = mock_dispersed[320:400, 65:70]
sum = one_trace.sum(axis=1)
x = np.arange(len(sum))

plt.subplot(1,2,1)
plt.plot(sum, x)
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Spectrum of trace")

plt.subplot(1,2,2)
plt.imshow(one_trace)

plt.title("One trace mock dispersion")

plt.show()