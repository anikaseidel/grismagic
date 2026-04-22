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

######################
# # mock image. star consisting of 5 pixels centered at x,y= 50,490

# a_star = np.array([1.2,-0.4,0.08,0.02,-0.01,0.005,0,0,0,0]) #manual spectral coefficients

# x=10
# y=250
# a_tilde = np.zeros(500*20*10)
# pixel_star = y*20+x
# a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star    # all 5 pixels belong to the same star so same spectral coefficients
# # pixel_star = 49150
# # a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star
# # pixel_star = 48950
# # a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star
# # pixel_star = 49051
# # a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star
# # pixel_star = 49049
# # a_tilde[pixel_star*10 : (pixel_star+1)*10] += a_star

# build = build_matrix()
# mock_direct = build.integrated_flux_image_PCA(a_tilde) # make direct image visible
# np.save("mock_20_500.npy",mock_direct)

# disp = dispersion()
# mock_dispersed = disp.dispersed_PCA(a_tilde) # compute dispersed image
# #mock_dispersed[mock_dispersed>0]=1 
# np.save("mock_dispersed_20_500.npy", mock_dispersed)

# recov = recovery()
# d= recov.recover_direct_from_traces_basis_matrix_PCA(mock_dispersed, image=False) # recovers image. image=False to output the vector d and not the ready image
# #mock_recovered= recov.recover_direct_from_traces_sensitivities_matrix(mock_dispersed)
# ########################## extracts spectrum of pixel
# Phi = build.eigenspectra_basis()
# x = 10
# y= 250
# k = y*20+x 
# n = 10
# a_k = d[k*n:(k+1)*n]
# spectrum = Phi @ a_k #recovered spectrum
# ######################## original spectrum
# spectrum_og = Phi @ a_star

# ###################### plot both spectra against each other
# plt.subplot(1,2,1)
# plt.plot(build.lambdas, spectrum_og)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title("Original spectrum (x,y)= (50,250)")

# plt.subplot(1,2,2)
# plt.plot(build.lambdas, spectrum)
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
# plt.title("Recovered spectrum (x,y)= (50,250)")
# plt.show()

# mock_recovered = build.integrated_flux_image_PCA(d) # converts recovered to visible image
# np.save("mock_recovered_20_500.npy", mock_recovered)

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
plt.imshow(mock_dispersed, cmap="inferno", vmin=-(mean1 + 0*std1), vmax=mean1 + 0*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Mock dispersed")

plt.subplot(1,4,3)
std1 = np.nanstd(mock_recovered)
mean1 = np.nanmean(mock_recovered)
max = np.max(mock_recovered)
plt.imshow(mock_recovered, vmin=0, vmax=max, cmap="inferno", interpolation="nearest", origin="lower",aspect="auto")
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
##################################

# tr = GrismTrace.from_file("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F200W.220725.conf",filter_name="F200W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")  # auto-detects format

# lo, hi = tr._lam_range("1", None, None) #minimum and maximum wavelength in microns for first
# lo = int(lo * 10000) #as int
# lo = max(lo, 7000) #for PCA, goes from 0.7 to 2.2
# hi= int(hi * 10000)   # as int
# hi = min(hi, 22000) #for PCA, goes from 0.7 to 2.2
# lambdas = np.linspace(lo, hi, 150) #wavelength list
# x0, y0 = 250, 250

# x_trace, y_trace = tr.get_trace_at_wavelength(
#     x0, y0, order="A", lam=lambdas
# )

# import matplotlib.pyplot as plt
# plt.plot(x_trace, y_trace)
# plt.title("Trace path")
# plt.show()

####################################
# is A correct?
#######################
# A_sens = load_npz("A_F150W_100_500_matrix_with_trace_count_sensitivities_all_orders.npz") #loads stored traces matrix
# A_sens=A_sens[:-1]
# A_sens[A_sens>0]=1
# pix = 1  # pick arbitrary column


# img = A_sens[:, 250].toarray().reshape(500, 100)
# mean= np.mean(A_sens)
# plt.imshow(img, origin="lower", vmin= -mean, vmax=mean)
# plt.title(f"Pixel {pix}")
# plt.colorbar()
# plt.show()

###############################
# Trying out sensitivity curves
################################
# hdu = fits.open("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\SenseConfig\\wfss-grism-configuration\\NIRISS.GR150R.F200W.1.etc.1.5.2.sens.fits") #F200W, GR150R
# hdu.info()
# #header0= hdu[1].header
# #print(header0)
# data1= hdu[1].data
# wavelength1 = data1["WAVELENGTH"]
# sensitivity1 = data1["SENSITIVITY"]
# mean1 = np.mean(sensitivity1)
# sens1=np.divide(sensitivity1, mean1, out=np.zeros_like(sensitivity1, dtype=float), where=sensitivity1!=0) #normalized

# hdu.close()

# hdu = fits.open("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\SenseConfig\\wfss-grism-configuration\\NIRISS.GR150R.F200W.0.etc.1.5.2.sens.fits") #F200W, GR150R
# hdu.info()
# #header0= hdu[1].header
# #print(header0)
# data0= hdu[1].data
# wavelength0 = data0["WAVELENGTH"]
# sensitivity0 = data0["SENSITIVITY"] 
# sens0 = np.divide(sensitivity0, mean1, out=np.zeros_like(sensitivity1, dtype=float), where=sensitivity1!=0) #normalized by the same factor as 1st order

# hdu.close()

# hdu = fits.open("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\SenseConfig\\wfss-grism-configuration\\NIRISS.GR150R.F200W.2.etc.1.5.2.sens.fits") #F200W, GR150R
# hdu.info()
# #header0= hdu[1].header
# #print(header0)
# data2= hdu[1].data
# wavelength2 = data2["WAVELENGTH"]
# sensitivity2 = data2["SENSITIVITY"] 
# sens2 = np.divide(sensitivity2, mean1, out=np.zeros_like(sensitivity1, dtype=float), where=sensitivity1!=0) #normalized by the same factor as 1st order

# hdu.close()

# plt.subplot(1,4,1)

# plt.plot(wavelength1, sens1)
# plt.xlabel("Wavelength [Å]")
# plt.ylabel("Sensitivity")
# plt.title("1st order")

# plt.subplot(1,4,2)
# plt.plot(wavelength0, sens0)
# plt.xlabel("Wavelength [Å]")
# plt.ylabel("Sensitivity")
# plt.title("0th order")

# plt.subplot(1,4,3)
# plt.plot(wavelength2, sens2)
# plt.xlabel("Wavelength [Å]")
# plt.ylabel("Sensitivity")
# plt.title("2nd order")

# plt.subplot(1,4,4)
# plt.plot(wavelength2, sens1+sens2+sens0)
# plt.xlabel("Wavelength [Å]")
# plt.ylabel("Sensitivity")
# plt.title("all order")
# plt.show()

# #########################
# #Trying out _lam_range
# ########################
# # tr = GrismTrace.from_file("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F200W.220725.conf",filter_name="F200W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")  # auto-detects format

# # lo, hi = tr._lam_range("1", None, None) #minimum and maximum wavelength in microns for first
# # print(lo)
# # print(hi)
# # lo, hi = tr._lam_range("0", None, None) #minimum and maximum wavelength in microns for first
# # print(lo)
# # print(hi)
# # lo, hi = tr._lam_range("-1", None, None) #minimum and maximum wavelength in microns for first
# # print(lo)
# # print(hi)
# # lo, hi = tr._lam_range("2", None, None) #minimum and maximum wavelength in microns for first
# # print(lo)
# # print(hi)
# # print(tr.orders)

###############################################
# function to mask nans
#######################################
def nan_local_mean(arr, size=5, mode='reflect'):
    """
    Replace NaNs with mean of surrounding entries in a (size x size) window.
    
    size: odd integer (e.g. 3, 5)
    mode: boundary handling ('reflect', 'nearest', 'constant', ...)
    """
    arr = np.asarray(arr, dtype=float)

    # mask of valid values
    valid_mask = ~np.isnan(arr)

    # replace NaNs with 0 for summation
    arr_filled = np.where(valid_mask, arr, 0.0)

    # kernel
    kernel = np.ones((size, size))

    # sum of neighbors
    local_sum = convolve(arr_filled, kernel, mode=mode)

    # count of valid neighbors
    local_count = convolve(valid_mask.astype(float), kernel, mode=mode)
    
    # compute mean safely
    local_mean = np.zeros_like(arr, dtype=float)
    nonzero_mask = local_count > 0
    local_mean[nonzero_mask] = local_sum[nonzero_mask] / local_count[nonzero_mask]


    # fill only NaNs
    result = arr.copy()
    result[~valid_mask] = local_mean[~valid_mask]

    return result

# ###################################
# # Original direct tryout
# ######################################
# # # just do the two following lines once to build the matrix
#A = build_matrix("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F150W.220725.conf",filter_name="F150W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")
#A.build_and_save_trace_matrix_sensitivities_all_orders()
# base = Path(__file__).resolve().parent

# hdu_1 = fits.open(base / "RateFiles" / "Match" / "jw01090001001_28101_00001_nis_rate.fits") #F200W, GR150R
# hdu_1.info()

# image_data = hdu_1['SCI'].data
# hdu_1.close()

# direct_masked = np.array(nan_local_mean(image_data))

# direct_masked = direct_masked[0:500,0:20]
# np.save("original_direct_20_500_jw01090001001_28101_00001_nis_rate.npy", direct_masked)


# disp = dispersion()
# start = time.time()
# dispersed = disp.compute_dispersed_linear_sensitivities(direct_masked)
# np.save("dispersed_uniform_102order_sens_adapted_20_500_jw01090001001_28101_00001_nis_rate.npy", dispersed)
# end = time.time()
# print(f"Dispersion Time: {end - start:.2f} seconds")



# recov = recovery()
# recovered = recov.recover_direct_from_traces_basis_matrix_PCA(dispersed)
# np.save("recovered_uniform_102order_sens_adapted_20_500_jw01090001001_28101_00001_nis_rate.npy",recovered)




###################################################
#Recovery just from dispersed data
#################################################
# do the following two lines just once to build matrix
#H = build_matrix("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F150W.220725.conf",filter_name="F150W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")
#H.build_and_save_trace_matrix_coefficients_PCA_sensitivity()

# base = Path(__file__).resolve().parent
# hdu_1 = fits.open(base / "RateFiles" / "Match" / "jw01090001001_27101_00004_nis_rate.fits") #F200W, GR150R
# hdu_1.info()
# print(hdu_1[0].header["PUPIL"])

# image_data_dispersed = hdu_1['SCI'].data
# hdu_1.close()


# dispersed_masked = np.array(nan_local_mean(image_data_dispersed))

# dispersed_masked = dispersed_masked[0:500,0:20]

# np.save("original_dispersed_20_500_jw01090001001_27101_00004_nis_rate.npy", dispersed_masked)


# recov = recovery()
# recovered = recov.recover_direct_from_traces_basis_matrix_PCA(dispersed_masked) 
# np.save("recovered_F150W_PCA_sens_orders_20_500_jw01090001001_27101_00004_nis_rate.npy", recovered)


# #####################################################################
# loading saved matrices
################################################################
base = Path(__file__).resolve().parent
# based on the original direct image
#original_direct = np.load(base / "original_direct_100_500_jw01090001001_34101_00001_nis_rate.npy")
#dispersed = np.load(base / "dispersed_uniform_sensitivities_102orders_100_500_jw01090001001_34101_00001_nis_rate.npy")
#recovered_og_direct =  np.load(base / "recovered_uniform_sensitivities_102orders_100_500_jw01090001001_34101_00001_nis_rate.npy")

# based on original dispersed
#original_dispersed = np.load(base / "original_dispersed_100_500_jw01090001001_39101_00002_nis_rate.npy")
#recovered_og_dispersed = np.load(base / "recovered_uniform_sensitivities_102orders_100_500_jw01090001001_39101_00002_nis_rate.npy")

# based on the original direct image
original_direct = np.load(base / "original_direct_20_500_jw01090001001_28101_00001_nis_rate.npy")
dispersed = np.load(base / "dispersed_uniform_102order_sens_adapted_20_500_jw01090001001_28101_00001_nis_rate.npy")
recovered_og_direct =  np.load(base / "recovered_uniform_102order_sens_adapted_20_500_jw01090001001_28101_00001_nis_rate.npy")

# based on original dispersed
original_dispersed = np.load(base / "original_dispersed_20_500_jw01090001001_27101_00004_nis_rate.npy")
#original_dispersed = original_dispersed[0:500, 0:100]
recovered_og_dispersed = np.load(base / "recovered_F150W_PCA_sens_orders_20_500_jw01090001001_27101_00004_nis_rate.npy")


# ####################################################################
# # plot saved matrices
# ##################################################################
# # based on the original direct image
plt.subplot(1,4,1)
std1 = np.nanstd(original_direct)
mean1 = np.nanmean(original_direct)
plt.imshow(original_direct, cmap="inferno", vmin=0, vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Original Direct")


plt.subplot(1,4,2)
std2 = np.nanstd(dispersed)
mean2 = np.nanmean(dispersed)
plt.imshow(dispersed, cmap="inferno", vmin=0, vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Dispersed")


plt.subplot(1,4,3)
std3 = np.nanstd(recovered_og_direct)
mean3 = np.nanmean(recovered_og_direct)
max = np.max(recovered_og_direct)
plt.imshow(recovered_og_direct, cmap="inferno", vmin=0, vmax=max/10, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Recovered_from_Dispersed")


plt.subplot(1,4,4)
std4 = np.nanstd(original_direct -recovered_og_direct)
mean4 = np.nanmean(original_direct -recovered_og_direct)
plt.imshow(original_direct -recovered_og_direct, cmap="inferno", vmin=-(mean4 + 2*std4), vmax=mean4 + 2*std4, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Residuals: Original_Direct-Recovered_from_Dispersed")

plt.show()

# snippet for spectrum of a specific trace

one_trace = dispersed[320:400, 15:17]
sum = one_trace.sum(axis=1)
x = np.arange(len(sum))
plt.subplot(1,4,1)
plt.plot(sum, x)
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Manual Dispersion")

plt.subplot(1,4,2)
plt.imshow(one_trace)

plt.title("Trace Manual Dispersion")


one_trace_disp = original_dispersed[125:205, 10:15]
sum_disp = one_trace_disp.sum(axis=1)
x = np.arange(len(sum))
plt.subplot(1,4,3)
plt.plot(sum_disp, x)
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Original dispersion")


plt.subplot(1,4,4)
plt.imshow(one_trace_disp)
plt.title("Trace Original dispersion")
plt.show()
# based on original dispersed

plt.subplot(1,3,1)
std1 = np.nanstd(original_dispersed)
mean1 = np.nanmean(original_dispersed)
plt.imshow(original_dispersed, cmap="inferno", vmin=-(mean1 + 2*std1), vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Original Dispersed")

plt.subplot(1,3,2)
std2 = np.nanstd(recovered_og_dispersed)
mean2 = np.nanmean(recovered_og_dispersed)
max = np.max(recovered_og_dispersed)
plt.imshow(recovered_og_dispersed, cmap="inferno", vmin=-(max/10), vmax=max/10, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Recovered from original Dispersed")

plt.subplot(1,3,3)
clipped_residual = sigma_clip(original_direct -recovered_og_dispersed)
std3 = np.nanstd(clipped_residual)
mean3 = np.nanmean(clipped_residual)
plt.imshow(clipped_residual, cmap="inferno", vmin=-(mean3 + 2*std3), vmax=mean3 + 2*std3, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()
plt.title("Residuals: Original Direct - Recovered from original Dispersed")

plt.show()



