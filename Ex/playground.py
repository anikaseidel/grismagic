from grismagic.traces import GrismTrace
from build_matrix import build_matrix
from dispersion import dispersion
from recovery import recovery
import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import time 

#########################
#Trying out _lam_range
########################
# tr = GrismTrace.from_file("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F200W.220725.conf",filter_name="F200W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")  # auto-detects format

# lo, hi = tr._lam_range("1", None, None) #minimum and maximum wavelength in microns for first
# print(lo)
# print(hi)
# lo, hi = tr._lam_range("0", None, None) #minimum and maximum wavelength in microns for first
# print(lo)
# print(hi)
# lo, hi = tr._lam_range("-1", None, None) #minimum and maximum wavelength in microns for first
# print(lo)
# print(hi)
# lo, hi = tr._lam_range("2", None, None) #minimum and maximum wavelength in microns for first
# print(lo)
# print(hi)
# print(tr.orders)

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

###################################
# Original direct tryout
######################################
# # just do the two following lines once to build the matrix
# #A = build_matrix("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F200W.220725.conf",filter_name="F200W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")
# #A.build_and_save_trace_matrix()

hdu_1 = fits.open("Rate Files\\1090\\jw01090001001_34101_00001_nis_rate.fits") #F200W, GR150R
hdu_1.info()

image_data = hdu_1['SCI'].data
hdu_1.close()

direct_masked = np.array(nan_local_mean(image_data))

direct_masked = direct_masked[0:500,0:500]

plt.figure()
std1 = np.nanstd(direct_masked)
mean1 = np.nanmean(direct_masked)
plt.imshow(direct_masked, cmap="inferno", vmin=0, vmax=mean1 + 2*std1)
plt.colorbar()
#plt.title(fitsfile)
plt.show()

disp = dispersion()
start = time.time()
dispersed = disp.compute_dispersed_linear(direct_masked)
end = time.time()
print(f"Dispersion Time: {end - start:.2f} seconds")

plt.figure()
std2 = np.nanstd(dispersed)
mean2 = np.nanmean(dispersed)
plt.imshow(dispersed, cmap="inferno", vmin=0, vmax=mean2 + 2*std2)
plt.colorbar()
#plt.title(fitsfile)
plt.show()

recov = recovery()
recovered = recov.recover_direct_from_traces_matrix(dispersed)
plt.figure()
std3 = np.nanstd(recovered)
mean3 = np.nanmean(recovered)
plt.imshow(recovered, cmap="inferno", vmin=0, vmax=mean3 + 2*std3)
plt.colorbar()
plt.show()

plt.subplot(1,5,1)
plt.imshow(direct_masked, cmap="inferno", vmin=0, vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()

plt.subplot(1,5,2)
plt.imshow(dispersed, cmap="inferno", vmin=0, vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()


plt.subplot(1,5,3)
plt.imshow(recovered, cmap="inferno", vmin=0, vmax=mean3 + 2*std3, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()


plt.subplot(1,5,4)
std4 = np.nanstd(direct_masked -recovered)
mean4 = np.nanmean(direct_masked -recovered)
plt.imshow(direct_masked - recovered, cmap="inferno", vmin=0, vmax=mean4 + 2*std4, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()

plt.subplot(1,5,5)
std4 = np.nanstd(direct_masked + dispersed)
mean4 = np.nanmean(direct_masked + dispersed)
plt.imshow(direct_masked/5 + dispersed, cmap="inferno", vmin=0, vmax=mean4 + 2*std4, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()

plt.show()


###################################################
#Recovery just from dispersed data
#################################################
# do the following two lines just once to build matrix
#H = build_matrix("C:\\Users\\anika\\GitHub\\grismagic\\Ex\\Config Files\\GR150R.F200W.220725.conf",filter_name="F200W",wavelengthrange_file="C:\\Users\\anika\\GitHub\\grismagic\\Ex\\jwst_niriss_wavelengthrange_0002.asdf")
#H.build_and_save_trace_matrix_coefficients_orders()

hdu_1 = fits.open("Rate Files\\1090\\jw01090001001_39101_00002_nis_rate.fits") #F200W, GR150R
hdu_1.info()
print(hdu_1[0].header["PUPIL"])

image_data_dispersed = hdu_1['SCI'].data
hdu_1.close()


dispersed_masked = np.array(nan_local_mean(image_data_dispersed))

dispersed_masked = dispersed_masked[0:500,0:500]
std1 = np.nanstd(dispersed_masked)
mean1 = np.nanmean(dispersed_masked)

#dispersed_masked[dispersed_masked<std1]=0

plt.figure()

plt.imshow(dispersed_masked, cmap="inferno", vmin=0, vmax=mean1 + 2*std1)
plt.colorbar()
#plt.title(fitsfile)
plt.show()

recov = recovery()
recovered = recov.recover_via_lsqr_bounds(dispersed_masked) #now with uniform dist
plt.figure()
std2 = np.nanstd(recovered)
mean2 = np.nanmean(recovered)
plt.imshow(recovered, cmap="inferno", vmin=0, vmax=mean2 + 2*std2)
plt.colorbar()
#plt.title(fitsfile)
plt.show()

plt.subplot(1,3,1)
plt.imshow(dispersed_masked, cmap="inferno", vmin=0, vmax=mean1 + 2*std1, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(recovered, cmap="inferno", vmin=0, vmax=mean2 + 2*std2, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()

plt.subplot(1,3,3)
std3 = np.nanstd(direct_masked -recovered)
mean3 = np.nanmean(direct_masked -recovered)
plt.imshow(direct_masked - recovered, cmap="inferno", vmin=-(mean3 + 2*std3), vmax=mean3 + 2*std3, interpolation="nearest", origin="lower",aspect="auto")
plt.colorbar()

plt.show()

#ToDos
#sigma clipping
#sensitivity curves included in matrix
#model spectra comparing real trace to predicted trace
