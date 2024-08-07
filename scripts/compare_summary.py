import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
L1_VB_var2_256 = np.mean(var2_256_VB[:,1])
print('L1_VB_var2_256: ', L1_VB_var2_256)

L2_VB_var2_256 = np.mean(var2_256_VB[:,2])
print('L2_VB_var2_256: ', L2_VB_var2_256)

unif_width_11_var2_256_VB = np.median(var2_256_VB[:,3])
print('unif_width_11_var2_256_VB: ', unif_width_11_var2_256_VB)

unif_width_re_12_var2_256_VB = np.median(var2_256_VB[:,4])
print('unif_width_re_12_var2_256_VB: ', unif_width_re_12_var2_256_VB)

unif_width_im_21_var2_256_VB = np.median(var2_256_VB[:,5])
print('unif_width_im_21_var2_256_VB: ', unif_width_im_21_var2_256_VB)

unif_width_22_var2_256_VB = np.median(var2_256_VB[:,6])
print('unif_width_22_var2_256_VB: ', unif_width_22_var2_256_VB)

unif_coverage_var2_256_VB = np.mean(var2_256_VB[:,7])
print('unif_coverage_var2_256_VB: ', unif_coverage_var2_256_VB)

point_width_11_var2_256_VB = np.median(var2_256_VB[:,8])
print('point_width_11_var2_256_VB: ', point_width_11_var2_256_VB)

point_width_re_12_var2_256_VB = np.median(var2_256_VB[:,9])
print('point_width_re_12_var2_256_VB: ', point_width_re_12_var2_256_VB)

point_width_im_21_var2_256_VB = np.median(var2_256_VB[:,10])
print('point_width_im_21_var2_256_VB: ', point_width_im_21_var2_256_VB)

point_width_22_var2_256_VB = np.median(var2_256_VB[:,11])
print('point_width_22_var2_256_VB: ', point_width_22_var2_256_VB)

point_coverage_var2_256_VB = np.mean(var2_256_VB[:,12])
print('point_coverage_var2_256_VB: ', point_coverage_var2_256_VB)

#------------------------------------------------------------------------------------------------------------
#VNPC VAR2 256
var2_256_VNPC = pd.read_csv('VAR2_256_errors_coverages_len_VNPC_combined_results.csv').values

L1_VNPC_var2_256 = np.mean(var2_256_VNPC[:,1])
print('L1_VNPC_var2_256: ', L1_VNPC_var2_256)

L2_VNPC_var2_256 = np.mean(var2_256_VNPC[:,2])
print('L2_VNPC_var2_256: ', L2_VNPC_var2_256)

unif_width_11_var2_256_VNPC = np.median(var2_256_VNPC[:,3])
print('unif_width_11_var2_256_VNPC: ', unif_width_11_var2_256_VNPC)

unif_width_re_12_var2_256_VNPC = np.median(var2_256_VNPC[:,4])
print('unif_width_re_12_var2_256_VNPC: ', unif_width_re_12_var2_256_VNPC)

unif_width_im_21_var2_256_VNPC = np.median(var2_256_VNPC[:,5])
print('unif_width_im_21_var2_256_VNPC: ', unif_width_im_21_var2_256_VNPC)

unif_width_22_var2_256_VNPC = np.median(var2_256_VNPC[:,6])
print('unif_width_22_var2_256_VNPC: ', unif_width_22_var2_256_VNPC)

unif_coverage_var2_256_VNPC = np.mean(var2_256_VNPC[:,7])
print('unif_coverage_var2_256_VNPC: ', unif_coverage_var2_256_VNPC)

point_width_11_var2_256_VNPC = np.median(var2_256_VNPC[:,8])
print('point_width_11_var2_256_VNPC: ', point_width_11_var2_256_VNPC)

point_width_re_12_var2_256_VNPC = np.median(var2_256_VNPC[:,9])
print('point_width_re_12_var2_256_VNPC: ', point_width_re_12_var2_256_VNPC)

point_width_im_21_var2_256_VNPC = np.median(var2_256_VNPC[:,10])
print('point_width_im_21_var2_256_VNPC: ', point_width_im_21_var2_256_VNPC)

point_width_22_var2_256_VNPC = np.median(var2_256_VNPC[:,11])
print('point_width_22_var2_256_VNPC: ', point_width_22_var2_256_VNPC)

point_coverage_var2_256_VNPC = np.mean(var2_256_VNPC[:,12])
print('point_coverage_var2_256_VNPC: ', point_coverage_var2_256_VNPC)

#-----------------------------------------------------------------------------------------------------------
#VB VAR2 512
var2_512_VB = pd.read_csv('VAR2_512_errors_coverages_len_VB_combined_results.csv').values

L1_VB_var2_512 = np.mean(var2_512_VB[:,1])
print('L1_VB_var2_512: ', L1_VB_var2_512)

L2_VB_var2_512 = np.mean(var2_512_VB[:,2])
print('L2_VB_var2_512: ', L2_VB_var2_512)

unif_width_11_var2_512_VB = np.median(var2_512_VB[:,3])
print('unif_width_11_var2_512_VB: ', unif_width_11_var2_512_VB)

unif_width_re_12_var2_512_VB = np.median(var2_512_VB[:,4])
print('unif_width_re_12_var2_512_VB: ', unif_width_re_12_var2_512_VB)

unif_width_im_21_var2_512_VB = np.median(var2_512_VB[:,5])
print('unif_width_im_21_var2_512_VB: ', unif_width_im_21_var2_512_VB)

unif_width_22_var2_512_VB = np.median(var2_512_VB[:,6])
print('unif_width_22_var2_512_VB: ', unif_width_22_var2_512_VB)

unif_coverage_var2_512_VB = np.mean(var2_512_VB[:,7])
print('unif_coverage_var2_512_VB: ', unif_coverage_var2_512_VB)

point_width_11_var2_512_VB = np.median(var2_512_VB[:,8])
print('point_width_11_var2_512_VB: ', point_width_11_var2_512_VB)

point_width_re_12_var2_512_VB = np.median(var2_512_VB[:,9])
print('point_width_re_12_var2_512_VB: ', point_width_re_12_var2_512_VB)

point_width_im_21_var2_512_VB = np.median(var2_512_VB[:,10])
print('point_width_im_21_var2_512_VB: ', point_width_im_21_var2_512_VB)

point_width_22_var2_512_VB = np.median(var2_512_VB[:,11])
print('point_width_22_var2_512_VB: ', point_width_22_var2_512_VB)

point_coverage_var2_512_VB = np.mean(var2_512_VB[:,12])
print('point_coverage_var2_512_VB: ', point_coverage_var2_512_VB)

#-----------------------------------------------------------------------------------------------------------
#VNPC VAR2 512
var2_512_VNPC = pd.read_csv('VAR2_512_errors_coverages_len_VNPC_combined_results.csv').values

L1_VNPC_var2_512 = np.mean(var2_512_VNPC[:,1])
print('L1_VNPC_var2_512: ', L1_VNPC_var2_512)

L2_VNPC_var2_512 = np.mean(var2_512_VNPC[:,2])
print('L2_VNPC_var2_512: ', L2_VNPC_var2_512)

unif_width_11_var2_512_VNPC = np.median(var2_512_VNPC[:,3])
print('unif_width_11_var2_512_VNPC: ', unif_width_11_var2_512_VNPC)

unif_width_re_12_var2_512_VNPC = np.median(var2_512_VNPC[:,4])
print('unif_width_re_12_var2_512_VNPC: ', unif_width_re_12_var2_512_VNPC)

unif_width_im_21_var2_512_VNPC = np.median(var2_512_VNPC[:,5])
print('unif_width_im_21_var2_512_VNPC: ', unif_width_im_21_var2_512_VNPC)

unif_width_22_var2_512_VNPC = np.median(var2_512_VNPC[:,6])
print('unif_width_22_var2_512_VNPC: ', unif_width_22_var2_512_VNPC)

unif_coverage_var2_512_VNPC = np.mean(var2_512_VNPC[:,7])
print('unif_coverage_var2_512_VNPC: ', unif_coverage_var2_512_VNPC)

point_width_11_var2_512_VNPC = np.median(var2_512_VNPC[:,8])
print('point_width_11_var2_512_VNPC: ', point_width_11_var2_512_VNPC)

point_width_re_12_var2_512_VNPC = np.median(var2_512_VNPC[:,9])
print('point_width_re_12_var2_512_VNPC: ', point_width_re_12_var2_512_VNPC)

point_width_im_21_var2_512_VNPC = np.median(var2_512_VNPC[:,10])
print('point_width_im_21_var2_512_VNPC: ', point_width_im_21_var2_512_VNPC)

point_width_22_var2_512_VNPC = np.median(var2_512_VNPC[:,11])
print('point_width_22_var2_512_VNPC: ', point_width_22_var2_512_VNPC)

point_coverage_var2_512_VNPC = np.mean(var2_512_VNPC[:,12])
print('point_coverage_var2_512_VNPC: ', point_coverage_var2_512_VNPC)

#-----------------------------------------------------------------------------------------------------------
#VB VAR2 1024
var2_1024_VB = pd.read_csv('VAR2_1024_errors_coverages_len_VB_combined_results.csv').values

L1_VB_var2_1024 = np.mean(var2_1024_VB[:,1])
print('L1_VB_var2_1024: ', L1_VB_var2_1024)

L2_VB_var2_1024 = np.mean(var2_1024_VB[:,2])
print('L2_VB_var2_1024: ', L2_VB_var2_1024)

unif_width_11_var2_1024_VB = np.median(var2_1024_VB[:,3])
print('unif_width_11_var2_1024_VB: ', unif_width_11_var2_1024_VB)

unif_width_re_12_var2_1024_VB = np.median(var2_1024_VB[:,4])
print('unif_width_re_12_var2_1024_VB: ', unif_width_re_12_var2_1024_VB)

unif_width_im_21_var2_1024_VB = np.median(var2_1024_VB[:,5])
print('unif_width_im_21_var2_1024_VB: ', unif_width_im_21_var2_1024_VB)

unif_width_22_var2_1024_VB = np.median(var2_1024_VB[:,6])
print('unif_width_22_var2_1024_VB: ', unif_width_22_var2_1024_VB)

unif_coverage_var2_1024_VB = np.mean(var2_1024_VB[:,7])
print('unif_coverage_var2_1024_VB: ', unif_coverage_var2_1024_VB)

point_width_11_var2_1024_VB = np.median(var2_1024_VB[:,8])
print('point_width_11_var2_1024_VB: ', point_width_11_var2_1024_VB)

point_width_re_12_var2_1024_VB = np.median(var2_1024_VB[:,9])
print('point_width_re_12_var2_1024_VB: ', point_width_re_12_var2_1024_VB)

point_width_im_21_var2_1024_VB = np.median(var2_1024_VB[:,10])
print('point_width_im_21_var2_1024_VB: ', point_width_im_21_var2_1024_VB)

point_width_22_var2_1024_VB = np.median(var2_1024_VB[:,11])
print('point_width_22_var2_1024_VB: ', point_width_22_var2_1024_VB)

point_coverage_var2_1024_VB = np.mean(var2_1024_VB[:,12])
print('point_coverage_var2_1024_VB: ', point_coverage_var2_1024_VB)

#-----------------------------------------------------------------------------------------------------------
#VNPC VAR2 1024
var2_1024_VNPC = pd.read_csv('VAR2_1024_errors_coverages_len_VNPC_combined_results.csv').values

L1_VNPC_var2_1024 = np.mean(var2_1024_VNPC[:,1])
print('L1_VNPC_var2_1024: ', L1_VNPC_var2_1024)

L2_VNPC_var2_1024 = np.mean(var2_1024_VNPC[:,2])
print('L2_VNPC_var2_1024: ', L2_VNPC_var2_1024)

unif_width_11_var2_1024_VNPC = np.median(var2_1024_VNPC[:,3])
print('unif_width_11_var2_1024_VNPC: ', unif_width_11_var2_1024_VNPC)

unif_width_re_12_var2_1024_VNPC = np.median(var2_1024_VNPC[:,4])
print('unif_width_re_12_var2_1024_VNPC: ', unif_width_re_12_var2_1024_VNPC)

unif_width_im_21_var2_1024_VNPC = np.median(var2_1024_VNPC[:,5])
print('unif_width_im_21_var2_1024_VNPC: ', unif_width_im_21_var2_1024_VNPC)

unif_width_22_var2_1024_VNPC = np.median(var2_1024_VNPC[:,6])
print('unif_width_22_var2_1024_VNPC: ', unif_width_22_var2_1024_VNPC)

unif_coverage_var2_1024_VNPC = np.mean(var2_1024_VNPC[:,7])
print('unif_coverage_var2_1024_VNPC: ', unif_coverage_var2_1024_VNPC)

point_width_11_var2_1024_VNPC = np.median(var2_1024_VNPC[:,8])
print('point_width_11_var2_1024_VNPC: ', point_width_11_var2_1024_VNPC)

point_width_re_12_var2_1024_VNPC = np.median(var2_1024_VNPC[:,9])
print('point_width_re_12_var2_1024_VNPC: ', point_width_re_12_var2_1024_VNPC)

point_width_im_21_var2_1024_VNPC = np.median(var2_1024_VNPC[:,10])
print('point_width_im_21_var2_1024_VNPC: ', point_width_im_21_var2_1024_VNPC)

point_width_22_var2_1024_VNPC = np.median(var2_1024_VNPC[:,11])
print('point_width_22_var2_1024_VNPC: ', point_width_22_var2_1024_VNPC)

point_coverage_var2_1024_VNPC = np.mean(var2_1024_VNPC[:,12])
print('point_coverage_var2_1024_VNPC: ', point_coverage_var2_1024_VNPC)

#-----------------------------------------------------------------------------------------------------------
#VB VMA1 256
import os
os.chdir('D:/VB var2 errors len coverage')
vma1_256_VB = pd.read_csv('VMA1_256_errors_coverages_len_VB_combined_results.csv').values

L1_VB_vma1_256 = np.mean(vma1_256_VB[:,1])
print('L1_VB_vma1_256: ', L1_VB_vma1_256)

L2_VB_vma1_256 = np.mean(vma1_256_VB[:,2])
print('L2_VB_vma1_256: ', L2_VB_vma1_256)

unif_width_11_vma1_256_VB = np.median(vma1_256_VB[:,3])
print('unif_width_11_vma1_256_VB: ', unif_width_11_vma1_256_VB)

unif_width_re_12_vma1_256_VB = np.median(vma1_256_VB[:,4])
print('unif_width_re_12_vma1_256_VB: ', unif_width_re_12_vma1_256_VB)

unif_width_im_21_vma1_256_VB = np.median(vma1_256_VB[:,5])
print('unif_width_im_21_vma1_256_VB: ', unif_width_im_21_vma1_256_VB)

unif_width_22_vma1_256_VB = np.median(vma1_256_VB[:,6])
print('unif_width_22_vma1_256_VB: ', unif_width_22_vma1_256_VB)

unif_coverage_vma1_256_VB = np.mean(vma1_256_VB[:,7])
print('unif_coverage_vma1_256_VB: ', unif_coverage_vma1_256_VB)

point_width_11_vma1_256_VB = np.median(vma1_256_VB[:,8])
print('point_width_11_vma1_256_VB: ', point_width_11_vma1_256_VB)

point_width_re_12_vma1_256_VB = np.median(vma1_256_VB[:,9])
print('point_width_re_12_vma1_256_VB: ', point_width_re_12_vma1_256_VB)

point_width_im_21_vma1_256_VB = np.median(vma1_256_VB[:,10])
print('point_width_im_21_vma1_256_VB: ', point_width_im_21_vma1_256_VB)

point_width_22_vma1_256_VB = np.median(vma1_256_VB[:,11])
print('point_width_22_vma1_256_VB: ', point_width_22_vma1_256_VB)

point_coverage_vma1_256_VB = np.mean(vma1_256_VB[:,12])
print('point_coverage_vma1_256_VB: ', point_coverage_vma1_256_VB)

#------------------------------------------------------------------------------------------------------------
#VNPC VMA1 256
vma1_256_VNPC = pd.read_csv('VMA1_256_errors_coverages_len_VNPC_combined_results.csv').values

L1_VNPC_vma1_256 = np.mean(vma1_256_VNPC[:,1])
print('L1_VNPC_vma1_256: ', L1_VNPC_vma1_256)

L2_VNPC_vma1_256 = np.mean(vma1_256_VNPC[:,2])
print('L2_VNPC_vma1_256: ', L2_VNPC_vma1_256)

unif_width_11_vma1_256_VNPC = np.median(vma1_256_VNPC[:,3])
print('unif_width_11_vma1_256_VNPC: ', unif_width_11_vma1_256_VNPC)

unif_width_re_12_vma1_256_VNPC = np.median(var2_256_VNPC[:,4])
print('unif_width_re_12_vma1_256_VNPC: ', unif_width_re_12_vma1_256_VNPC)

unif_width_im_21_vma1_256_VNPC = np.median(vma1_256_VNPC[:,5])
print('unif_width_im_21_vma1_256_VNPC: ', unif_width_im_21_vma1_256_VNPC)

unif_width_22_vma1_256_VNPC = np.median(vma1_256_VNPC[:,6])
print('unif_width_22_vma1_256_VNPC: ', unif_width_22_vma1_256_VNPC)

unif_coverage_vma1_256_VNPC = np.mean(vma1_256_VNPC[:,7])
print('unif_coverage_vma1_256_VNPC: ', unif_coverage_vma1_256_VNPC)

point_width_11_vma1_256_VNPC = np.median(vma1_256_VNPC[:,8])
print('point_width_11_vma1_256_VNPC: ', point_width_11_vma1_256_VNPC)

point_width_re_12_vma1_256_VNPC = np.median(vma1_256_VNPC[:,9])
print('point_width_re_12_vma1_256_VNPC: ', point_width_re_12_vma1_256_VNPC)

point_width_im_21_vma1_256_VNPC = np.median(vma1_256_VNPC[:,10])
print('point_width_im_21_vma1_256_VNPC: ', point_width_im_21_vma1_256_VNPC)

point_width_22_vma1_256_VNPC = np.median(vma1_256_VNPC[:,11])
print('point_width_22_vma1_256_VNPC: ', point_width_22_vma1_256_VNPC)

point_coverage_vma1_256_VNPC = np.mean(vma1_256_VNPC[:,12])
print('point_coverage_vma1_256_VNPC: ', point_coverage_vma1_256_VNPC)

#-------------------------------------------------------------------------------------------------------------
#VB VMA1 512
vma1_512_VB = pd.read_csv('VMA1_512_errors_coverages_len_VB_combined_results.csv').values

L1_VB_vma1_512 = np.mean(vma1_512_VB[:,1])
print('L1_VB_vma1_512: {:.3f}'.format(L1_VB_vma1_512))

L2_VB_vma1_512 = np.mean(vma1_512_VB[:,2])
print('L2_VB_vma1_512: {:.3f}'.format(L2_VB_vma1_512))

unif_width_11_vma1_512_VB = np.median(vma1_512_VB[:,3])
print('unif_width_11_vma1_512_VB: {:.3f}'.format(unif_width_11_vma1_512_VB))

unif_width_re_12_vma1_512_VB = np.median(vma1_512_VB[:,4])
print('unif_width_re_12_vma1_512_VB: {:.3f}'.format(unif_width_re_12_vma1_512_VB))

unif_width_im_21_vma1_512_VB = np.median(vma1_512_VB[:,5])
print('unif_width_im_21_vma1_512_VB: {:.3f}'.format(unif_width_im_21_vma1_512_VB))

unif_width_22_vma1_512_VB = np.median(vma1_512_VB[:,6])
print('unif_width_22_vma1_512_VB: {:.3f}'.format(unif_width_22_vma1_512_VB))

unif_coverage_vma1_512_VB = np.mean(vma1_512_VB[:,7])
print('unif_coverage_vma1_512_VB: {:.3f}'.format(unif_coverage_vma1_512_VB))

point_width_11_vma1_512_VB = np.median(vma1_512_VB[:,8])
print('point_width_11_vma1_512_VB: {:.3f}'.format(point_width_11_vma1_512_VB))

point_width_re_12_vma1_512_VB = np.median(vma1_512_VB[:,9])
print('point_width_re_12_vma1_512_VB: {:.3f}'.format(point_width_re_12_vma1_512_VB))

point_width_im_21_vma1_512_VB = np.median(vma1_512_VB[:,10])
print('point_width_im_21_vma1_512_VB: {:.3f}'.format(point_width_im_21_vma1_512_VB))

point_width_22_vma1_512_VB = np.median(vma1_512_VB[:,11])
print('point_width_22_vma1_512_VB: {:.3f}'.format(point_width_22_vma1_512_VB))

point_coverage_vma1_512_VB = np.mean(vma1_512_VB[:,12])
print('point_coverage_vma1_512_VB: {:.3f}'.format(point_coverage_vma1_512_VB))

#--------------------------------------------------------------------------------------------------------------
# VNPC VMA1 512
vma1_512_VNPC = pd.read_csv('VMA1_512_errors_coverages_len_VNPC_combined_results.csv').values

L1_VNPC_vma1_512 = np.mean(vma1_512_VNPC[:,1])
print('L1_VNPC_vma1_512: {:.3f}'.format(L1_VNPC_vma1_512))

L2_VNPC_vma1_512 = np.mean(vma1_512_VNPC[:,2])
print('L2_VNPC_vma1_512: {:.3f}'.format(L2_VNPC_vma1_512))

unif_width_11_vma1_512_VNPC = np.median(vma1_512_VNPC[:,3])
print('unif_width_11_vma1_512_VNPC: {:.3f}'.format(unif_width_11_vma1_512_VNPC))

unif_width_re_12_vma1_512_VNPC = np.median(vma1_512_VNPC[:,4])
print('unif_width_re_12_vma1_512_VNPC: {:.3f}'.format(unif_width_re_12_vma1_512_VNPC))

unif_width_im_21_vma1_512_VNPC = np.median(vma1_512_VNPC[:,5])
print('unif_width_im_21_vma1_512_VNPC: {:.3f}'.format(unif_width_im_21_vma1_512_VNPC))

unif_width_22_vma1_512_VNPC = np.median(vma1_512_VNPC[:,6])
print('unif_width_22_vma1_512_VNPC: {:.3f}'.format(unif_width_22_vma1_512_VNPC))

unif_coverage_vma1_512_VNPC = np.mean(vma1_512_VNPC[:,7])
print('unif_coverage_vma1_512_VNPC: {:.3f}'.format(unif_coverage_vma1_512_VNPC))

point_width_11_vma1_512_VNPC = np.median(vma1_512_VNPC[:,8])
print('point_width_11_vma1_512_VNPC: {:.3f}'.format(point_width_11_vma1_512_VNPC))

point_width_re_12_vma1_512_VNPC = np.median(vma1_512_VNPC[:,9])
print('point_width_re_12_vma1_512_VNPC: {:.3f}'.format(point_width_re_12_vma1_512_VNPC))

point_width_im_21_vma1_512_VNPC = np.median(vma1_512_VNPC[:,10])
print('point_width_im_21_vma1_512_VNPC: {:.3f}'.format(point_width_im_21_vma1_512_VNPC))

point_width_22_vma1_512_VNPC = np.median(vma1_512_VNPC[:,11])
print('point_width_22_vma1_512_VNPC: {:.3f}'.format(point_width_22_vma1_512_VNPC))

point_coverage_vma1_512_VNPC = np.mean(vma1_512_VNPC[:,12])
print('point_coverage_vma1_512_VNPC: {:.3f}'.format(point_coverage_vma1_512_VNPC))

#-------------------------------------------------------------------------------------------------------------
#VB VMA1 1024
vma1_1024_VB = pd.read_csv('VMA1_1024_errors_coverages_len_VB_combined_results.csv').values

L1_VB_vma1_1024 = np.mean(vma1_1024_VB[:,1])
print('L1_VB_vma1_1024: {:.3f}'.format(L1_VB_vma1_1024))

L2_VB_vma1_1024 = np.mean(vma1_1024_VB[:,2])
print('L2_VB_vma1_1024: {:.3f}'.format(L2_VB_vma1_1024))

unif_width_11_vma1_1024_VB = np.median(vma1_1024_VB[:,3])
print('unif_width_11_vma1_1024_VB: {:.3f}'.format(unif_width_11_vma1_1024_VB))

unif_width_re_12_vma1_1024_VB = np.median(vma1_1024_VB[:,4])
print('unif_width_re_12_vma1_1024_VB: {:.3f}'.format(unif_width_re_12_vma1_1024_VB))

unif_width_im_21_vma1_1024_VB = np.median(vma1_1024_VB[:,5])
print('unif_width_im_21_vma1_1024_VB: {:.3f}'.format(unif_width_im_21_vma1_1024_VB))

unif_width_22_vma1_1024_VB = np.median(vma1_1024_VB[:,6])
print('unif_width_22_vma1_1024_VB: {:.3f}'.format(unif_width_22_vma1_1024_VB))

unif_coverage_vma1_1024_VB = np.mean(vma1_1024_VB[:,7])
print('unif_coverage_vma1_1024_VB: {:.3f}'.format(unif_coverage_vma1_1024_VB))

point_width_11_vma1_1024_VB = np.median(vma1_1024_VB[:,8])
print('point_width_11_vma1_1024_VB: {:.3f}'.format(point_width_11_vma1_1024_VB))

point_width_re_12_vma1_1024_VB = np.median(vma1_1024_VB[:,9])
print('point_width_re_12_vma1_1024_VB: {:.3f}'.format(point_width_re_12_vma1_1024_VB))

point_width_im_21_vma1_1024_VB = np.median(vma1_1024_VB[:,10])
print('point_width_im_21_vma1_1024_VB: {:.3f}'.format(point_width_im_21_vma1_1024_VB))

point_width_22_vma1_1024_VB = np.median(vma1_1024_VB[:,11])
print('point_width_22_vma1_1024_VB: {:.3f}'.format(point_width_22_vma1_1024_VB))

point_coverage_vma1_1024_VB = np.mean(vma1_1024_VB[:,12])
print('point_coverage_vma1_1024_VB: {:.3f}'.format(point_coverage_vma1_1024_VB))

#--------------------------------------------------------------------------------------------------------------
# VNPC VMA1 1024
vma1_1024_VNPC = pd.read_csv('VMA1_1024_errors_coverages_len_VNPC_combined_results.csv').values

L1_VNPC_vma1_1024 = np.mean(vma1_1024_VNPC[:,1])
print('L1_VNPC_vma1_1024: {:.3f}'.format(L1_VNPC_vma1_1024))

L2_VNPC_vma1_1024 = np.mean(vma1_1024_VNPC[:,2])
print('L2_VNPC_vma1_1024: {:.3f}'.format(L2_VNPC_vma1_1024))

unif_width_11_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,3])
print('unif_width_11_vma1_1024_VNPC: {:.3f}'.format(unif_width_11_vma1_1024_VNPC))

unif_width_re_12_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,4])
print('unif_width_re_12_vma1_1024_VNPC: {:.3f}'.format(unif_width_re_12_vma1_1024_VNPC))

unif_width_im_21_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,5])
print('unif_width_im_21_vma1_1024_VNPC: {:.3f}'.format(unif_width_im_21_vma1_1024_VNPC))

unif_width_22_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,6])
print('unif_width_22_vma1_1024_VNPC: {:.3f}'.format(unif_width_22_vma1_1024_VNPC))

unif_coverage_vma1_1024_VNPC = np.mean(vma1_1024_VNPC[:,7])
print('unif_coverage_vma1_1024_VNPC: {:.3f}'.format(unif_coverage_vma1_1024_VNPC))

point_width_11_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,8])
print('point_width_11_vma1_1024_VNPC: {:.3f}'.format(point_width_11_vma1_1024_VNPC))

point_width_re_12_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,9])
print('point_width_re_12_vma1_1024_VNPC: {:.3f}'.format(point_width_re_12_vma1_1024_VNPC))

point_width_im_21_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,10])
print('point_width_im_21_vma1_1024_VNPC: {:.3f}'.format(point_width_im_21_vma1_1024_VNPC))

point_width_22_vma1_1024_VNPC = np.median(vma1_1024_VNPC[:,11])
print('point_width_22_vma1_1024_VNPC: {:.3f}'.format(point_width_22_vma1_1024_VNPC))

point_coverage_vma1_1024_VNPC = np.mean(vma1_1024_VNPC[:,12])
print('point_coverage_vma1_1024_VNPC: {:.3f}'.format(point_coverage_vma1_1024_VNPC))

#-------------------------------------------------------------------------------------------------------------

vnpc_L1_var2_256 = var2_256_VNPC[:,1]
vi_L1_var2_256 = var2_256_VB[:,1]
vnpc_L1_var2_512 = var2_512_VNPC[:,1]
vi_L1_var2_512 = var2_512_VB[:,1]
vnpc_L1_var2_1024 = var2_1024_VNPC[:,1]
vi_L1_var2_1024 = var2_1024_VB[:,1]

fig, ax = plt.subplots(1,1, figsize = (20, 6))

box_data = [vnpc_L1_var2_256, vi_L1_var2_256, vnpc_L1_var2_512, vi_L1_var2_512, vnpc_L1_var2_1024, vi_L1_var2_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]  
box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True)

colors = ['lightblue', 'lightgreen'] * 3
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    
ax.legend([box_plot['boxes'][0], box_plot['boxes'][1]], ['VNPC', 'VI'], loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L1 Errors for VNPC and VB Methods for VAR(2) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L1 Errors')

plt.show()

#-----------------------------------------------------------------------------------------------------------
vnpc_L2_var2_256 = var2_256_VNPC[:,2]
vi_L2_var2_256 = var2_256_VB[:,2]
vnpc_L2_var2_512 = var2_512_VNPC[:,2]
vi_L2_var2_512 = var2_512_VB[:,2]
vnpc_L2_var2_1024 = var2_1024_VNPC[:,2]
vi_L2_var2_1024 = var2_1024_VB[:,2]

fig, ax = plt.subplots(1,1, figsize = (20, 6))

box_data = [vnpc_L2_var2_256, vi_L2_var2_256, vnpc_L2_var2_512, vi_L2_var2_512, vnpc_L2_var2_1024, vi_L2_var2_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]  
box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True)

colors = ['lightblue', 'lightgreen'] * 3
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    
ax.legend([box_plot['boxes'][0], box_plot['boxes'][1]], ['VNPC', 'VI'], loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L2 Errors for VNPC and VI Methods for VAR(2) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L2 Errors')

plt.show()

#-----------------------------------------------------------------------------------------------------------
vnpc_L1_vma1_256 = vma1_256_VNPC[:,1]
vi_L1_vma1_256 = vma1_256_VB[:,1]
vnpc_L1_vma1_512 = vma1_512_VNPC[:,1]
vi_L1_vma1_512 = vma1_512_VB[:,1]
vnpc_L1_vma1_1024 = vma1_1024_VNPC[:,1]
vi_L1_vma1_1024 = vma1_1024_VB[:,1]

fig, ax = plt.subplots(1,1, figsize = (20, 6))

box_data = [vnpc_L1_vma1_256, vi_L1_vma1_256, vnpc_L1_vma1_512, vi_L1_vma1_512, vnpc_L1_vma1_1024, vi_L1_vma1_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]  
box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True)

colors = ['lightblue', 'lightgreen'] * 3
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    
ax.legend([box_plot['boxes'][0], box_plot['boxes'][1]], ['VNPC', 'VI'], loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L1 Errors for VNPC and VI Methods for VMA(1) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L1 Errors')

plt.show()

#------------------------------------------------------------------------------------------------------------
vnpc_L2_vma1_256 = vma1_256_VNPC[:,2]
vi_L2_vma1_256 = vma1_256_VB[:,2]
vnpc_L2_vma1_512 = vma1_512_VNPC[:,2]
vi_L2_vma1_512 = vma1_512_VB[:,2]
vnpc_L2_vma1_1024 = vma1_1024_VNPC[:,2]
vi_L2_vma1_1024 = vma1_1024_VB[:,2]

fig, ax = plt.subplots(1,1, figsize = (20, 6))

box_data = [vnpc_L2_vma1_256, vi_L2_vma1_256, vnpc_L2_vma1_512, vi_L2_vma1_512, vnpc_L2_vma1_1024, vi_L2_vma1_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]  
box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True)

colors = ['lightblue', 'lightgreen'] * 3
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    
ax.legend([box_plot['boxes'][0], box_plot['boxes'][1]], ['VNPC', 'VI'], loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L2 Errors for VNPC and VI Methods for VMA(1) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L2 Errors')

plt.show()

#------------------------------------------------------------------------------------------------------
#time interval for VAR2
var2_256_VB = pd.read_csv('VAR2_256_errors_coverages_len_VB_combined_results.csv').values
time_var2_256_VB = np.mean(var2_256_VB[:,15])
print('time_var2_256_VB:', time_var2_256_VB)
var2_512_VB = pd.read_csv('VAR2_512_errors_coverages_len_VB_combined_results.csv').values
time_var2_512_VB = np.mean(var2_512_VB[:,15])
print('time_var2_512_VB:', time_var2_512_VB)
var2_1024_VB = pd.read_csv('VAR2_1024_errors_coverages_len_VB_combined_results.csv').values
time_var2_1024_VB = np.mean(var2_1024_VB[:,15])
print('time_var2_1024_VB:', time_var2_1024_VB)

var2_256_VNPC = pd.read_csv('VAR2_256_errors_coverages_len_VNPC_combined_results.csv').values
time_var2_256_VNPC = np.mean(var2_256_VNPC[:,13])
print('time_var2_256_VNPC:', time_var2_256_VNPC)
var2_512_VNPC = pd.read_csv('VAR2_512_errors_coverages_len_VNPC_combined_results.csv').values
time_var2_512_VNPC = np.mean(var2_512_VNPC[:,13])
print('time_var2_512_VNPC:', time_var2_512_VNPC)
var2_1024_VNPC = pd.read_csv('VAR2_1024_errors_coverages_len_VNPC_combined_results.csv').values
time_var2_1024_VNPC = np.mean(var2_1024_VNPC[:,13])
print('time_var2_1024_VNPC:', time_var2_1024_VNPC)

#------------------------------------------------------------------------------------------------------
#time interval for VMA1
vma1_256_VB = pd.read_csv('VMA1_256_errors_coverages_len_VB_combined_results.csv').values
time_vma1_256_VB = np.mean(vma1_256_VB[:,15])
print('time_vma1_256_VB:', time_vma1_256_VB)
vma1_512_VB = pd.read_csv('VMA1_512_errors_coverages_len_VB_combined_results.csv').values
time_vma1_512_VB = np.mean(vma1_512_VB[:,15])
print('time_vma1_512_VB:', time_vma1_512_VB)
vma1_1024_VB = pd.read_csv('VMA1_1024_errors_coverages_len_VB_combined_results.csv').values
time_vma1_1024_VB = np.mean(vma1_1024_VB[:,15])
print('time_vma1_1024_VB:', time_vma1_1024_VB)

vma1_256_VNPC = pd.read_csv('VMA1_256_errors_coverages_len_VNPC_combined_results.csv').values
time_vma1_256_VNPC = np.mean(vma1_256_VNPC[:,13])
print('time_vma1_256_VNPC:', time_vma1_256_VNPC)
vma1_512_VNPC = pd.read_csv('VMA1_512_errors_coverages_len_VNPC_combined_results.csv').values
time_vma1_512_VNPC = np.mean(vma1_512_VNPC[:,13])
print('time_vma1_512_VNPC:', time_vma1_512_VNPC)
vma1_1024_VNPC = pd.read_csv('VMA1_1024_errors_coverages_len_VNPC_combined_results.csv').values
time_vma1_1024_VNPC = np.mean(vma1_1024_VNPC[:,13])
print('time_vma1_1024_VNPC:', time_vma1_1024_VNPC)
'''
#-------------------------------------------------------------------------------------------------------
'''
fig, ax = plt.subplots(1,1, figsize = (20, 6))

box_data = [var2_256_VNPC[:,13], var2_256_VB[:,15], var2_512_VNPC[:,13], 
            var2_512_VB[:,15], var2_1024_VNPC[:,13], var2_1024_VB[:,15]]
positions = [1, 1.5, 3, 3.5, 5, 5.5]  
box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True)

colors = ['lightblue', 'lightgreen'] * 3
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    
ax.legend([box_plot['boxes'][0], box_plot['boxes'][1]], ['VNPC', 'VI'], loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L2 Errors for VNPC and VI Methods for VAR(2) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L2 Errors')

plt.show()


#--------------------------------------------------------------------------------------------------------
#violin plots

vnpc_L1_var2_256 = var2_256_VNPC[:,1]
vi_L1_var2_256 = var2_256_VB[:,1]
vnpc_L1_var2_512 = var2_512_VNPC[:,1]
vi_L1_var2_512 = var2_512_VB[:,1]
vnpc_L1_var2_1024 = var2_1024_VNPC[:,1]
vi_L1_var2_1024 = var2_1024_VB[:,1]

fig, ax = plt.subplots(1, 1, figsize=(20, 6))

violin_data = [vnpc_L1_var2_256, vi_L1_var2_256, vnpc_L1_var2_512, vi_L1_var2_512, vnpc_L1_var2_1024, vi_L1_var2_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]

parts = ax.violinplot(violin_data, positions=positions, showmeans=False, showmedians=True)

colors = ['lightblue', 'lightgreen'] * 3
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

from matplotlib.patches import Patch
legend_patches = [Patch(facecolor='lightblue', edgecolor='black', label='VNPC'),
                  Patch(facecolor='lightgreen', edgecolor='black', label='VB')]
ax.legend(handles=legend_patches, loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L1 Errors for VNPC and VB Methods for VAR(2) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L1 Errors')

plt.show()
'''
#-----------------------------------------------------------------------------------------------------------
from matplotlib.patches import Patch
vnpc_L2_var2_256 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_var2.csv').values[:,1]
vi_L2_var2_256 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_var2.csv').values[:,0]
vnpc_L2_var2_512 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_var2.csv').values[:,3]
vi_L2_var2_512 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_var2.csv').values[:,2]
vnpc_L2_var2_1024 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_var2.csv').values[:,5]
vi_L2_var2_1024 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_var2.csv').values[:,4]

fig, ax = plt.subplots(1, 1, figsize=(20, 6))

violin_data = [vnpc_L2_var2_256, vi_L2_var2_256, vnpc_L2_var2_512, vi_L2_var2_512, vnpc_L2_var2_1024, vi_L2_var2_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]

parts = ax.violinplot(violin_data, positions=positions, showmeans=False, showmedians=True)

colors = ['lightblue', 'lightgreen'] * 3
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

legend_patches = [Patch(facecolor='lightblue', edgecolor='black', label='VNPC'),
                  Patch(facecolor='lightgreen', edgecolor='black', label='VB')]
ax.legend(handles=legend_patches, loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L2 Errors for VNPC and SGVB Methods for VAR(2) model', fontsize=15)
ax.set_xlabel('Data Length', fontsize=15)
ax.set_ylabel('L2 Errors', fontsize=15)

plt.show()

#-----------------------------------------------------------------------------------------------------------
'''
vnpc_L1_vma1_256 = vma1_256_VNPC[:,1]
vi_L1_vma1_256 = vma1_256_VB[:,1]
vnpc_L1_vma1_512 = vma1_512_VNPC[:,1]
vi_L1_vma1_512 = vma1_512_VB[:,1]
vnpc_L1_vma1_1024 = vma1_1024_VNPC[:,1]
vi_L1_vma1_1024 = vma1_1024_VB[:,1]

fig, ax = plt.subplots(1, 1, figsize=(20, 6))

violin_data = [vnpc_L1_vma1_256, vi_L1_vma1_256, vnpc_L1_vma1_512, vi_L1_vma1_512, vnpc_L1_vma1_1024, vi_L1_vma1_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]

parts = ax.violinplot(violin_data, positions=positions, showmeans=False, showmedians=True)

colors = ['lightblue', 'lightgreen'] * 3
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

legend_patches = [Patch(facecolor='lightblue', edgecolor='black', label='VNPC'),
                  Patch(facecolor='lightgreen', edgecolor='black', label='VB')]
ax.legend(handles=legend_patches, loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L1 Errors for VNPC and VB Methods for VMA(1) model')
ax.set_xlabel('Data Length')
ax.set_ylabel('L1 Errors')

plt.show()
'''
#-----------------------------------------------------------------------------------------------------------

vnpc_L2_vma1_256 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_vma1.csv').values[:,1]
vi_L2_vma1_256 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_vma1.csv').values[:,0]
vnpc_L2_vma1_512 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_vma1.csv').values[:,3]
vi_L2_vma1_512 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_vma1.csv').values[:,2]
vnpc_L2_vma1_1024 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_vma1.csv').values[:,5]
vi_L2_vma1_1024 = pd.read_csv('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/data set used for the plot in paper/L2_errors_vma1.csv').values[:,4]

fig, ax = plt.subplots(1, 1, figsize=(20, 6))

violin_data = [vnpc_L2_vma1_256, vi_L2_vma1_256, vnpc_L2_vma1_512, vi_L2_vma1_512, vnpc_L2_vma1_1024, vi_L2_vma1_1024]
positions = [1, 1.5, 3, 3.5, 5, 5.5]

parts = ax.violinplot(violin_data, positions=positions, showmeans=False, showmedians=True)

colors = ['lightblue', 'lightgreen'] * 3
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

legend_patches = [Patch(facecolor='lightblue', edgecolor='black', label='VNPC'),
                  Patch(facecolor='lightgreen', edgecolor='black', label='VB')]
ax.legend(handles=legend_patches, loc='upper right')

ax.set_xticks([1.25, 3.25, 5.25])
ax.set_xticklabels(['256', '512', '1024'])

ax.set_title('L2 Errors for VNPC and SGVB Methods for VMA(1) model', fontsize=15)
ax.set_xlabel('Data Length', fontsize=15)
ax.set_ylabel('L2 Errors', fontsize=15)

plt.show()























