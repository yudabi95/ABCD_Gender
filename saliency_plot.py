import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load('/home/users/ybi3/PyC_Project/tmp/pycharm_project_816/temp/AO_sex_iter__nSub_0.nii').get_fdata()

plt.plot(img)
plt.show()