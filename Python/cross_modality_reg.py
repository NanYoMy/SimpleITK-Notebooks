import SimpleITK as sitk
import registration_utilities as ru
import registration_callbacks as rc

import matplotlib.pyplot as plt

from ipywidgets import interact, fixed

#utility method that either downloads data from the Girder repository or
#if already downloaded returns the file name for reading from disk (cached data)
from popi_utilities_setup import *
import glob
basedir = '../../dataset/JTDX/SERN/heart_ct_mr_Myo_96_cmspace_stand/test10'
mv_images = []
mv_masks = []
fixed_images = []
fixed_masks = []

mv_image_pathes = glob.glob(basedir + "/ct*image.nii.gz")
mv_mask_pathes = glob.glob(basedir + "/ct*label.nii.gz")
fix_image_pathes = glob.glob(basedir + "/mr*image.nii.gz")
fix_mask_pathes = glob.glob(basedir + "/mr*label.nii.gz")
i = 0
for ct_i, ct_m, mr_i, mr_m in zip(mv_image_pathes, mv_mask_pathes, fix_image_pathes, fix_mask_pathes):
    mv_images.append(sitk.ReadImage(ct_i, sitk.sitkFloat32))  # read and cast to format required for registration
    mv_masks.append(sitk.ReadImage(ct_m))
    fixed_images.append(sitk.ReadImage(mr_i, sitk.sitkFloat32))  # read and cast to format required for registration
    fixed_masks.append(sitk.ReadImage(mr_m))

    i = i + 1
    if i > 4:
        break

def bspline_intra_modal_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None,
                                     moving_points=None):
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [4.0, 4.0, 4.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=20)
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent,
                                       lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points,
                                                                                   moving_points))

    return registration_method.Execute(fixed_image, moving_image)



moving_image_index = 0
fixed_image_index=0
tx = bspline_intra_modal_registration(fixed_image = mv_images[moving_image_index],
                                      moving_image = fixed_images[fixed_image_index],
                                      fixed_image_mask = None,
                                      fixed_points = None,
                                      moving_points = None
                                     )

# Transfer the segmentation via the estimated transformation. Use Nearest Neighbor interpolation to retain the labels.
transformed_labels = sitk.Resample(mv_masks[moving_image_index],
                                   fixed_images[fixed_image_index],
                                   tx,
                                   sitk.sitkNearestNeighbor,
                                   0.0,
                                   mv_masks[moving_image_index].GetPixelID())

segmentations_before_and_after = [mv_masks[moving_image_index], transformed_labels]


# Compute the Dice coefficient and Hausdorff distance between the segmentations before, and after registration.
ground_truth = fixed_masks[fixed_image_index] == myo_label
before_registration = mv_masks[moving_image_index] == myo_label
after_registration = transformed_labels == myo_label

label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
# print(f'{ground_truth.GetSpacing()} : {before_registration.GetSpacing()}')
# label_overlap_measures_filter.Execute(ground_truth, before_registration)
# print(f"Dice coefficient before registration: {label_overlap_measures_filter.GetDiceCoefficient():.2f}")
label_overlap_measures_filter.Execute(ground_truth, after_registration)
print(f"Dice coefficient after registration: {label_overlap_measures_filter.GetDiceCoefficient():.2f}")

hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
# hausdorff_distance_image_filter.Execute(ground_truth, before_registration)
# print(f"Hausdorff distance before registration: {hausdorff_distance_image_filter.GetHausdorffDistance():.2f}")
hausdorff_distance_image_filter.Execute(ground_truth, after_registration)
print(f"Hausdorff distance after registration: {hausdorff_distance_image_filter.GetHausdorffDistance():.2f}")

