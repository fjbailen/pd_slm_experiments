# pd_slm_experiments
Programs to carry out a PD analysis of the prototype of IMaX improved with an SLM

# Way to use it

Run the programs in the next order:

1. correct_dark_and_flats.py: to correct a single FITS file from dark and flat
2. alignment.py: to align a pair of focused-defocused images corrected from dark and flat
3. parallel_subpatches_central_subfield: to run PD over a pair of aligned focused-defocused images
4. plot_parallel_central_subpatch: to visualize the PD results and the reconstructed images
