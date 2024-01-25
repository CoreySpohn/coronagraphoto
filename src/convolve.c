void convolve_c(double *counts, const double *disk, const double *psfs,
                const int nrows, const int ncols) {
  int psfs_index = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      int disk_ind = 0;
      for (int ii = 0; ii < nrows; ii++) {
        for (int jj = 0; jj < ncols; jj++) {
          counts[disk_ind] += disk[disk_ind] * psfs[psfs_index + disk_ind];
          disk_ind++;
        }
      }
      psfs_index++;
    }
  }
}
