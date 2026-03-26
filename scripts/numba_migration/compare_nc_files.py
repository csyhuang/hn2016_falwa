#!/usr/bin/env python
"""
Script to compare two netCDF files and report differences beyond a relative tolerance.

This script is used to validate the F2PY to Numba translation by comparing
output files before and after the translation.

Usage:
    python compare_nc_files.py \
        --reference output_plots_v2.3.3_NH18/lwa_reference_output_v2.3.3_NH18.nc \
        --translated output_plots_ce9e383_NH18/lwa_reference_output_NH18.nc \
        --rtol 1.e-4

"""

import argparse
import numpy as np
import xarray as xr
from typing import Optional


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare two netCDF files and report differences beyond tolerance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to the reference netCDF file (before translation)')
    parser.add_argument('--translated', type=str, required=True,
                        help='Path to the translated netCDF file (after translation)')
    parser.add_argument('--rtol', type=float, default=1.e-4,
                        help='Relative tolerance for comparison (default: 1.e-4)')
    parser.add_argument('--atol', type=float, default=0.0,
                        help='Absolute tolerance for comparison (default: 0.0)')
    parser.add_argument('--max_mismatches', type=int, default=50,
                        help='Maximum number of mismatches to print per variable (default: 50)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Optional file to save detailed mismatch report')
    return parser.parse_args()


def get_coordinate_labels(var_data: xr.DataArray, indices: tuple) -> dict:
    """
    Get coordinate labels for the given indices.
    
    Parameters
    ----------
    var_data : xr.DataArray
        The data array with coordinate information
    indices : tuple
        Tuple of integer indices
        
    Returns
    -------
    dict
        Dictionary mapping dimension names to coordinate values
    """
    coord_labels = {}
    dims = var_data.dims
    for i, dim in enumerate(dims):
        if dim in var_data.coords:
            coord_labels[dim] = float(var_data.coords[dim].values[indices[i]])
        else:
            coord_labels[dim] = indices[i]
    return coord_labels


def compare_variable(ref_data: xr.DataArray, trans_data: xr.DataArray, 
                     var_name: str, rtol: float, atol: float, 
                     max_mismatches: int) -> tuple[bool, list]:
    """
    Compare a single variable between reference and translated datasets.
    
    Parameters
    ----------
    ref_data : xr.DataArray
        Reference data array
    trans_data : xr.DataArray
        Translated data array
    var_name : str
        Name of the variable
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    max_mismatches : int
        Maximum number of mismatches to report
        
    Returns
    -------
    tuple[bool, list]
        (match_status, list of mismatch details)
    """
    ref_values = ref_data.values
    trans_values = trans_data.values
    
    # Check for shape mismatch
    if ref_values.shape != trans_values.shape:
        return False, [f"Shape mismatch: reference {ref_values.shape} vs translated {trans_values.shape}"]
    
    # Handle NaN values - treat NaN == NaN as matching
    ref_nan = np.isnan(ref_values)
    trans_nan = np.isnan(trans_values)
    
    # Check if NaN locations match
    if not np.array_equal(ref_nan, trans_nan):
        nan_mismatch_count = np.sum(ref_nan != trans_nan)
        mismatches = [f"NaN location mismatch: {nan_mismatch_count} positions differ in NaN status"]
        return False, mismatches
    
    # For non-NaN values, compute relative difference
    # Mask where both are not NaN
    valid_mask = ~ref_nan & ~trans_nan
    
    if not np.any(valid_mask):
        # All values are NaN, consider it a match
        return True, []
    
    ref_valid = ref_values[valid_mask]
    trans_valid = trans_values[valid_mask]
    
    # Compute relative difference: |ref - trans| / max(|ref|, |trans|, atol)
    abs_diff = np.abs(ref_valid - trans_valid)
    denominator = np.maximum(np.abs(ref_valid), np.abs(trans_valid))
    denominator = np.maximum(denominator, atol if atol > 0 else 1e-15)  # Avoid division by zero
    
    rel_diff = abs_diff / denominator
    
    # Find mismatches
    mismatch_mask_valid = rel_diff > rtol
    
    if not np.any(mismatch_mask_valid):
        return True, []
    
    # Get full mismatch mask
    mismatch_mask = np.zeros_like(ref_values, dtype=bool)
    mismatch_mask[valid_mask] = mismatch_mask_valid
    
    # Get indices of mismatches
    mismatch_indices = np.argwhere(mismatch_mask)
    
    total_mismatches = len(mismatch_indices)
    mismatches = []
    
    # Report mismatch summary
    mismatches.append(f"Total mismatches: {total_mismatches} out of {ref_values.size} values "
                      f"({100*total_mismatches/ref_values.size:.4f}%)")
    
    # Compute max relative difference
    max_rel_diff = np.max(rel_diff[mismatch_mask_valid])
    mismatches.append(f"Maximum relative difference: {max_rel_diff:.6e}")
    
    # Report detailed mismatches (up to max_mismatches)
    mismatches.append(f"\nDetailed mismatches (showing first {min(max_mismatches, total_mismatches)}):")
    mismatches.append("-" * 100)
    
    for i, idx in enumerate(mismatch_indices[:max_mismatches]):
        idx_tuple = tuple(idx)
        ref_val = ref_values[idx_tuple]
        trans_val = trans_values[idx_tuple]
        diff = abs(ref_val - trans_val)
        rel_d = diff / max(abs(ref_val), abs(trans_val), 1e-15)
        
        coord_labels = get_coordinate_labels(ref_data, idx_tuple)
        coord_str = ", ".join([f"{k}={v}" for k, v in coord_labels.items()])
        
        mismatches.append(
            f"  [{i+1}] ({coord_str}): "
            f"ref={ref_val:.10e}, trans={trans_val:.10e}, "
            f"abs_diff={diff:.6e}, rel_diff={rel_d:.6e}"
        )
    
    if total_mismatches > max_mismatches:
        mismatches.append(f"  ... and {total_mismatches - max_mismatches} more mismatches")
    
    return False, mismatches


def compare_nc_files(reference_file: str, translated_file: str, 
                     rtol: float = 1.e-4, atol: float = 0.0,
                     max_mismatches: int = 50,
                     output_file: Optional[str] = None) -> bool:
    """
    Compare two netCDF files and report differences.
    
    Parameters
    ----------
    reference_file : str
        Path to reference netCDF file
    translated_file : str
        Path to translated netCDF file
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    max_mismatches : int
        Maximum mismatches to print per variable
    output_file : str, optional
        File to save detailed report
        
    Returns
    -------
    bool
        True if all variables match within tolerance, False otherwise
    """
    print("=" * 100)
    print("NetCDF File Comparison Report")
    print("=" * 100)
    print(f"\nReference file:  {reference_file}")
    print(f"Translated file: {translated_file}")
    print(f"Relative tolerance: {rtol}")
    print(f"Absolute tolerance: {atol}")
    print("=" * 100)
    
    # Open datasets
    ds_ref = xr.open_dataset(reference_file)
    ds_trans = xr.open_dataset(translated_file)
    
    # Print version info if available
    print("\nDataset Attributes:")
    print(f"  Reference FALWA version:  {ds_ref.attrs.get('falwa_version', 'N/A')}")
    print(f"  Translated FALWA version: {ds_trans.attrs.get('falwa_version', 'N/A')}")
    print(f"  QGField type: {ds_ref.attrs.get('qgfield_type', 'N/A')}")
    
    # Check variable lists
    ref_vars = set(ds_ref.data_vars)
    trans_vars = set(ds_trans.data_vars)
    
    only_in_ref = ref_vars - trans_vars
    only_in_trans = trans_vars - ref_vars
    common_vars = ref_vars & trans_vars
    
    print(f"\nVariables in reference:  {len(ref_vars)}")
    print(f"Variables in translated: {len(trans_vars)}")
    print(f"Common variables:        {len(common_vars)}")
    
    if only_in_ref:
        print(f"\n⚠️  Variables only in reference: {sorted(only_in_ref)}")
    if only_in_trans:
        print(f"\n⚠️  Variables only in translated: {sorted(only_in_trans)}")
    
    print("\n" + "=" * 100)
    print("Variable Comparison Results")
    print("=" * 100)
    
    all_match = True
    report_lines = []
    
    # Compare each common variable
    for var_name in sorted(common_vars):
        ref_data = ds_ref[var_name]
        trans_data = ds_trans[var_name]
        
        match, mismatches = compare_variable(
            ref_data, trans_data, var_name, rtol, atol, max_mismatches
        )
        
        if match:
            print(f"\n✅ {var_name}: MATCH (dims: {ref_data.dims})")
        else:
            print(f"\n❌ {var_name}: MISMATCH (dims: {ref_data.dims})")
            for msg in mismatches:
                print(f"   {msg}")
            all_match = False
            
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"Variable: {var_name}")
            report_lines.append(f"Dimensions: {ref_data.dims}")
            report_lines.extend(mismatches)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if all_match and not only_in_ref and not only_in_trans:
        print("\n✅ ALL VARIABLES MATCH within tolerance!")
    else:
        if not all_match:
            print("\n❌ SOME VARIABLES DO NOT MATCH within tolerance.")
        if only_in_ref or only_in_trans:
            print("⚠️  Variable lists differ between files.")
    
    # Save report if requested
    if output_file and report_lines:
        with open(output_file, 'w') as f:
            f.write("NetCDF File Comparison Report\n")
            f.write(f"Reference: {reference_file}\n")
            f.write(f"Translated: {translated_file}\n")
            f.write(f"Relative tolerance: {rtol}\n")
            f.write(f"Absolute tolerance: {atol}\n")
            f.write("\n".join(report_lines))
        print(f"\nDetailed report saved to: {output_file}")
    
    ds_ref.close()
    ds_trans.close()
    
    return all_match


def main():
    """Main entry point."""
    args = parse_arguments()
    
    success = compare_nc_files(
        reference_file=args.reference,
        translated_file=args.translated,
        rtol=args.rtol,
        atol=args.atol,
        max_mismatches=args.max_mismatches,
        output_file=args.output_file
    )
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == '__main__':
    main()

