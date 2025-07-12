"""
Reduction and post-processing framework for coronagraphoto v2.

This module defines the post-processing pipeline that transforms raw observatory
data products into science-ready images. It uses the Strategy pattern to allow
different reduction algorithms to be easily composed.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import numpy as np
import xarray as xr
import astropy.units as u


class ReductionStep(ABC):
    """
    Abstract base class for reduction steps.
    
    Each reduction step implements a specific post-processing algorithm
    that can be composed into a reduction pipeline.
    """
    
    @abstractmethod
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Process the input data and return the reduced data.
        
        Args:
            data: 
                Input xarray Dataset to process
                
        Returns:
            Processed xarray Dataset
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this reduction step."""
        pass


class ReductionPipeline:
    """
    A pipeline of reduction steps for processing observatory data.
    
    This class manages a sequence of ReductionStep objects and applies
    them in order to transform raw data into science-ready products.
    """
    
    def __init__(self, steps: List[ReductionStep]):
        """
        Initialize the reduction pipeline.
        
        Args:
            steps: 
                List of ReductionStep objects to apply in order
        """
        self.steps = steps
        self._validate_steps()
    
    def _validate_steps(self):
        """Validate that all steps are properly configured."""
        if not self.steps:
            raise ValueError("Reduction pipeline must have at least one step")
        
        for i, step in enumerate(self.steps):
            if not isinstance(step, ReductionStep):
                raise TypeError(f"Step {i} is not a ReductionStep instance")
    
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Process data through the entire reduction pipeline.
        
        Args:
            data: 
                Input xarray Dataset from the observatory
                
        Returns:
            Final processed xarray Dataset
        """
        current_data = data
        
        for step in self.steps:
            print(f"Applying reduction step: {step.name}")
            current_data = step.process(current_data)
            
            # Add processing history to metadata
            if 'processing_history' not in current_data.attrs:
                current_data.attrs['processing_history'] = []
            current_data.attrs['processing_history'].append(step.name)
        
        return current_data
    
    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)
    
    def __getitem__(self, index: int) -> ReductionStep:
        """Get a reduction step by index."""
        return self.steps[index]


class ReferenceSubtract(ReductionStep):
    """
    Reference subtraction reduction step.
    
    This step performs reference differential imaging (RDI) by subtracting
    reference frames from science frames.
    """
    
    def __init__(self, method: str = "normalized_subtraction", scale_factor: float = 1.0):
        """
        Initialize the reference subtraction step.
        
        Args:
            method: 
                Method for reference subtraction ('normalized_subtraction', 'optimal_subtraction')
            scale_factor: 
                Scaling factor for reference subtraction
        """
        self.method = method
        self.scale_factor = scale_factor
    
    @property
    def name(self) -> str:
        """Return the name of this reduction step."""
        return f"ReferenceSubtract({self.method})"
    
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply reference subtraction to the data.
        
        Args:
            data: 
                Input dataset with science and reference observations
                
        Returns:
            Dataset with reference-subtracted data
        """
        # This is a placeholder implementation
        # In practice, this would identify science vs reference observations
        # and perform the appropriate subtraction
        
        # For now, just return the input data with a marker
        processed_data = data.copy()
        processed_data.attrs['reference_subtraction_applied'] = True
        processed_data.attrs['reference_method'] = self.method
        processed_data.attrs['reference_scale_factor'] = self.scale_factor
        
        return processed_data


class Derotate(ReductionStep):
    """
    Derotation reduction step.
    
    This step corrects for field rotation by rotating images to align
    with a common reference frame.
    """
    
    def __init__(self, reference_angle: Optional[u.Quantity] = None):
        """
        Initialize the derotation step.
        
        Args:
            reference_angle: 
                Reference angle for derotation (defaults to 0 degrees)
        """
        self.reference_angle = reference_angle or 0 * u.deg
    
    @property
    def name(self) -> str:
        """Return the name of this reduction step."""
        return f"Derotate(ref_angle={self.reference_angle})"
    
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply derotation to the data.
        
        Args:
            data: 
                Input dataset with observations at different roll angles
                
        Returns:
            Dataset with derotated images
        """
        # This is a placeholder implementation
        # In practice, this would rotate each image based on its roll angle
        # to align with the reference angle
        
        processed_data = data.copy()
        processed_data.attrs['derotation_applied'] = True
        processed_data.attrs['reference_angle'] = self.reference_angle.to(u.deg).value
        
        return processed_data


class StackFrames(ReductionStep):
    """
    Frame stacking reduction step.
    
    This step combines multiple frames into a single image using
    various stacking methods (mean, median, etc.).
    """
    
    def __init__(self, method: str = "mean", sigma_clip: Optional[float] = None):
        """
        Initialize the frame stacking step.
        
        Args:
            method: 
                Stacking method ('mean', 'median', 'sum')
            sigma_clip: 
                Optional sigma clipping threshold for outlier rejection
        """
        self.method = method
        self.sigma_clip = sigma_clip
    
    @property
    def name(self) -> str:
        """Return the name of this reduction step."""
        return f"StackFrames({self.method})"
    
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply frame stacking to the data.
        
        Args:
            data: 
                Input dataset with multiple frames
                
        Returns:
            Dataset with stacked frames
        """
        # This is a placeholder implementation
        # In practice, this would stack frames along the observation dimension
        
        processed_data = data.copy()
        processed_data.attrs['stacking_applied'] = True
        processed_data.attrs['stacking_method'] = self.method
        processed_data.attrs['sigma_clip'] = self.sigma_clip
        
        return processed_data


class BackgroundSubtract(ReductionStep):
    """
    Background subtraction reduction step.
    
    This step removes background signal from the images using
    various background estimation methods.
    """
    
    def __init__(self, method: str = "median", annulus_radii: Optional[tuple] = None):
        """
        Initialize the background subtraction step.
        
        Args:
            method: 
                Background estimation method ('median', 'mean', 'annulus')
            annulus_radii: 
                Inner and outer radii for annulus background estimation
        """
        self.method = method
        self.annulus_radii = annulus_radii
    
    @property
    def name(self) -> str:
        """Return the name of this reduction step."""
        return f"BackgroundSubtract({self.method})"
    
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply background subtraction to the data.
        
        Args:
            data: 
                Input dataset with images
                
        Returns:
            Dataset with background-subtracted images
        """
        # This is a placeholder implementation
        # In practice, this would estimate and subtract background
        
        processed_data = data.copy()
        processed_data.attrs['background_subtraction_applied'] = True
        processed_data.attrs['background_method'] = self.method
        processed_data.attrs['annulus_radii'] = self.annulus_radii
        
        return processed_data


class ContrastCurve(ReductionStep):
    """
    Contrast curve calculation reduction step.
    
    This step calculates detection limits as a function of angular separation
    from the central star.
    """
    
    def __init__(self, separations: np.ndarray, sigma_level: float = 5.0):
        """
        Initialize the contrast curve calculation step.
        
        Args:
            separations: 
                Array of angular separations to calculate contrast for
            sigma_level: 
                Detection threshold in units of sigma
        """
        self.separations = separations
        self.sigma_level = sigma_level
    
    @property
    def name(self) -> str:
        """Return the name of this reduction step."""
        return f"ContrastCurve(sigma={self.sigma_level})"
    
    def process(self, data: xr.Dataset) -> xr.Dataset:
        """
        Calculate contrast curve from the data.
        
        Args:
            data: 
                Input dataset with processed images
                
        Returns:
            Dataset with contrast curve data added
        """
        # This is a placeholder implementation
        # In practice, this would calculate noise statistics at each separation
        
        processed_data = data.copy()
        
        # Add placeholder contrast curve data
        processed_data['contrast_curve'] = xr.DataArray(
            np.ones_like(self.separations) * 1e-9,  # Placeholder values
            dims=['separation'],
            coords={'separation': self.separations},
            attrs={'sigma_level': self.sigma_level}
        )
        
        processed_data.attrs['contrast_curve_calculated'] = True
        
        return processed_data