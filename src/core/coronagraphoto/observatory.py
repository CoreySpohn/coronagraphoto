"""
Observatory execution engine for coronagraphoto v2.

This module defines the Observatory class which executes observation sequences
through configurable light paths. It implements the core functional execution
engine that processes observations through pure functions.
"""

from typing import Dict, List, Callable, Any, Optional, Union
import numpy as np
import astropy.units as u
from astropy.time import Time
import xarray as xr

from .data_models import IntermediateData, PropagationContext
from .observation import Observation, ObservationSequence


# Type definitions for light path components
LightPathFunction = Callable[..., IntermediateData]  # More flexible function signature
LightPath = List[LightPathFunction]
# Also accept callable objects like PathPipeline
LightPathCallable = Callable[[IntermediateData, PropagationContext], IntermediateData]


class Observatory:
    """
    The execution engine for coronagraph observations.
    
    This class is a pure executor that takes pre-defined light paths and
    executes observation sequences through them. It implements the functional
    approach where the simulation is a sequence of pure functions.
    """
    
    def __init__(self, light_paths: Dict[str, Union[LightPath, LightPathCallable]]):
        """
        Initialize the Observatory with available light paths.
        
        Args:
            light_paths: 
                Dictionary mapping path names to light path functions or callable objects
        """
        self.light_paths = light_paths
        self._validate_light_paths()
    
    def _validate_light_paths(self):
        """Validate that all light paths are properly configured."""
        if not self.light_paths:
            raise ValueError("Observatory must have at least one light path")
        
        for path_name, path_functions in self.light_paths.items():
            if not isinstance(path_name, str):
                raise TypeError(f"Light path name must be string, got {type(path_name)}")
            
            # Handle both list and callable types
            if isinstance(path_functions, list):
                if not path_functions:
                    raise ValueError(f"Light path '{path_name}' cannot be empty")
            elif callable(path_functions):
                # Callable objects like PathPipeline are valid
                pass
            else:
                raise TypeError(f"Light path '{path_name}' must be a list of functions or callable object")
    
    def run(self, observation_sequence: ObservationSequence, seed: Optional[int] = None) -> xr.Dataset:
        """
        Execute an observation sequence through the observatory.
        
        This method implements the two-phase execution process:
        1. Grid Planning: Determine the computational grid for each observation
        2. Grid Execution: Execute the light path for each grid point
        
        Args:
            observation_sequence: 
                The sequence of observations to execute
            seed: 
                Optional random seed for reproducible results
                
        Returns:
            xarray Dataset containing the simulated data product
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Process each observation in the sequence
        data_products = []
        for i, observation in enumerate(observation_sequence):
            print(f"Processing observation {i+1}/{len(observation_sequence)}: {observation.target.name}")
            
            # Execute the observation
            obs_data = self._execute_observation(observation)
            
            # Add observation metadata
            obs_data.attrs.update({
                'observation_index': i,
                'target_name': observation.target.name,
                'start_time': observation.start_time.iso,
                'exposure_time': observation.exposure_time.to(u.s).value,
                'path_name': observation.path_name,
                'roll_angle': observation.roll_angle.to(u.deg).value if observation.roll_angle is not None else None,
            })
            
            data_products.append(obs_data)
        
        # Combine all observations into a single dataset
        return self._combine_observations(data_products)
    
    def _execute_observation(self, observation: Observation) -> xr.Dataset:
        """
        Execute a single observation through its specified light path.
        
        Args:
            observation: 
                The observation to execute
                
        Returns:
            xarray Dataset for this observation
        """
        # Get the light path for this observation
        if observation.path_name not in self.light_paths:
            raise ValueError(f"Light path '{observation.path_name}' not found in observatory")
        
        light_path = self.light_paths[observation.path_name]
        
        # Phase 1: Grid Planning
        # For now, use a simple grid - in the future this would be determined
        # by interrogating the parameters of the light path functions
        time_grid = self._plan_time_grid(observation)
        wavelength_grid = self._plan_wavelength_grid(observation)
        
        # Phase 2: Grid Execution
        # Execute the light path for each (time, wavelength) grid point
        results = []
        for time_point in time_grid:
            for wavelength_point in wavelength_grid:
                # Create propagation context for this grid point
                context = PropagationContext(
                    time=time_point,
                    wavelength=wavelength_point,
                    bandpass_slice=0.1 * u.nm,  # Placeholder
                    time_step=observation.exposure_time,
                    rng_key=np.random.randint(0, 2**32)
                )
                
                # Load the scene data
                initial_data = observation.target.load_scene()
                
                # Execute the light path
                result_data = self._execute_light_path(initial_data, light_path, context)
                results.append(result_data)
        
        # Combine results into a single dataset for this observation
        return self._combine_grid_results(results, time_grid, wavelength_grid)
    
    def _execute_light_path(self, data: IntermediateData, light_path: Union[LightPath, LightPathCallable], context: PropagationContext) -> IntermediateData:
        """
        Execute a single light path for a given propagation context.
        
        Args:
            data: 
                Initial data to propagate
            light_path: 
                List of functions or callable object to execute
            context: 
                Propagation context for this execution
                
        Returns:
            Final data after propagation through the light path
        """
        if callable(light_path) and not isinstance(light_path, list):
            # Handle callable objects like PathPipeline
            return light_path(data, context)
        else:
            # Handle list of functions
            current_data = data
            
            for step_func in light_path:
                # Each function in the light path should be a partial function
                # with parameters already bound, so we just need to pass data and context
                current_data = step_func(current_data, context)
            
            return current_data
    
    def _plan_time_grid(self, observation: Observation) -> List[Time]:
        """
        Plan the time grid for an observation.
        
        Args:
            observation: 
                The observation to plan for
                
        Returns:
            List of time points for the grid
        """
        # For now, just use a single time point at the start of the observation
        # In the future, this would be determined by detector parameters
        return [observation.start_time]
    
    def _plan_wavelength_grid(self, observation: Observation) -> List[u.Quantity]:
        """
        Plan the wavelength grid for an observation.
        
        Args:
            observation: 
                The observation to plan for
                
        Returns:
            List of wavelength points for the grid
        """
        # For now, use a simple wavelength grid
        # In the future, this would be determined by spectrograph parameters
        return [500 * u.nm, 600 * u.nm, 700 * u.nm]
    
    def _combine_grid_results(self, results: List[IntermediateData], 
                             time_grid: List[Time], 
                             wavelength_grid: List[u.Quantity]) -> xr.Dataset:
        """
        Combine results from different grid points into a single dataset.
        
        Args:
            results: 
                List of IntermediateData results
            time_grid: 
                Time grid points
            wavelength_grid: 
                Wavelength grid points
                
        Returns:
            Combined xarray Dataset
        """
        # This is a placeholder implementation
        # In practice, this would properly combine the multi-dimensional results
        if results:
            return results[0].dataset
        else:
            return xr.Dataset()
    
    def _combine_observations(self, data_products: List[xr.Dataset]) -> xr.Dataset:
        """
        Combine multiple observation datasets into a final data product.
        
        Args:
            data_products: 
                List of datasets from individual observations
                
        Returns:
            Combined dataset for the entire observation sequence
        """
        if not data_products:
            return xr.Dataset()
        
        # For now, concatenate along a new 'observation' dimension
        # In practice, this would handle proper merging of similar observations
        try:
            combined = xr.concat(data_products, dim='observation')
            
            # Add global metadata
            combined.attrs.update({
                'num_observations': len(data_products),
                'total_exposure_time': sum(ds.attrs.get('exposure_time', 0) for ds in data_products),
                'coronagraphoto_version': '2.0.0-dev'
            })
            
            return combined
        except Exception:
            # If concatenation fails, return the first dataset as fallback
            return data_products[0]