"""
Observation planning components for coronagraphoto v2.

This module defines the "what" of an observation: what targets to observe,
when to observe them, and how to sequence multiple observations.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import astropy.units as u
from astropy.time import Time
from .data_models import IntermediateData


class Target:
    """
    A wrapper around a scene generator for an astrophysical target.
    
    This class represents a single pointing direction at an astrophysical source.
    It provides a uniform interface for different scene types (ExoVista, synthetic, etc.).
    """
    
    def __init__(self, scene_path: str, name: Optional[str] = None):
        """
        Initialize a Target with a scene file.
        
        Args:
            scene_path: 
                Path to the scene file (e.g., ExoVista FITS file)
            name: 
                Optional name for the target (defaults to scene filename)
        """
        self.scene_path = scene_path
        self.name = name or scene_path.split('/')[-1]
    
    def load_scene(self) -> IntermediateData:
        """
        Load the scene data into an IntermediateData object.
        
        Returns:
            IntermediateData instance with the loaded scene
        """
        # Use the ExoVista integration
        from .light_paths import load_scene_from_exovista
        from .data_models import PropagationContext
        
        # Create a default context for scene loading
        context = PropagationContext(
            time=Time.now(),
            wavelength=550 * u.nm,
            bandpass_slice=1.0 * u.nm,
            time_step=1.0 * u.s,
            rng_key=42
        )
        
        return load_scene_from_exovista(self.scene_path, context)
    
    def __repr__(self) -> str:
        return f"Target(name='{self.name}', scene_path='{self.scene_path}')"


@dataclass(frozen=True)
class Observation:
    """
    Parameters for a single exposure.
    
    This immutable dataclass captures all the information needed to plan
    a single observation: what to observe, when to observe it, and how
    to process it through the observatory.
    """
    target: Target
    start_time: Time
    exposure_time: u.Quantity
    path_name: str
    roll_angle: Optional[u.Quantity] = None
    dither_position: Optional[tuple] = None
    
    def __post_init__(self):
        """Validate the observation parameters."""
        if self.exposure_time.to(u.s).value <= 0:
            raise ValueError("Exposure time must be positive")
        if not isinstance(self.path_name, str):
            raise ValueError("Path name must be a string")


class ObservationSequence:
    """
    A container for a list of observations.
    
    This class manages an ordered sequence of observations and provides
    builder methods for creating common observing patterns like ADI and RDI.
    """
    
    def __init__(self, observations: List[Observation]):
        """
        Initialize with a list of observations.
        
        Args:
            observations: 
                List of Observation objects to execute in order
        """
        self.observations = observations
        self._validate_sequence()
    
    def _validate_sequence(self):
        """Validate the observation sequence."""
        if not self.observations:
            raise ValueError("Observation sequence cannot be empty")
        
        for i, obs in enumerate(self.observations):
            if not isinstance(obs, Observation):
                raise TypeError(f"Observation {i} is not an Observation instance")
    
    def __len__(self) -> int:
        """Return the number of observations in the sequence."""
        return len(self.observations)
    
    def __getitem__(self, index: int) -> Observation:
        """Get an observation by index."""
        return self.observations[index]
    
    def __iter__(self):
        """Iterate over observations."""
        return iter(self.observations)
    
    @property
    def total_exposure_time(self) -> u.Quantity:
        """Calculate the total exposure time for all observations."""
        return sum(obs.exposure_time for obs in self.observations)
    
    @property
    def total_duration(self) -> u.Quantity:
        """Calculate the total duration from start to end."""
        if not self.observations:
            return 0 * u.s
        
        start_time = self.observations[0].start_time
        end_time = self.observations[-1].start_time + self.observations[-1].exposure_time
        return (end_time - start_time).to(u.s)
    
    @classmethod
    def for_adi(
        cls,
        target: Target,
        path_name: str,
        n_exposures: int,
        exposure_time: u.Quantity,
        start_time: Time,
        total_roll_angle: u.Quantity = 360 * u.deg,
        frame_time: Optional[u.Quantity] = None
    ) -> 'ObservationSequence':
        """
        Create an Angular Differential Imaging (ADI) observation sequence.
        
        Args:
            target: 
                The target to observe
            path_name: 
                Name of the light path to use for all observations
            n_exposures: 
                Number of exposures in the sequence
            exposure_time: 
                Duration of each individual exposure
            start_time: 
                Start time of the first observation
            total_roll_angle: 
                Total rotation angle over the sequence
            frame_time: 
                Time between frames (defaults to exposure_time)
                
        Returns:
            ObservationSequence configured for ADI
        """
        if n_exposures <= 0:
            raise ValueError("Number of exposures must be positive")
        
        if frame_time is None:
            frame_time = exposure_time
        
        # Calculate roll angles evenly distributed over the sequence
        roll_angles_values = np.linspace(0, total_roll_angle.to(u.deg).value, n_exposures)
        roll_angles = roll_angles_values * u.deg
        
        observations = []
        for i in range(n_exposures):
            obs_start_time = start_time + i * frame_time
            obs = Observation(
                target=target,
                start_time=obs_start_time,
                exposure_time=exposure_time,
                path_name=path_name,
                roll_angle=roll_angles[i]
            )
            observations.append(obs)
        
        return cls(observations)
    
    @classmethod
    def for_rdi(
        cls,
        science_target: Target,
        ref_target: Target,
        science_path_name: str,
        ref_path_name: str,
        exposure_time: u.Quantity,
        start_time: Time,
        frame_time: Optional[u.Quantity] = None,
        n_ref_per_science: int = 1
    ) -> 'ObservationSequence':
        """
        Create a Reference Differential Imaging (RDI) observation sequence.
        
        Args:
            science_target: 
                The science target to observe
            ref_target: 
                The reference target to observe
            science_path_name: 
                Name of the light path for science observations
            ref_path_name: 
                Name of the light path for reference observations
            exposure_time: 
                Duration of each individual exposure
            start_time: 
                Start time of the first observation
            frame_time: 
                Time between frames (defaults to exposure_time)
            n_ref_per_science: 
                Number of reference exposures per science exposure
                
        Returns:
            ObservationSequence configured for RDI
        """
        if frame_time is None:
            frame_time = exposure_time
        
        observations = []
        current_time = start_time
        
        # Create alternating science and reference observations
        science_obs = Observation(
            target=science_target,
            start_time=current_time,
            exposure_time=exposure_time,
            path_name=science_path_name
        )
        observations.append(science_obs)
        current_time += frame_time
        
        # Add reference observations
        for _ in range(n_ref_per_science):
            ref_obs = Observation(
                target=ref_target,
                start_time=current_time,
                exposure_time=exposure_time,
                path_name=ref_path_name
            )
            observations.append(ref_obs)
            current_time += frame_time
        
        return cls(observations)
    
    @classmethod
    def single_exposure(
        cls,
        target: Target,
        path_name: str,
        exposure_time: u.Quantity,
        start_time: Time,
        roll_angle: Optional[u.Quantity] = None
    ) -> 'ObservationSequence':
        """
        Create a single exposure observation sequence.
        
        Args:
            target: 
                The target to observe
            path_name: 
                Name of the light path to use
            exposure_time: 
                Duration of the exposure
            start_time: 
                Start time of the observation
            roll_angle: 
                Optional telescope roll angle
                
        Returns:
            ObservationSequence with a single observation
        """
        observation = Observation(
            target=target,
            start_time=start_time,
            exposure_time=exposure_time,
            path_name=path_name,
            roll_angle=roll_angle
        )
        return cls([observation])