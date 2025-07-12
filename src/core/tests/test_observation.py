"""
Tests for the observation module components.

This module tests the foundational observation planning components:
Target, Observation, and ObservationSequence.
"""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time

from coronagraphoto.observation import Target, Observation, ObservationSequence


class TestTarget:
    """Test cases for the Target class."""
    
    def test_target_creation(self):
        """Test basic target creation."""
        target = Target("test_scene.fits")
        assert target.scene_path == "test_scene.fits"
        assert target.name == "test_scene.fits"
    
    def test_target_with_name(self):
        """Test target creation with custom name."""
        target = Target("path/to/scene.fits", name="My Target")
        assert target.scene_path == "path/to/scene.fits"
        assert target.name == "My Target"
    
    def test_target_name_from_path(self):
        """Test that target name defaults to filename."""
        target = Target("path/to/my_scene.fits")
        assert target.name == "my_scene.fits"
    
    def test_target_repr(self):
        """Test target string representation."""
        target = Target("test.fits", name="Test")
        repr_str = repr(target)
        assert "Target" in repr_str
        assert "Test" in repr_str
        assert "test.fits" in repr_str


class TestObservation:
    """Test cases for the Observation dataclass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.target = Target("test_scene.fits")
        self.start_time = Time("2024-01-01T00:00:00")
        self.exposure_time = 300 * u.s
        self.path_name = "test_path"
    
    def test_observation_creation(self):
        """Test basic observation creation."""
        obs = Observation(
            target=self.target,
            start_time=self.start_time,
            exposure_time=self.exposure_time,
            path_name=self.path_name
        )
        assert obs.target == self.target
        assert obs.start_time == self.start_time
        assert obs.exposure_time == self.exposure_time
        assert obs.path_name == self.path_name
        assert obs.roll_angle is None
        assert obs.dither_position is None
    
    def test_observation_with_roll_angle(self):
        """Test observation with roll angle."""
        roll_angle = 45 * u.deg
        obs = Observation(
            target=self.target,
            start_time=self.start_time,
            exposure_time=self.exposure_time,
            path_name=self.path_name,
            roll_angle=roll_angle
        )
        assert obs.roll_angle == roll_angle
    
    def test_observation_validation_positive_exposure_time(self):
        """Test that exposure time must be positive."""
        with pytest.raises(ValueError, match="Exposure time must be positive"):
            Observation(
                target=self.target,
                start_time=self.start_time,
                exposure_time=-100 * u.s,
                path_name=self.path_name
            )
    
    def test_observation_validation_zero_exposure_time(self):
        """Test that exposure time cannot be zero."""
        with pytest.raises(ValueError, match="Exposure time must be positive"):
            Observation(
                target=self.target,
                start_time=self.start_time,
                exposure_time=0 * u.s,
                path_name=self.path_name
            )
    
    def test_observation_validation_path_name_string(self):
        """Test that path name must be a string."""
        with pytest.raises(ValueError, match="Path name must be a string"):
            Observation(
                target=self.target,
                start_time=self.start_time,
                exposure_time=self.exposure_time,
                path_name=123  # Not a string
            )
    
    def test_observation_immutability(self):
        """Test that observations are immutable."""
        obs = Observation(
            target=self.target,
            start_time=self.start_time,
            exposure_time=self.exposure_time,
            path_name=self.path_name
        )
        
        # Should not be able to modify the observation
        with pytest.raises(AttributeError):
            obs.exposure_time = 600 * u.s


class TestObservationSequence:
    """Test cases for the ObservationSequence class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.target = Target("test_scene.fits")
        self.start_time = Time("2024-01-01T00:00:00")
        self.exposure_time = 300 * u.s
        self.path_name = "test_path"
        
        # Create a sample observation
        self.obs1 = Observation(
            target=self.target,
            start_time=self.start_time,
            exposure_time=self.exposure_time,
            path_name=self.path_name
        )
        
        self.obs2 = Observation(
            target=self.target,
            start_time=self.start_time + 400 * u.s,
            exposure_time=self.exposure_time,
            path_name=self.path_name
        )
    
    def test_sequence_creation(self):
        """Test basic sequence creation."""
        seq = ObservationSequence([self.obs1, self.obs2])
        assert len(seq) == 2
        assert seq[0] == self.obs1
        assert seq[1] == self.obs2
    
    def test_sequence_iteration(self):
        """Test sequence iteration."""
        seq = ObservationSequence([self.obs1, self.obs2])
        observations = list(seq)
        assert len(observations) == 2
        assert observations[0] == self.obs1
        assert observations[1] == self.obs2
    
    def test_sequence_total_exposure_time(self):
        """Test total exposure time calculation."""
        seq = ObservationSequence([self.obs1, self.obs2])
        total_time = seq.total_exposure_time
        expected_time = self.exposure_time + self.exposure_time
        assert total_time == expected_time
    
    def test_sequence_total_duration(self):
        """Test total duration calculation."""
        seq = ObservationSequence([self.obs1, self.obs2])
        total_duration = seq.total_duration
        # Duration should be from start of first to end of last
        expected_duration = (400 + 300) * u.s
        assert abs((total_duration - expected_duration).to(u.s).value) < 1e-10
    
    def test_sequence_empty_validation(self):
        """Test that empty sequences are rejected."""
        with pytest.raises(ValueError, match="Observation sequence cannot be empty"):
            ObservationSequence([])
    
    def test_sequence_type_validation(self):
        """Test that non-Observation objects are rejected."""
        with pytest.raises(TypeError, match="is not an Observation instance"):
            ObservationSequence([self.obs1, "not an observation"])
    
    def test_single_exposure_builder(self):
        """Test the single exposure builder."""
        seq = ObservationSequence.single_exposure(
            target=self.target,
            path_name=self.path_name,
            exposure_time=self.exposure_time,
            start_time=self.start_time
        )
        
        assert len(seq) == 1
        assert seq[0].target == self.target
        assert seq[0].path_name == self.path_name
        assert seq[0].exposure_time == self.exposure_time
        assert seq[0].start_time == self.start_time
    
    def test_single_exposure_with_roll_angle(self):
        """Test single exposure with roll angle."""
        roll_angle = 30 * u.deg
        seq = ObservationSequence.single_exposure(
            target=self.target,
            path_name=self.path_name,
            exposure_time=self.exposure_time,
            start_time=self.start_time,
            roll_angle=roll_angle
        )
        
        assert seq[0].roll_angle == roll_angle


class TestObservationSequenceADI:
    """Test cases for ADI observation sequence builder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.target = Target("test_scene.fits")
        self.start_time = Time("2024-01-01T00:00:00")
        self.exposure_time = 300 * u.s
        self.path_name = "adi_path"
    
    def test_adi_basic(self):
        """Test basic ADI sequence creation."""
        n_exposures = 10
        seq = ObservationSequence.for_adi(
            target=self.target,
            path_name=self.path_name,
            n_exposures=n_exposures,
            exposure_time=self.exposure_time,
            start_time=self.start_time
        )
        
        assert len(seq) == n_exposures
        assert seq.total_exposure_time == n_exposures * self.exposure_time
    
    def test_adi_roll_angles(self):
        """Test that ADI sequences have correct roll angles."""
        n_exposures = 4
        total_roll = 180 * u.deg
        
        seq = ObservationSequence.for_adi(
            target=self.target,
            path_name=self.path_name,
            n_exposures=n_exposures,
            exposure_time=self.exposure_time,
            start_time=self.start_time,
            total_roll_angle=total_roll
        )
        
        # Check that roll angles are evenly distributed
        expected_angles = [0, 60, 120, 180] * u.deg
        for i, obs in enumerate(seq):
            assert obs.roll_angle.to(u.deg).value == pytest.approx(expected_angles[i].value, abs=1e-10)
    
    def test_adi_timing(self):
        """Test that ADI sequences have correct timing."""
        n_exposures = 3
        frame_time = 400 * u.s
        
        seq = ObservationSequence.for_adi(
            target=self.target,
            path_name=self.path_name,
            n_exposures=n_exposures,
            exposure_time=self.exposure_time,
            start_time=self.start_time,
            frame_time=frame_time
        )
        
        # Check timing
        assert seq[0].start_time == self.start_time
        assert seq[1].start_time == self.start_time + frame_time
        assert seq[2].start_time == self.start_time + 2 * frame_time
    
    def test_adi_default_frame_time(self):
        """Test that frame time defaults to exposure time."""
        n_exposures = 2
        seq = ObservationSequence.for_adi(
            target=self.target,
            path_name=self.path_name,
            n_exposures=n_exposures,
            exposure_time=self.exposure_time,
            start_time=self.start_time
        )
        
        time_diff = seq[1].start_time - seq[0].start_time
        assert abs((time_diff.to(u.s) - self.exposure_time).to(u.s).value) < 1e-10
    
    def test_adi_validation_positive_exposures(self):
        """Test that number of exposures must be positive."""
        with pytest.raises(ValueError, match="Number of exposures must be positive"):
            ObservationSequence.for_adi(
                target=self.target,
                path_name=self.path_name,
                n_exposures=0,
                exposure_time=self.exposure_time,
                start_time=self.start_time
            )


class TestObservationSequenceRDI:
    """Test cases for RDI observation sequence builder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.science_target = Target("science_scene.fits")
        self.ref_target = Target("ref_scene.fits")
        self.start_time = Time("2024-01-01T00:00:00")
        self.exposure_time = 300 * u.s
        self.science_path = "science_path"
        self.ref_path = "ref_path"
    
    def test_rdi_basic(self):
        """Test basic RDI sequence creation."""
        seq = ObservationSequence.for_rdi(
            science_target=self.science_target,
            ref_target=self.ref_target,
            science_path_name=self.science_path,
            ref_path_name=self.ref_path,
            exposure_time=self.exposure_time,
            start_time=self.start_time
        )
        
        # Should have 1 science + 1 reference = 2 total
        assert len(seq) == 2
        assert seq[0].target == self.science_target
        assert seq[0].path_name == self.science_path
        assert seq[1].target == self.ref_target
        assert seq[1].path_name == self.ref_path
    
    def test_rdi_multiple_refs(self):
        """Test RDI with multiple reference exposures."""
        n_ref_per_science = 3
        seq = ObservationSequence.for_rdi(
            science_target=self.science_target,
            ref_target=self.ref_target,
            science_path_name=self.science_path,
            ref_path_name=self.ref_path,
            exposure_time=self.exposure_time,
            start_time=self.start_time,
            n_ref_per_science=n_ref_per_science
        )
        
        # Should have 1 science + 3 reference = 4 total
        assert len(seq) == 4
        assert seq[0].path_name == self.science_path
        assert seq[1].path_name == self.ref_path
        assert seq[2].path_name == self.ref_path
        assert seq[3].path_name == self.ref_path
    
    def test_rdi_timing(self):
        """Test RDI sequence timing."""
        frame_time = 400 * u.s
        seq = ObservationSequence.for_rdi(
            science_target=self.science_target,
            ref_target=self.ref_target,
            science_path_name=self.science_path,
            ref_path_name=self.ref_path,
            exposure_time=self.exposure_time,
            start_time=self.start_time,
            frame_time=frame_time
        )
        
        # Check timing
        assert seq[0].start_time == self.start_time
        assert seq[1].start_time == self.start_time + frame_time