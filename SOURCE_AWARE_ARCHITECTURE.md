# Source-Aware Light Path Architecture

## Overview

The key insight is that **star, planets, and disk interact differently with the coronagraph**:

- **Star**: Uses the stellar intensity map (`stellar_intens`)
- **Planets**: Use off-axis PSFs based on their positions (`offax`)
- **Disk**: Uses PSF datacube convolution (`psf_datacube`)

## The Problem

Previously, we were trying to handle all sources the same way through the coronagraph step. This is physically incorrect and leads to spaghetti code.

## The Solution

### User Interface (Simple)

The user defines a single pipeline using the `>>` operator:

```python
pipeline = (PathPipeline(Path.primary(primary_params)) >> 
           Path.coronagraph(coronagraph_params) >> 
           Path.filter(filter_params) >> 
           Path.detector(detector_params))
```

### Internal Implementation (Physics-Aware)

Internally, the system creates **three different paths** - one for each source type:

1. **Star Path**:
   ```python
   Primary → Coronagraph(stellar_intens) → Filter → Detector
   ```

2. **Planet Path**:
   ```python
   Primary → Coronagraph(offax_psf) → Filter → Detector
   ```

3. **Disk Path**:
   ```python
   Primary → Coronagraph(psf_datacube) → Filter → Detector
   ```

## Key Design Principles

### 1. No Pre-Creation of Data

- **Don't** create xarray datasets ahead of time
- **Do** evaluate flux from ExoVista at specific times/wavelengths during execution
- This follows the pattern in `observation.py`:
  ```python
  star_flux = system.star.spec_flux_density(wavelength, time)
  ```

### 2. Use Existing Tools

- Use `flux_conserving_affine` from `image_transforms.py` for resampling
- Don't create new resampling systems
- This is already proven to work correctly

### 3. Source-Specific Coronagraph Interactions

Following `observation.py`, each source type has its own coronagraph function:

```python
def apply_coronagraph_to_star(data, context, coronagraph_params):
    # Get stellar diameter in λ/D units
    stellar_diam_lod = star.angular_diameter.to(u.lod, ...)
    
    # Get stellar intensity map
    stellar_intens = coronagraph.stellar_intens(stellar_diam_lod)
    
    # Apply to flux
    return star_flux * stellar_intens

def apply_coronagraph_to_planets(data, context, coronagraph_params):
    # Propagate orbits to get positions
    # For each planet:
    psf = coronagraph.offax(x, y, lam=wavelength, D=diameter)
    planet_map = planet_flux * psf
    
    # Sum all planet contributions
    return sum(planet_maps)

def apply_coronagraph_to_disk(data, context, coronagraph_params):
    # Resample disk to coronagraph pixel scale
    resampled = flux_conserving_affine(disk_flux_map, ...)
    
    # Convolve with PSF datacube
    return np.einsum('ij,ijxy->xy', resampled, coronagraph.psf_datacube)
```

## Implementation Strategy

### SourceAwareLightPath Class

```python
class SourceAwareLightPath:
    def __init__(self, user_path, include_sources=None):
        self.user_path = user_path
        self.include_sources = include_sources or {
            'star': True, 'planets': True, 'disk': True
        }
        # Create three internal paths
        self._create_source_specific_paths()
    
    def __call__(self, system, context):
        # Process each source type through its specific path
        results = {}
        
        if self.include_sources['star']:
            star_flux = system.star.spec_flux_density(wavelength, time)
            results['star'] = self._run_star_path(star_flux, context)
            
        # Similar for planets and disk...
        
        return IntermediateData(results)
```

### Observatory Integration

The Observatory would use this internally:

```python
class SourceAwareObservatory:
    def __init__(self, light_path):
        # Convert user's single path to source-aware path
        self.aware_path = create_coronagraph_aware_path(light_path)
    
    def run(self, observation_sequence):
        for obs in observation_sequence:
            # The aware_path handles the three source types
            result = self.aware_path(obs.target.system, context)
```

## Benefits

1. **Clean User Interface**: User defines one pipeline with `>>`
2. **Correct Physics**: Each source uses appropriate coronagraph interaction
3. **No Spaghetti Code**: Clear separation of concerns
4. **Flexible**: User can enable/disable sources
5. **Efficient**: Only evaluates flux when needed

## Comparison to Original `observation.py`

Our new architecture closely follows the proven pattern from `observation.py`:

| Original observation.py | New Architecture |
|------------------------|------------------|
| `gen_star_count_rate()` | `apply_coronagraph_to_star()` |
| `gen_planet_count_rate()` | `apply_coronagraph_to_planets()` |
| `gen_disk_count_rate()` | `apply_coronagraph_to_disk()` |
| Evaluates flux at specific λ,t | Same |
| Uses `flux_conserving_affine` | Same |
| Handles sources separately | Same |

## Next Steps

1. Integrate `SourceAwareLightPath` into the Observatory
2. Add proper orbit propagation for planet positions
3. Handle wavelength-dependent observations
4. Add validation and error handling