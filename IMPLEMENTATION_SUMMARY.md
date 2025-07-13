# Implementation Summary

## What Needs to Be Done

### 1. Fix the Architecture

**Current Problem**: We're treating all sources the same through the coronagraph, which is physically incorrect.

**Solution**: Implement source-aware light paths where:
- Star uses `coronagraph.stellar_intens()`
- Planets use `coronagraph.offax()` 
- Disk uses `coronagraph.psf_datacube` convolution

### 2. Use Existing Tools

**Don't**:
- Create new resampling systems
- Pre-create xarray datasets
- Write spaghetti code

**Do**:
- Use `flux_conserving_affine` from `image_transforms.py` for resampling
- Evaluate flux from ExoVista at execution time: `system.star.spec_flux_density(wavelength, time)`
- Follow the patterns from `observation.py`

### 3. Clean Pipeline Pattern

**User Interface** (what they see):
```python
pipeline = Path.primary(params) >> Path.coronagraph(params) >> Path.detector(params)
```

**Internal Implementation** (what actually happens):
- Three different paths for star, planets, disk
- Each uses the appropriate coronagraph physics
- Flux evaluation happens at execution time

### 4. Fix ExoVista Integration

**Current Issue**: `planetbins.dat` is in `ExoVista/ExoVista/data` not `ExoVista/data`

**Also Need**:
- Proper orbit propagation for planet positions
- Correct pixel scale conversions
- Handle the ExovistaSystem constructor properly (expects Path object)

### 5. Key Code to Write

1. **SourceAwareLightPath** class that:
   - Takes user's single pipeline
   - Creates three internal paths
   - Evaluates flux at execution time
   - Routes to appropriate coronagraph function

2. **Coronagraph step replacements**:
   - `apply_coronagraph_to_star()` - uses stellar intensity
   - `apply_coronagraph_to_planets()` - uses off-axis PSFs
   - `apply_coronagraph_to_disk()` - uses PSF datacube

3. **Integration with Observatory**:
   - Observatory should use SourceAwareLightPath internally
   - Handle the PropagationContext properly
   - Add diameter to context or get from observation scenario

## The Right Approach

1. Start with the patterns from `observation.py` - they work!
2. Wrap them in a clean >> pipeline interface
3. Don't overcomplicate - the physics is already implemented
4. Test with real yippy and ExoVista data

## What NOT to Do

- Don't create complex resampling systems (use flux_conserving_affine)
- Don't pre-evaluate ExoVista data
- Don't treat all sources the same through coronagraph
- Don't ignore the existing working code in observation.py