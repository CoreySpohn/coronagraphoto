# Propagate Method Architecture Summary

## Overview

The `.propagate()` method architecture provides a clean, JAX-friendly approach for building optical pipelines with explicit method chaining and source-specific pipeline separation.

## Key Design Pattern

```python
# Create pipeline with propagate chaining
pipeline = Pipeline(
    primary.propagate(coronagraph).propagate(filter).propagate(detector)
)

# Execute on source
result = pipeline(source)
```

## Architecture Benefits

### 1. Source Separation for JAX Efficiency

By creating separate pipelines for each source type, we eliminate conditionals in the hot path:

```python
# Different pipelines for different sources
star_pipeline = telescope.create_star_pipeline()      # With coronagraph
planet_pipeline = telescope.create_planet_pipeline()  # With coronagraph  
reference_pipeline = telescope.create_reference_pipeline()  # No coronagraph
```

This approach is optimal for JAX because:
- No branching logic in JIT-compiled code
- Each pipeline can be independently optimized
- Consistent array shapes (return zeros for missing sources)

### 2. Explicit Method Chaining

The `.propagate()` method provides clear, readable composition:

```python
# Clear data flow
chain = primary.propagate(coronagraph).propagate(filter).propagate(detector)

# vs conditional logic (JAX-unfriendly)
if use_coronagraph:
    field = coronagraph.apply(field)  # BAD for JAX
```

### 3. Pipeline Wrapper Pattern

The `Pipeline` class provides a clean execution interface:

```python
class Pipeline:
    def __init__(self, component_chain):
        self.component_chain = component_chain
    
    def __call__(self, source):
        return self.component_chain.apply(source)
    
    def jit_compile(self):
        return jax.jit(self.__call__)
```

## Implementation Highlights

### Component Base Class

```python
class OpticalComponent(ABC):
    @abstractmethod
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply component effect."""
        pass
    
    def propagate(self, next_component) -> ChainedComponent:
        """Chain with next component."""
        return ChainedComponent(self, next_component)
```

### Multi-Source Observatory

```python
class Observatory:
    def observe(self, sources):
        results = {}
        
        # Process each source with its pipeline
        if 'star' in sources:
            results['star'] = self.star_pipeline(sources['star'])
        else:
            results['star'] = np.zeros(shape)  # Consistent shape
        
        # Similar for planets, disk, etc.
        return results
```

## Comparison with Previous Approaches

### IntermediateData Pattern (Rejected)
```python
# Previous approach - carries all data
data = IntermediateData(components={'star': ..., 'planets': ...})
result = pipeline(data)  # Requires conditionals inside
```

**Problems:**
- Conditionals inside pipeline for different sources
- Complex data management
- JAX compilation challenges

### Current Approach (Adopted)
```python
# Separate pipelines per source
star_result = star_pipeline(star_source)
planet_results = [planet_pipeline(p) for p in planet_sources]
```

**Benefits:**
- No conditionals in pipelines
- Simple array inputs/outputs
- Efficient JAX compilation

## JAX Optimization Strategy

### 1. Pipeline Compilation
```python
# Compile each pipeline once
star_pipeline_jax = star_pipeline.jit_compile()
planet_pipeline_jax = planet_pipeline.jit_compile()
```

### 2. Batched Processing
```python
# Process multiple planets efficiently
batched_planet_pipeline = jax.vmap(planet_pipeline)
all_planet_images = batched_planet_pipeline(planet_array)
```

### 3. Consistent Shapes
```python
# Always return same shape (zeros for missing)
if source is None:
    return np.zeros(detector_shape)
else:
    return pipeline(source)
```

## Usage Patterns

### Basic Pipeline Creation
```python
telescope = TelescopeConfig(
    primary_diameter=6.5,
    coronagraph_path='input/coronagraphs/demo',
    filter_wavelength=550e-9,
    detector_shape=(256, 256),
    pixel_scale=0.02
)

pipeline = Pipeline(
    telescope.primary.propagate(telescope.coronagraph)
                    .propagate(telescope.filter)
                    .propagate(telescope.detector)
)
```

### Multi-Source Observation
```python
observatory = Observatory(telescope)

sources = {
    'star': star_field,
    'planets': [planet1_field, planet2_field],
    'reference': reference_field
}

# Returns dict with all results
results = observatory.observe(sources, combine=False)

# Or combined image
combined = observatory.observe(sources, combine=True)
```

## Key Advantages

1. **JAX Performance**: No conditionals in compiled code paths
2. **Clarity**: Explicit `.propagate()` shows data flow
3. **Modularity**: Each source type has its own optimized pipeline
4. **Consistency**: Always returns arrays of expected shape
5. **Flexibility**: Easy to create custom pipelines
6. **Type Safety**: Simple array inputs/outputs

## Design Principles

1. **Separate pipelines for separate sources** - Avoid conditionals
2. **Return zeros for missing sources** - Maintain shape consistency
3. **Explicit is better than implicit** - Use `.propagate()` not operators
4. **Composition over configuration** - Build pipelines by chaining
5. **Pure functions** - No side effects for JAX compatibility

## Future Extensions

1. **Polychromatic support**: Extend to handle wavelength arrays
2. **GPU arrays**: Direct support for CuPy/JAX DeviceArray
3. **Parallel execution**: Process sources in parallel
4. **Caching**: Cache compiled pipelines by configuration
5. **Serialization**: Save/load pipeline configurations

This architecture provides a solid foundation for high-performance optical simulations while maintaining clean, understandable code.