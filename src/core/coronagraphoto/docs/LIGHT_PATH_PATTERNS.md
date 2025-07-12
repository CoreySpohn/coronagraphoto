# Light Path Patterns in Coronagraphoto v2

## Overview

The new coronagraphoto v2 architecture introduces elegant functional composition patterns for creating light paths, replacing the old cumbersome approach of manually defining and composing step functions.

## The Problem with the Old Pattern

The original approach required defining separate functions for each step:

```python
# OLD PATTERN (cumbersome)
def primary_step(data, params, context):
    # Apply primary mirror effects
    return modified_data

def coronagraph_step(data, params, context):
    # Apply coronagraph effects
    return modified_data

# Manual composition was tedious and error-prone
result = detector_step(
    coronagraph_step(
        primary_step(data, primary_params, context),
        coronagraph_params, context
    ),
    detector_params, context
)
```

This approach was:
- ❌ **Verbose**: Required separate function definitions for each step
- ❌ **Error-prone**: Manual composition was fragile and hard to read
- ❌ **Inflexible**: Difficult to create conditional or dynamic paths
- ❌ **Non-composable**: Hard to reuse and combine path components

## The New Patterns

Coronagraphoto v2 introduces four elegant patterns for creating stateless light paths:

### 1. Pipeline Pattern (Recommended)

**Most readable for linear paths** - Uses the `>>` operator for left-to-right composition:

```python
# Create a complete light path
path = (PathPipeline(Path.primary(primary_params)) >> 
        Path.coronagraph(coro_params) >> 
        Path.filter(filter_params) >> 
        Path.detector(det_params))

# Execute the path
result = path(data, context)
```

**Advantages:**
- ✅ **Readable**: Left-to-right flow matches intuition
- ✅ **Chainable**: Easy to extend with `>>` operator
- ✅ **Natural**: Mimics Unix pipe syntax

### 2. Functional Composition

**Mathematical composition** - Functions applied right-to-left:

```python
# Compose functions mathematically
path = compose(
    Path.detector(det_params),
    Path.filter(filter_params),
    Path.coronagraph(coro_params),
    Path.primary(primary_params)
)

# Execute the composed function
result = path(data, context)
```

**Advantages:**
- ✅ **Pure functional**: Mathematical composition semantics
- ✅ **Concise**: Single function call for complex paths
- ✅ **Familiar**: Standard functional programming pattern

### 3. Builder Pattern

**Flexible for complex paths** - Fluent interface with conditional logic:

```python
# Build paths conditionally
builder = PathBuilder()
builder.primary(primary_params)
builder.coronagraph(coro_params)

if include_filter:
    builder.filter(filter_params)

if include_speckles:
    builder.speckles(speckle_params)

builder.detector(det_params)

# Build the final path
path = builder.build()
result = path(data, context)
```

**Advantages:**
- ✅ **Flexible**: Easy conditional composition
- ✅ **Readable**: Clear step-by-step construction
- ✅ **Extensible**: Easy to add custom components

### 4. Registry Pattern

**Dynamic composition** - Build paths from configuration:

```python
# List available components
components = Path.list_components()
print(components)  # {'primary': 'optics', 'coronagraph': 'optics', ...}

# Create paths from component names
component_configs = [
    ('primary', primary_params),
    ('coronagraph', coro_params),
    ('detector', det_params)
]

path_functions = [Path.from_registry(name, params) 
                 for name, params in component_configs]
path = compose(*reversed(path_functions))
```

**Advantages:**
- ✅ **Dynamic**: Build paths from configuration files
- ✅ **Discoverable**: Components are self-documenting
- ✅ **Extensible**: Easy to add new components

## Usage Examples

### Basic Usage

```python
import astropy.units as u
from coronagraphoto.light_paths import (
    Path, PathBuilder, PathPipeline, compose,
    PrimaryParams, CoronagraphParams, DetectorParams
)

# Define parameters
primary_params = PrimaryParams(
    diameter=6.5 * u.m,
    reflectivity=0.95
)

coro_params = CoronagraphParams(
    inner_working_angle=0.1 * u.arcsec,
    outer_working_angle=1.0 * u.arcsec,
    throughput=0.1,
    contrast=1e-10
)

detector_params = DetectorParams(
    pixel_scale=0.01 * u.arcsec,
    read_noise=3 * u.electron,
    dark_current=0.001 * u.electron / u.s
)

# Create path using pipeline pattern
path = (PathPipeline(Path.primary(primary_params)) >> 
        Path.coronagraph(coro_params) >> 
        Path.detector(detector_params))

# Execute
result = path(data, context)
```

### Custom Components

```python
# Define custom function
def custom_gain(data, context, gain=2.0):
    """Apply gain to star flux."""
    if data.star_flux is not None:
        boosted_flux = data.star_flux * gain
        return data.update(star_flux=boosted_flux)
    return data

# Use in any pattern
path = (PathBuilder()
        .primary(primary_params)
        .custom(lambda d, c: custom_gain(d, c, gain=1.5))
        .coronagraph(coro_params)
        .detector(detector_params)
        .build())
```

### Advanced Composition

```python
# Combine patterns
basic_optics = (PathPipeline(Path.primary(primary_params)) >> 
               Path.coronagraph(coro_params))

detection_chain = compose(
    Path.detector(detector_params),
    Path.filter(filter_params)
)

# Create composite path
full_path = basic_optics >> detection_chain
```

## Component Registration

### Using the Decorator

```python
from coronagraphoto.light_paths import path_component

@path_component("custom_component", "preprocessing")
def my_custom_step(data, context, params):
    """Custom preprocessing step."""
    # Your implementation here
    return processed_data

# Automatically available in registry
func = Path.from_registry("custom_component", my_params)
```

### Available Components

| Component | Category | Description |
|-----------|----------|-------------|
| `primary` | optics | Primary mirror collection and reflection |
| `coronagraph` | optics | Coronagraph suppression and PSF effects |
| `filter` | optics | Spectral filtering and bandpass |
| `detector` | hardware | Detector noise and quantization |
| `speckles` | noise | Speckle noise modeling |

## Pattern Comparison

| Pattern | Best For | Readability | Flexibility | Performance |
|---------|----------|-------------|-------------|-------------|
| Pipeline | Linear paths | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Functional | Mathematical composition | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Builder | Complex/conditional paths | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Registry | Dynamic/configurable paths | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Migration Guide

### From Old Pattern

```python
# OLD
def process_observation(data, params, context):
    data = apply_primary(data, params.primary, context)
    data = apply_coronagraph(data, params.coronagraph, context)
    data = apply_detector(data, params.detector, context)
    return data

# NEW
path = (PathPipeline(Path.primary(params.primary)) >> 
        Path.coronagraph(params.coronagraph) >> 
        Path.detector(params.detector))
result = path(data, context)
```

### Backward Compatibility

The old functions are still available for gradual migration:

```python
# These still work
from coronagraphoto.light_paths import (
    primary_step, coronagraph_step, detector_step
)

# But use the new patterns for new code
```

## Benefits of the New Approach

### ✅ **Cleaner Code**
- Eliminates boilerplate step function definitions
- Reduces code duplication and maintenance burden
- Makes intent clearer through composition patterns

### ✅ **Better Composability**
- Easy to create reusable path components
- Simple to combine and extend existing paths
- Clean separation of concerns

### ✅ **Improved Readability**
- Pipeline pattern reads naturally left-to-right
- Builder pattern shows construction step-by-step
- Functional composition is mathematically clear

### ✅ **Enhanced Flexibility**
- Conditional path construction with builder pattern
- Dynamic path creation from configuration
- Easy integration of custom components

### ✅ **Maintained Statefulness**
- All functions remain pure and stateless
- No side effects or hidden state
- Compatible with JAX transformations

### ✅ **Type Safety**
- Full type annotations for all patterns
- Clear interfaces and function signatures
- IDE support and autocompletion

## Best Practices

### 1. Choose the Right Pattern
- Use **Pipeline** for most linear light paths
- Use **Functional** for mathematical transformations
- Use **Builder** for complex conditional logic
- Use **Registry** for configuration-driven paths

### 2. Keep Functions Pure
- All path functions should be stateless
- No side effects or global state modifications
- Input parameters should be immutable

### 3. Use Type Hints
- Always provide type hints for custom functions
- Use the provided type aliases: `PathFunction`, `PathStep`
- Leverage IDE support for better development experience

### 4. Test Thoroughly
- Test each pattern produces equivalent results
- Verify stateless behavior with repeated execution
- Check parameter and context immutability

### 5. Document Custom Components
- Use clear docstrings for custom functions
- Specify parameter types and return values
- Register components with descriptive names

## Performance Considerations

- **Functional composition** is fastest (single function call)
- **Pipeline pattern** has minimal overhead
- **Builder pattern** creates multiple intermediate objects
- **Registry pattern** has lookup overhead

All patterns are suitable for production use. Choose based on readability and maintainability needs rather than micro-optimizations.

## Future Enhancements

The new pattern system enables several future enhancements:

1. **JAX Integration**: Automatic JIT compilation of composed paths
2. **Parallel Execution**: Automatic parallelization of independent components
3. **Caching**: Smart caching of intermediate results
4. **Validation**: Automatic parameter validation and type checking
5. **Visualization**: Graphical representation of light paths

## Summary

The new light path patterns in coronagraphoto v2 provide a much more elegant and maintainable approach to creating light paths. They eliminate the cumbersome manual composition while maintaining all the benefits of functional programming: purity, composability, and testability.

**Start with the Pipeline pattern** for most use cases, and explore the other patterns as your needs become more complex. The investment in learning these patterns will pay dividends in cleaner, more maintainable code.