# Coronagraphoto Architecture v2: An Observatory & Reduction Framework

---

## 1. Motivation

The previous architecture, based on a `Session` and `Pipeline` model, successfully introduced a modular structure. However, practical application revealed that it conflated several distinct concerns: the *planning* of an observation, the *execution* of that observation by an instrument, and the *post-processing* or reduction of the resulting data. This made it difficult to model complex scenarios, such as reference star observations, and obscured the physical meaning of the simulation steps.

Drawing from the insights in `refactoring_thoughts.md`, this new architecture proposes a clearer separation of concerns, built around three core ideas:

1.  **Observation Sequence:** What are we observing and when?
2.  **Observatory & Light Paths:** How do we observe it? What is the instrument hardware and what path does light take through it?
3.  **Reduction Pipeline:** What do we do with the data after we get it?

This paradigm shift moves to a **functional approach** for the core simulation. The flow of light is modeled as a sequence of pure, stateless functions. This provides a more intuitive and powerful framework for building end-to-end simulations, with clear, testable components that map directly to physical processes. It is designed to be highly extensible and paves a clear path to a high-performance, JAX-native engine.

---

## 2. Core Concepts

This architecture is composed of several key components that work together to simulate an observation and process its data.

| Concept | Purpose | Key API / Pattern |
| :--- | :--- | :--- |
| **Target** | The astrophysical scene to be observed. | `ExovistaSystem` wrapper |
| **Observation** | A single, planned exposure (target, time, duration, path). | Dataclass / DTO |
| **ObservationSequence** | An ordered list of `Observation`s for a campaign. | `Builder` pattern |
| **Observatory** | The simulated telescope & instrument suite. Executes a sequence. | Executor |
| **LightPath** | A named sequence of pure functions defining a simulation path. | `(data, params, context) -> data` |
| **DataProduct** | The output of the `Observatory`, typically an `xarray.Dataset`. | Wrapper around `xarray` |
| **ReductionStep** | A single post-processing algorithm (e.g., RDI, ADI). | `Strategy` pattern |
| **ReductionPipeline**| A sequence of `ReductionStep`s to process a `DataProduct`. | Pipeline |

---

## 3. High-Level Workflow

The simulation process is divided into three distinct phases: Planning, Execution, and Reduction.

```mermaid
graph TD
    subgraph "1. Planning & Configuration"
        A[Define Hardware Components];
        B[Create Light Paths <br/>(named lists of functions)];
        C(Build ObservationSequence);
    end

    subgraph "2. Execution"
        D{Configure Observatory <br/> with dictionary of Light Paths};
        B --> D;
        C --> E{Observatory.run()};
        D --> E;
    end

    subgraph "3. Reduction"
        F[Create ReductionPipeline];
        E --> G[DataProduct];
        G --> H{ReductionPipeline.process()};
        F --> H;
    end

    H --> I[Final Science Data];

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#ccf,stroke:#333,stroke-width:2px
```

1.  **Planning & Configuration:** Hardware components (e.g., `Primary`, `Coronagraph`) are defined. One or more named `LightPath`s—lists of pure functions bound to hardware parameters—are created and collected in a dictionary. An `ObservationSequence` is constructed to specify a series of exposures, with each exposure referencing the name of the `LightPath` it should use.
2.  **Execution:** An `Observatory` is configured with the dictionary of `LightPath`s. Its `run` method executes the `ObservationSequence`. For each exposure, it uses the specified name to select the correct `LightPath` and executes its functions in order.
3.  **Reduction:** A `ReductionPipeline` is assembled from various `ReductionStep`s. This pipeline is then used to `process` the `DataProduct` from the observatory into a final, science-ready format.

---

## 4. Detailed Component Design

### 4.1. The "What": Planning an Observation

-   **`Target`**: A simple wrapper around a scene generator, such as an `ExovistaSystem`. It represents a single pointing direction at an astrophysical source.
-   **`Observation`**: A dataclass that captures the parameters for a single exposure: the `Target`, start time, exposure duration, and importantly, the `path_name` string that identifies which `LightPath` to use.
-   **`ObservationSequence`**: A container for a `list[Observation]`. It uses the **Builder Pattern** via classmethods to facilitate the creation of complex observing patterns.

```python
@dataclass(frozen=True)
class Observation:
    target: Target
    start_time: Time
    exposure_time: u.Quantity
    path_name: str
    # ... other params like roll_angle, dither position, etc.

class ObservationSequence:
    def __init__(self, observations: list[Observation]):
        self.observations = observations

    @classmethod
    def for_adi(cls, target: Target, path_name: str, n_exposures: int, ...) -> 'ObservationSequence':
        # Builder logic to generate roll-subtraction sequence for a specific path
        ...
```

### 4.2. The "How": Defining and Executing Light Paths

The core of the simulation is re-imagined as a functional pipeline of pure, stateless functions.

-   **Stateless Light Path Functions**: Each step in the simulation (e.g., `apply_coronagraph`, `apply_detector_noise`) is a pure Python function. This function takes an `IntermediateData` frame, a set of parameters, and the `PropagationContext`, and returns a new `IntermediateData` frame.

-   **`LightPath`**: A `LightPath` is simply a list of these pure functions. `functools.partial` is used to bind the static hardware parameters to the functions, creating a specific, configured processing step.

-   **`Observatory`**: This class is now a pure **Executor**. It is initialized with a dictionary of named `LightPath`s. Its primary method, `run`, takes an `ObservationSequence`. For each `Observation` in the sequence, it looks up the `LightPath` by its `path_name` and executes it.

```python
# In coronagraphoto.light_paths
from functools import partial

# Example stateless function
def apply_coronagraph(data: xr.Dataset, params: CoronagraphParams, context: PropagationContext) -> xr.Dataset:
    # Pure function logic to apply coronagraph model
    ...
    return new_data

# In a user script
# 1. Define hardware
primary_params = Primary(...)
coronagraph_params = Coronagraph(...)

# 2. Build Light Paths
photometry_path = [
    partial(apply_primary, params=primary_params),
    partial(apply_coronagraph, params=coronagraph_params),
    # ... more functions
]
light_paths = {"photometry": photometry_path}

# 3. Configure Observatory
observatory = Observatory(light_paths=light_paths)

# 4. Plan and run
obs_seq = ObservationSequence.for_adi(target=..., path_name="photometry", ...)
raw_data_product = observatory.run(obs_seq)
```

This functional approach creates a clean separation between the static *configuration* of the instrument and the dynamic *execution* of an observing plan.

### 4.3. The "Result" and "After": Data Products & Reduction

-   **`DataProduct`**: The result of `Observatory.run`. It is an `xarray.Dataset` containing the simulated data cubes, exposure information, and rich metadata about the simulation parameters, preserving a clear history of how the data was generated.

-   **`ReductionPipeline` & `ReductionStep`**: The `ReductionPipeline` processes a `DataProduct`. It holds a list of `ReductionStep`s, where each step is a self-contained algorithm (e.g., `ReferenceSubtract`, `AlignAndStack`). This uses the **Strategy Pattern**, allowing different reduction strategies to be easily composed and applied.

```python
class ReductionPipeline:
    def __init__(self, steps: list[ReductionStep]):
        self.steps = steps

    def process(self, data_product: xr.Dataset) -> xr.Dataset:
        for step in self.steps:
            data_product = step.process(data_product)
        return data_product
```

---

## 5. Design Patterns Utilized

This architecture explicitly leverages several design patterns to achieve its goals of clarity, modularity, and extensibility.

-   **Executor (`Observatory`)**: The `Observatory`'s role is simplified to purely executing a pre-defined plan, without containing logic for how to build that plan.
-   **Builder (`ObservationSequence` builders)**: Decouples the construction of a complex observing sequence from its representation, allowing for easy creation of standard observing modes like ADI and RDI.
-   **Strategy (`ReductionStep`)**: Allows post-processing algorithms to be interchanged easily in a `ReductionPipeline`.
-   **Pure Functions (`LightPath` functions)**: Using stateless functions as the core of the simulation makes the system easier to reason about, test, and optimize, especially with JAX.

---

## 6. Example End-to-End Workflow

This conceptual code demonstrates how the components work together in a typical RDI scenario.

```python
from functools import partial
from coronagraphoto.light_paths import apply_primary, apply_coronagraph, apply_filter, apply_detector
from coronagraphoto.hardware import Primary, Coronagraph, Detector, Filter

# 1. Configuration
# Define hardware parameter objects
primary = Primary(diameter=8*u.m)
coronagraph = Coronagraph(...)
detector = Detector(...)
green_filter = Filter(...)

# Manually compose the Light Paths
science_path = [
    partial(apply_primary, params=primary),
    partial(apply_coronagraph, params=coronagraph),
    partial(apply_filter, params=green_filter),
    partial(apply_detector, params=detector),
]
# For an RDI reference, we might use a different path (e.g., without a coronagraph)
reference_path = [
    partial(apply_primary, params=primary),
    partial(apply_filter, params=green_filter),
    partial(apply_detector, params=detector),
]

light_paths = {
    "science_imaging": science_path,
    "reference_imaging": reference_path,
}

# Instantiate the executor with its available modes
my_observatory = Observatory(light_paths=light_paths)


# 2. Planning: Define targets and sequence
science_target = Target(ExovistaSystem("scene.fits"))
ref_target = Target(ExovistaSystem("ref_scene.fits"))

# The builder now creates a sequence with different path names
obs_seq = ObservationSequence.for_rdi(
    science_target=science_target,
    ref_target=ref_target,
    science_path_name="science_imaging",
    ref_path_name="reference_imaging",
    exposure_time=1 * u.d,
    frame_time=300*u.s,
)


# 3. Execution: Run the full observation sequence
raw_data_product = my_observatory.run(obs_seq, seed=123)


# 4. Reduction: Define pipeline and process data
reduction_pipeline = ReductionPipeline([
    ReferenceSubtract(method="normalized_subtraction"),
    Derotate(),
    StackFrames(),
])

final_image = reduction_pipeline.process(raw_data_product)

# final_image is now a science-ready xarray.Dataset
```

---

## 7. The Observatory's Core Loop: A Functional Execution Engine

The power of this architecture lies in the `Observatory`'s ability to execute arbitrary, user-defined simulation pathways.

### 7.1. The `IntermediateData` Frame: A Multi-Component Container

A critical design choice, motivated by the physics in `fundamental_concepts.md`, is that the data frame passed between `LightPath` functions is not a single image. Different astrophysical sources (star, planets, disk) require fundamentally different propagation methods through the coronagraph. To accommodate this, the intermediate data frame must preserve the identity of each source until after the coronagraph has been applied.

The `IntermediateData` object is therefore defined as a rich container—specifically an `xarray.Dataset`—that holds separate `DataArray`s for each source type. A `load_scene` function is responsible for creating this initial dataset from a `Target`.

A typical `IntermediateData` dataset would contain:

*   `star_flux`: A `DataArray` with dimensions `(wavelength)` holding the star's spectral flux density.
*   `planet_flux`: A `DataArray` with dimensions `(planet, wavelength)` holding the spectral flux density for each planet.
*   `planet_coords`: A `DataArray` with dimensions `(planet, time, coord_xy)` holding the sky coordinates of each planet.
*   `disk_flux_map`: A `DataArray` with dimensions `(x, y, wavelength)` holding the disk's spectral flux density map.

This structure allows functions like `apply_coronagraph` to be implemented correctly and unambiguously.

### 7.2. The `PropagationContext`

To handle time- and wavelength-dependent effects, a `PropagationContext` object is passed alongside the data to every function in the `LightPath`. This object provides the state for a single, fundamental propagation step.

@dataclass(frozen=True)
class PropagationContext:
    """
    Provides the state for a single propagation through the LightPath.
    """
    time: Time
    wavelength: u.Quantity
    bandpass_slice: u.Quantity # The width of the wavelength bin, d(lambda)
    time_step: u.Quantity     # The duration of the time step, dt
    rng_key: Any              # JAX PRNG key

### 7.3. The Two-Phase Execution Process: Planning and Propagation

For each `Observation` in the `ObservationSequence`, the engine performs a two-phase process. The core idea is to first plan a grid of all necessary `PropagationContext`s, and then execute the chosen `LightPath` for each context. This allows components to signal their required time or wavelength resolution.

**Phase 1: Grid Planning**

Before any calculations, the engine interrogates the parameters of every function in the selected `LightPath` to construct a computational grid of `PropagationContext` objects. For example, a `Detector`'s parameters might specify a required `frame_time`, while a `Spectrograph`'s parameters would define the wavelength bins. The engine combines these requirements to create the minimal necessary grid of `(time, wavelength)` points.

**Phase 2: Grid Execution via `scan`**

The engine then iterates through the planned computational grid. For each `(t, w)` point, it creates a `PropagationContext` and then executes the `LightPath` by "scanning" through the list of functions—feeding the output of one as the input to the next.

```mermaid
graph TD
    subgraph Single Observation Execution
        A[Start Observation] --> B{Loop over Time Steps 't'};
        B --> C{Loop over Wavelength Bins 'w'};
        C --> D[Create PropagationContext(t, w, ...)];
        D --> E[Execute Light Path];
        subgraph E [ ]
            direction LR
            E_A(Initial Data) --> E_B{function_1.apply};
            E_B --> E_C(Data) --> E_D{...};
            E_D --> E_E(Data) --> E_F{function_n.apply};
            E_F --> E_G(Final Data for this Propagation);
        end
        E --> H[Append result to Wavelength Cube];
        H --> C;
        C -- End of Wavelength Loop --> I[Integrate Wavelength Cube];
        I --> B;
        B -- End of Time Loop --> J[Append result to Final DataProduct];
    end
```

This approach ensures maximum efficiency: work is only performed at the resolution necessary, but the framework can gracefully handle the complexity when required.

---

## 8. JAX Integration and Performance Roadmap

The functional `LightPath` design is exceptionally well-suited for JAX. An entire pathway composed of JAX-compatible functions can be `jax.jit`-compiled into a single, highly-optimized kernel.

### 8.1. The Path to a Fully-JITted Pipeline

The development roadmap allows for a gradual transition from a pure Python implementation to a fully accelerated one.

**Phase 1: Python-Native Implementation (Current Focus)**
*   **Goal**: Validate the core architectural concepts (`Observatory`, functional `LightPath`, `DataProduct`) and ensure physical correctness.
*   **Implementation**: All `LightPath` functions will be Python-native, operating on `xarray.Dataset`. JAX can still be used for specific numerical tasks within a function, but the pipeline itself is not JIT-compiled.

**Phase 2: Hybrid Execution Engine (Near-Term Goal)**
*   **Goal**: Accelerate performance-critical sections of the simulation.
*   **Implementation**:
    1.  Develop JAX-native versions of key `LightPath` functions that operate directly on `jax.Array` objects.
    2.  The `Observatory`'s execution engine will manage the "JAX Boundary," converting `xarray.Dataset` objects to raw `jax.Array`s when entering a JAX-native section of a `LightPath` and converting back when exiting.
    3.  This allows for mixing Python and JAX functions within a single `LightPath`.

**Phase 3: Fully JIT-Compiled Pipelines (Future Goal)**
*   **Goal**: Achieve maximum end-to-end performance.
*   **Implementation**:
    1. Develop a JAX-native scene generator for simpler cases (e.g., point sources, basic disk models).
    2. Create entire `LightPath`s composed solely of JAX-native functions.
    3. The `Observatory` can then `jax.jit` an entire `LightPath`, `jax.vmap` it over observation parameters, or even `jax.grad` it for sensitivity analyses.

This roadmap ensures that we build a robust and correct simulation framework first, while paving a clear and maintainable path toward significant performance optimizations.