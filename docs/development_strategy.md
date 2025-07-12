# Coronagraphoto v2: Development and Testing Strategy

---

## 1. Introduction

This document outlines the development strategy and roadmap for implementing the v2 architecture of `coronagraphoto`, as described in `refactor_2_architecture.md`. The goal is to provide a clear, phased approach to development, guided by the principles of Test-Driven Development (TDD). This strategy prioritizes building a robust, correct, and maintainable framework first, while paving a clear path toward significant performance optimizations with JAX.

---

## 2. Guiding Principles

-   **Test-Driven Development (TDD):** Development will follow the Red-Green-Refactor cycle. A failing test will be written first to define the desired functionality, followed by the minimal code to make the test pass, and finally, refactoring to improve the design.
-   **Modularity and Isolation:** Each component (`LightPathComponent`, `ReductionStep`, etc.) will be designed to be independent and testable in isolation. This simplifies testing and allows for incremental development.
-   **Clarity Before Optimization:** The initial focus will be on a correct, clear, and well-documented Python-native implementation. Performance optimization with JAX will be a distinct, subsequent phase, ensuring we don't sacrifice correctness for speed prematurely.
-   **Incremental Implementation:** The roadmap is broken into phases, with each phase delivering a tangible and testable part of the final system.

---

## 3. Development Roadmap

The development is structured into six phases, moving from foundational data structures to a fully-featured, high-performance simulation framework.

| Phase | Title | Goal | Key Components | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Foundational Components** | Establish the core, immutable data structures for defining an observation. | `Target`, `Observation`, `ObservationSequence` | To Do |
| **2** | **Core Simulation Engine** | Build the skeleton of the execution engine that can run a simulation. | `LightPathComponent`, `LightPath`, `ExecutionContext`, `Observatory` (skeleton), `DataProduct` | To Do |
| **3** | **End-to-End Simulation (Python-Native)** | Implement the concrete physics models to run a basic, scientifically valid simulation from scene to detector. | Concrete `LightPath` functions (`apply_primary`, `apply_coronagraph`, `apply_detector`) | To Do |
| **4** | **Post-Processing Framework** | Develop the tools to process the simulated data into science-ready images. | `ReductionStep`, `ReductionPipeline`, `ReferenceSubtract`, `Derotate`, `StackFrames` | To Do |
| **5** | **Performance Optimization (Hybrid Engine)** | Accelerate the most computationally intensive parts of the simulation using JAX. | JAX-native `LightPath` functions, "JAX Boundary" in `Observatory` | To Do |
| **6** | **Advanced Capabilities** | Implement advanced features like time-variable effects and enable fully JIT-compiled pipelines. | JAX-native scene generators, fully JIT-compiled `LightPath`s | To Do |

---

## 4. Testing Strategy

A robust testing suite is critical for ensuring the correctness of a scientific simulation tool. We will use `pytest` for structuring tests and `hypothesis` for property-based testing to uncover edge cases in our numerical code.

### 4.1. Test Structure

-   The `tests/` directory will mirror the `src/coronagraphoto/` directory structure.
-   Each module (e.g., `observation.py`) will have a corresponding test file (e.g., `tests/test_observation.py`).

### 4.2. Unit Tests with `pytest`

-   **Fixtures:** `pytest` fixtures will be used extensively to create reusable setup for complex objects like `Target`s, `ObservationSequence`s, and `Observatory` configurations.
-   **Parametrization:** `pytest.mark.parametrize` will be used to test functions against a wide range of explicit inputs, ensuring predictable behavior.
-   **Focus:** Each test function will focus on a single behavior of a single method.

### 4.3. Property-Based Tests with `hypothesis`

`hypothesis` is exceptionally well-suited for testing a scientific library, as it can automatically explore the input space of functions to find failing edge cases. We will use it to verify the fundamental properties of our components.

**Key Properties to Test:**

1.  **`ObservationSequence` Builders:**
    -   `for_adi` always produces the requested number of exposures.
    -   The total roll angle change across an ADI sequence is as expected.
    -   Total exposure time is conserved.

2.  **Numerical Components (`Detector`, `Speckle`):**
    -   Input data types and shapes are handled correctly.
    -   The component behaves correctly with zero, very large, or NaN inputs.
    -   Statistical properties are maintained (e.g., noise from a `Detector` model has the correct mean and standard deviation).

3.  **Transformations (`flux_conserving_affine`):**
    -   **Flux Conservation:** The total flux in an image must be conserved after rotation or scaling. This is a critical property and a primary test case.
    -   **Idempotence:** Applying a rotation and then its inverse should return the original image (within some tolerance).
    -   **Continuity:** Small changes to the input image should result in small changes to the output image.

### 4.4. Integration Tests

-   Integration tests will verify that components work together correctly.
-   The primary integration test will be an end-to-end run:
    1.  **Planning:** Create an `ObservationSequence`.
    2.  **Execution:** Run it through an `Observatory` with a defined `LightPath`.
    3.  **Reduction:** Process the resulting `DataProduct` with a `ReductionPipeline`.
-   These tests will be slower and will run separately from the unit tests. They will validate the full workflow and catch issues at the boundaries between components.

---

## 5. Priority Tasks & Implementation Order

This section details the initial steps for **Phase 1 and 2**, providing a concrete starting point for development.

### Task 1: Define `IntermediateData` Structure (Phase 2 Prerequisite)

-   **Context**: As defined in the architecture, the simulation requires a data structure that keeps star, planet, and disk components separate for propagation through the `LightPath`.
-   **Goal**: Formalize and implement the `xarray.Dataset` structure for this `IntermediateData` object.
-   **Implementation**: Create a module (e.g., `coronagraphoto.core.data_models`) that defines this structure. This might include helper functions for creating or validating these objects.
-   **Test**: Write unit tests to ensure the structure can be created correctly and that it enforces the separation of star, planet, and disk components. This is a critical prerequisite for all `LightPath` function implementation.

### Task 2: `Observation` and `Target` (Phase 1)

-   **`Observation`:**
    -   **Implementation:** Create the `Observation` dataclass.
    -   **Test:** Verify that it correctly stores parameters like `start_time`, `exposure_time`, etc. Use `hypothesis` to generate strategies for creating valid `Observation` objects.
-   **`Target`:**
    -   **Implementation:** Create the `Target` wrapper class. Initially, it can just hold a path to a scene file.
    -   **Test:** Verify that the `Target` object is initialized correctly.

### Task 3: `ObservationSequence` (Phase 1)

-   **Implementation:** Create the `ObservationSequence` class with its list of `Observation`s. Implement the `for_adi` and `for_rdi` classmethod builders, ensuring they accept a `path_name`.
-   **Test:**
    -   Write unit tests for the basic list functionality.
    -   Write dedicated tests for each builder (`for_adi`, `for_rdi`).
    -   Use `hypothesis` to test properties: does `for_adi(n_exposures=10, ...)` always create a sequence of length 10? Are the roll angles distributed as expected?

### Task 4: Core `LightPath` Functions (Phase 2)

-   **Context**: The simulation is driven by pure functions. We need to create the first set of these.
-   **Implementation**:
    -   Create a new module, e.g., `coronagraphoto.light_paths`.
    -   Implement initial, simple versions of `apply_primary`, `apply_coronagraph`, etc. They should accept `(data, params, context)` and return `data`.
    -   Define simple dataclasses for the `params` of each function.
-   **Test**:
    -   Test each function in isolation. Since they are pure, they should be easy to unit test.
    -   Use mock parameter and context objects to verify the function's core logic.

### Task 5: Critical Utility: `flux_conserving_affine`

-   **Context:** As noted in the architecture document, a robust, JAX-compatible, flux-conserving affine transformation is a key enabler for many components (e.g., telescope roll). The existing `map_coordinates.py` provides a basis for interpolation, but it must be rigorously adapted and tested for flux conservation.
-   **Implementation:** Develop `flux_conserving_affine` as a standalone, pure function.
-   **Test (`hypothesis` is essential here):**
    -   **Primary Test:** The sum of the output array must equal the sum of the input array to within a tight numerical tolerance.
    -   Test with various transformations (rotation, scaling, translation).
    -   Test edge cases: empty arrays, arrays with NaNs, etc.

### Task 6: `Observatory` Executor Skeleton (Phase 2)

-   **`Observatory`:**
    -   **Implementation:** Create the `Observatory` class. Its `__init__` should accept a dictionary of named `LightPath`s. The `run` method should perform the two-phase process: 1) Plan the grid of `PropagationContext` objects based on the parameters of the functions in the selected path, and 2) Execute the `LightPath` for each context.
    -   **Test (with Mocks):**
        -   Verify that the `Observatory` correctly selects the `LightPath` based on the `path_name` in an `Observation`.
        -   Use mock functions in a test `LightPath` to verify that they are called in the correct order.
        -   Verify that the `run` method correctly constructs the time and wavelength grid from the (mock) parameter objects in the `LightPath`.

With these foundational pieces and their corresponding tests in place, the project will have a solid, verifiable core upon which the more complex physics models and processing steps can be built. 