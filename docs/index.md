# coronagraphoto

A JAX-accelerated coronagraphic observation simulator for HWO mission planning.

```{toctree}
:maxdepth: 2
:caption: Documentation

functional_approach
```

```{toctree}
:maxdepth: 2
:caption: API Reference

autoapi/coronagraphoto/index
```

## Overview

**coronagraphoto** is a Python library for simulating coronagraphic observations of exoplanetary systems. It provides:

- JAX-native implementations for GPU/TPU acceleration
- Modular source models (stars, planets, disks, zodiacal light)
- Coronagraph integration via [yippy](https://github.com/CoreySpohn/yippy)
- Realistic detector noise models
- Integration with [coronalyze](https://github.com/CoreySpohn/coronalyze) for post-processing

## Installation

```bash
pip install coronagraphoto
```

## Quick Start

```python
from coronagraphoto import ...
# See examples/ for usage
```
