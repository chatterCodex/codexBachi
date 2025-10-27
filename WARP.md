# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a research project for a Bachelor's thesis on **"Optimization of Cable-Road Layouts in Smart Forestry"**. The project implements multi-objective optimization algorithms for cable-road layout planning in forestry, integrating geospatial data processing, mechanical calculations, and genetic algorithms.

## Development Environment

### Python Environment Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running Tests
```powershell
# Run all tests
python -m pytest src/tests/

# Run specific test files
python -m pytest src/tests/test_cable_roads.py
python -m pytest src/tests/test_geometry.py
python -m pytest src/tests/test_mechanical_computations.py

# Run with verbose output
python -m pytest -v src/tests/
```

### Running Jupyter Notebooks
```powershell
# Start Jupyter and navigate to notebooks
jupyter notebook 01_Notebooks/

# Key notebooks for exploration:
# - interface.ipynb: Interactive cable road selection
# - optimization.ipynb: Multi-objective optimization examples
# - compute_cable_corridors.ipynb: Core computation workflows
# - debugging_and_visualization.ipynb: Visualization and debugging tools
```

### Running the Main Application
```powershell
# Execute core computations from src/main directory
cd src/main
python cable_road_computation_main.py

# Interactive interface (requires Jupyter)
python interface.py
```

## Code Architecture

### Core Module Structure

The project follows a modular architecture with clear separation of concerns:

**`src/main/`** - Core computation modules:
- `cable_road_computation_main.py`: Central orchestrator that generates and validates cable road configurations
- `cable_road_computation.py`: Core cable road computation logic (support trees, anchor points)
- `mechanical_computations.py`: Mechanical calculations (tension, forces, angles, structural validation)
- `geometry_utilities.py`: Geometric operations and calculations
- `optimization_execution.py`: Multi-objective optimization algorithms (Augmented ε-Constraint, weighted optimization)

**Class System**:
- `classes_cable_road_computation.py`: `Cable_Road`, `Support`, `forest_area` classes for managing cable road configurations
- `classes_geometry_objects.py`: 3D geometric primitives (`Point_3D`, `LineString_3D`)
- `classes_linear_optimization.py` & `classes_mo_optimization.py`: Optimization problem definitions

**Global Resources**:
- `global_vars.py`: Manages shared spatial data structures (KD-Trees) for efficient spatial queries across modules

**User Interface**:
- `frontend/`: Interactive visualization components (maps, charts, tables)
- `interface.py`: Main interactive interface using ipywidgets

### Data Flow Architecture

1. **Initialization**: `global_vars.init(height_gdf)` sets up spatial data structures
2. **Generation**: `generate_possible_lines()` creates cable road candidates using geometric and mechanical constraints
3. **Validation**: Multi-stage filtering by slope deviation, support trees, anchor configurations, and collision detection
4. **Optimization**: Single/multi-objective optimization of valid configurations
5. **Visualization**: Interactive selection and analysis through Jupyter interface

### Key Dependencies

- **Geospatial**: `geopandas`, `rasterio`, `shapely` for geographic data processing
- **Optimization**: `pymoo` (multi-objective), `pulp` (linear), `spopt` (spatial optimization)
- **Scientific Computing**: `numpy`, `scipy`, `scikit-learn`, `networkx`
- **Visualization**: `matplotlib`, `plotly` for 2D/3D plotting
- **ML/AI**: `torch` for neural network components

### Testing Strategy

Tests are organized by functional area:
- `test_cable_roads.py`: Cable road generation, validation, and mechanical properties
- `test_geometry.py`: Geometric operations and utilities
- `test_mechanical_computations.py`: Structural and mechanical calculations
- `test_classes.py`: Object model validation

Tests use helper functions from `helper_functions.py` to set up consistent test data (GeoDataFrames for lines, trees, height data).

### Important Implementation Notes

- **Spatial Indexing**: Uses KD-Trees for efficient spatial queries - initialize via `global_vars.init()` before spatial operations
- **3D Computations**: Custom 3D geometry classes handle elevation-aware calculations
- **Mechanical Validation**: Cable roads undergo multi-stage validation (collision detection, anchor holding capacity, support tension)
- **Memory Management**: Large pickle files (.pkl) contain cached results - excluded from git but may be present locally
- **Data Privacy**: Underlying forest maps are not included in repository for privacy reasons

### Performance Considerations

- Spatial operations are optimized using KD-Tree indexing
- Caching mechanisms store intermediate results in pickle files
- Multi-objective optimization can be computationally intensive for large forest areas
- Interactive visualizations work best with preprocessed datasets

This is research code focused on algorithmic development rather than production deployment. The architecture prioritizes modularity and experimental flexibility over enterprise concerns.