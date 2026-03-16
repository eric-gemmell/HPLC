# HPLC Library — Architecture & Requirements

## Design Philosophy
- Fluent API with method chaining
- Immutable transformations (analysis and dispolay methods return new objects)
- Jupyter-first, multi-cell workflow
- Simple by default, powerful when needed

---

## Core Classes

### `HPLC` (module-level entry point)
Top-level namespace. Not instantiated — just a collection of factory functions.

```python
import HPLC

chrom = HPLC.load("file.dat")
compound = HPLC.compound(rt=3.5, spread=0.1, rf=1000)
composite = HPLC.composite([compound_a, compound_b])
```

**Open questions:**
- Should `HPLC.load()` auto-detect file format (.dat, .csv, .txt) or require explicit format?
- Should there be an `HPLC.batch_load()` for processing multiple files at once?

---

### `Signal`
Represents a single detector trace. Thin wrapper around the raw data — each signal owns its own time axis since detectors can sample at different rates.

**Core data:**
- `name` — detector name, e.g. `"UV"`, `"ELSD"`, `"RI"`
- `time` — `np.array` of time points (minutes)
- `values` — `np.array` of intensity values
- `units` — `str` or `None`, e.g. `"mAU"`, `"mV"`

---

### `Chromatogram`
Returned by `HPLC.load()`. The main class — holds signals, peaks, and analysis state.

**Core data:**
- `signals` — `list[Signal]`, one per detector channel
- `peaks` — list of `Peak` objects (empty until `identify_peaks()` is called)
- `metadata` — dict of file metadata (method, date, instrument, etc.). Structure varies by instrument source.

**Methods:**

| Method | Returns | Mutates? | Description |
|--------|---------|----------|-------------|
| `display(detector=None)` | self | No | Plot chromatogram. Stacked subplots, one per signal. Filter with `detector=`. |
| `identify_peaks(detector=None, threshold=None)` | new Chromatogram | No | Find peaks via curve fitting. |
| `list_peaks()` | None (prints) | No | Pretty-print table of found peaks. |
| `associate_compounds(compounds)` | new Chromatogram | No | Link compounds to peaks by retention time matching. |

| `display_publication_graph(filename=None)` | self | No | Render a clean, publication-ready figure. Saves if filename given. |
| `concentrations()` | dict | No | Calculate concentrations using associated compound response factors. |
| `export_results(filename)` | self | No | Export concentrations / peak data to CSV. |

**Open questions:**
- Should `display()` return a Plotly figure object too, for further customization? e.g. `fig = chrom.display().figure`

---

### `Chromatogram3D`
Represents 3D chromatographic data — a matrix of intensity over time × wavelength (e.g. DAD/PDA detector data). Fundamentally different from `Chromatogram` which holds discrete detector traces.

**Core data:**
- `time` — `np.array` of time points (minutes)
- `wavelengths` — `np.array` of wavelength values (nm)
- `intensity` — 2D `np.array` (time × wavelength)
- `metadata` — dict of file metadata

**Methods:**

| Method | Returns | Mutates? | Description |
|--------|---------|----------|-------------|
| `display()` | self | No | Interactive 3D surface or contour plot via Plotly. |
| `extract_signal(wavelength)` | Signal | No | Slice a single wavelength trace from the 3D data. |

**Future:** `extract_signal()` returns a `Signal` object, which can be used to construct a `Chromatogram` from 3D data when only discrete traces are available.

**Open questions:**
- Should `HPLC.load()` auto-detect and return the appropriate type (`Chromatogram` vs `Chromatogram3D`)?
- Should `Chromatogram3D` support `extract_chromatogram(wavelengths=[...])` to build a full `Chromatogram` from multiple slices?

---

### `Compound`
Represents a known compound you expect to find in the chromatogram.

```python
pNPS = HPLC.compound(
    name="pNPS",           # optional human-readable name
    rt=3.5,                # expected retention time (minutes)
    spread=0.1,            # standard deviation (σ) of peak width in minutes
    rf=1000                # response factor: number or callable
)
```

**Core data:**
- `name` — optional label
- `rt` — expected retention time
- `spread` — standard deviation (σ) of the peak in minutes. Used for matching: peaks within ±2–3σ of the expected RT are candidates.
- `rf` — response factor. Either:
  - A number: `concentration = area / rf`
  - A callable: `concentration = rf(area)` for nonlinear responses

**Open questions:**
- Do compounds need a `detector` field? e.g. "this compound is only visible on ELSD"

---

### `CompositeCompound`
For compounds that produce multiple peaks (like your GlcNAc example).

```python
GlcNAc = HPLC.composite(
    name="GlcNAc",
    compounds=[GlcNAc_part_1, GlcNAc_part_2]
)
```

**Behavior:**
- Associates with multiple peaks
- `concentrations()` sums or combines the individual peak contributions
- Displays as a single logical compound in output

**Open questions:**
- How should concentration be calculated — sum of individual concentrations? Weighted?
- Should the individual parts still be visible in `list_peaks()` or collapsed under the composite name?

---

### `Peak`
Represents a detected peak. Created internally by `identify_peaks()`.

**Core data:**
- `id` — integer index
- `rt` — retention time (peak maximum)
- `area` — integrated area
- `height` — peak height
- `width` — peak width at half-maximum or base
- `bounds` — (start_time, end_time) of integration
- `compound` — associated `Compound` or `None`
- `fit_params` — curve fitting parameters (if Gaussian/other model was used)

**Open questions:**
- Should peaks store the raw fitted curve data for display?
- Do you need asymmetric peak models (tailing, fronting) or is Gaussian sufficient?

---

## Association Logic

### Global RT Shift

Peaks in a chromatogram tend to move together — column aging, temperature drift, and mobile phase differences shift *all* retention times, not individual ones. Rather than each compound independently flagging a mismatch, we model this as a single global offset:

- **`rt_shift`** — a chromatogram-level parameter (estimated or user-provided) representing the global offset in minutes.
- **Expected RT** for matching: `compound.rt + rt_shift`
- **Auto-estimation**: find the single offset that minimizes total distance between all expected RTs and observed peaks (1D alignment step before per-compound matching).

This can be passed explicitly or auto-estimated during `associate_compounds()`:

```python
chrom = chrom.associate_compounds(compounds, rt_shift="auto")  # estimate from data
chrom = chrom.associate_compounds(compounds, rt_shift=0.3)     # manual override
```

### Per-Compound Matching

After applying the global RT shift, for each compound:
1. Compute `expected_rt = compound.rt + rt_shift`
2. Find peaks within ±2–3σ (`compound.spread`) of `expected_rt`
3. If exactly one peak → associate
4. If multiple peaks → pick closest to `expected_rt`? Or error?
5. If no peaks → warn

For `CompositeCompound`, each sub-compound is matched independently, then grouped.

**Open questions:**
- What happens if two compounds claim the same peak?
- Should there be a confidence score for the match?
- Should the shift model support linear scaling (proportional shift) in addition to constant offset?

---

## File Format Support

Your example filename: `EGG_HILIC_GlcNAc_0.2mM_20ul_80%MeCN_pH3_20mM_NH4OHC13.03.2026 17-17-16.dat`

**Open questions:**
- What instrument/software produces these .dat files? (Agilent ChemStation, Waters Empower, Shimadzu LabSolutions, etc.)
- Is there metadata in the file header or is it purely signal data?
- What other formats do you need? CSV exports, other instrument formats?
- Is the data one detector per file or multiple detectors in one file?

---

## Display / Plotting

Using Plotly for all visualization.

- **`Chromatogram.display()`** — the primary display method. Interactive Plotly in Jupyter. Handles all data types (1D signals, 3D data) — rendering strategy is an implementation detail.
- **`display_publication_graph()`** — static clean figure, proper axis labels, exportable to PNG/SVG/PDF.

**Open questions:**
- What should the publication graph look like? Minimal axes, compound labels above peaks, black/white or color?

---

## Example Workflow

```python
import HPLC

# Load and inspect
chrom = HPLC.load("./data/sample.dat")
chrom.display()

# Find peaks
chrom = chrom.identify_peaks()
chrom.list_peaks()

# Define known compounds
pNPS = HPLC.compound(name="pNPS", rt=3.5, spread=0.1, rf=1000)
pNP = HPLC.compound(name="pNP", rt=0.3, spread=0.1, rf=lambda area: 10000 * area ** 2)
GlcNAc_1 = HPLC.compound(name="GlcNAc (1)", rt=3.76, spread=0.1, rf=lambda area: 5600 * area ** 2)
GlcNAc_2 = HPLC.compound(name="GlcNAc (2)", rt=4.34, spread=0.1, rf=lambda area: 3400 * area ** 2)
GlcNAc = HPLC.composite(name="GlcNAc", compounds=[GlcNAc_1, GlcNAc_2])

# Associate and visualize
chrom = chrom.associate_compounds([pNPS, pNP, GlcNAc])
chrom.display()

# Get results
chrom.concentrations()
# → {"pNPS": 0.15, "pNP": 0.03, "GlcNAc": 0.22}

# Export
chrom.display_publication_graph(filename="figure_1.png")
chrom.export_results("results.csv")
```

---

## TODO / Decisions Needed
- [ ] File format parsing — what instrument?
- [ ] Response factor convention (multiply or divide?)
- [ ] Peak model — Gaussian only or also asymmetric?
- [ ] Composite concentration calculation method
- [ ] Multi-detector handling strategy
- [ ] Publication graph style preferences
- [ ] Batch processing API
