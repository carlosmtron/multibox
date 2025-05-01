# MultiBox: Climate Energy Balance Models

## Overview

**MultiBox** is a collection of climate energy balance models (EBMs) that simulate Earth's climate system through different box configurations. The repository includes implementations of two-box, three-box, and multi-box models that compute temperature distributions based on solar radiation, albedo, and energy transport mechanisms.

These models were developed as part of a thesis research project and offer tools for studying climate dynamics through the lens of **energy balance** and **maximum entropy production (MEP)** principles.

## Features

- 🌞 **Solar Radiation Model**: Computes incoming solar radiation as a function of latitude.  
- 📦 **Multiple Model Configurations**:
  - Two-box model (polar and equatorial zones)
  - Three-box model (southern, equatorial, and northern zones)
  - Multi-box model (custom number of latitude bands)
- 🧰 **Data Processing Tools**: Scripts for processing and visualizing climate and satellite data.
- 📊 **Validation Against Observations**: Comparisons with:
  - NCEP/NCAR Reanalysis temperature data
  - CERES satellite radiation measurements
  - Published estimates of meridional heat transport

## Requirements

- [`Python 3.x`](https://python.org)  
- [`numpy`](https://numpy.org)  
- [`scipy`](https://scipy.org)  
- [`matplotlib`](https://matplotlib.org)  
- [`xarray`](https://docs.xarray.dev)

## Usage

### Solar Radiation Model

The `SWA.py` module calculates the mean annual solar radiation as a function of latitude, accounting for Earth's axial tilt and orbital characteristics:

```python
import SWA

# Calculate for latitude -90° (South Pole) with albedo 0.31
latitude = -90
albedo = 0.31
radiation = SWA.SWA_calc(latitude, albedo)

print(f"Annual mean absorbed radiation: {radiation} W/m²")
```

### Data Processing Tools

The repository includes scripts for working with observational datasets:

- `mapa_albedo.py`: Calculates albedo from satellite data and estimates surface temperatures.
- `temperaturas_tierra.py`: Computes zonal temperature averages with area weighting.

### Climate Models

Core model scripts include:

- `twobox_Lorenz.py`: Two-box model (polar and equatorial zones)
- `threebox.py`: Three-box model (southern, equatorial, and northern zones)
- `multibox_2.0.py`: Multi-box model with a configurable number of latitude bands

## Validation

The models have been validated against observational and reanalysis data:

- 🌡️ Comparison of modeled temperatures with NCEP/NCAR Reanalysis
- ☀️ Comparison of radiation fluxes with CERES satellite observations
- 🔁 Comparison of meridional heat transport with literature values

## Documentation

For an in-depth explanation of the models and theoretical foundations, refer to:

- 📘 **Thesis (in Spanish)**: [http://dx.doi.org/10.13140/RG.2.2.36246.97603](http://dx.doi.org/10.13140/RG.2.2.36246.97603)
- 📚 **Project Wiki**: Available via the GitHub [DeepWiki](https://deepwiki.com/carlosmtron/multibox)

## License

This project is licensed under the **GNU General Public License v3.0** – see the [LICENSE](LICENSE) file for details.

## Author

**Carlos Silva**  
📧 csilva@fceia.unr.edu.ar

## Citation

If you use this code in your research, please cite:

```
Silva, C. M. (2024). MultiBox: Climate Energy Balance Models. GitHub repository. https://github.com/carlosmtron/multibox
```

And the thesis:

```
Silva, C. M. (2024). El clima de la Tierra como un estado estacionario de máxima producción de entropía. http://dx.doi.org/10.13140/RG.2.2.36246.97603
```

## Acknowledgments

- **NCEP/NCAR** for reanalysis datasets  
- **CERES Team** for satellite radiation measurements  
- Researchers and authors in the field of EBMs and MEP theory  
