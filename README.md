## Environment (mamba / conda)

From the project root:

```bash
mamba env create -f environment.yml
mamba activate dl-water
```

If `dl-water` already exists, or after you change `environment.yml`, run:

```bash
mamba env update -f environment.yml --prune
```

Alternatively: `pip install -r requirements.txt` inside any Python 3.11 env.

## Citations

Bastidas Pacheco, C. J., N. Atallah, J. S. Horsburgh (2023). High Resolution Residential Water Use Data in Cache County, Utah, USA, HydroShare, https://doi.org/10.4211/hs.0b72cddfc51c45b188e0e6cd8927227e
