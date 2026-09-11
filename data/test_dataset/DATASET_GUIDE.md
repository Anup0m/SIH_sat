# SatQuery AI — Laptop Testing Dataset Guide 🛰️

This directory contains clean, verified, lightweight satellite imagery across optical, GeoTIFF, SAR radar, and bi-temporal modalities.
All files are pre-screened and optimized to execute in under 1 second on a standard laptop CPU or RTX laptop GPU.

---

## 📂 Dataset Inventory & Suggested Queries

### 1. Optical Satellite Imagery (`data/test_dataset/optical/`)
| File Name | Description | Key Features | Recommended Query |
| :--- | :--- | :--- | :--- |
| `optical_sentinel2_bahamas_water.jpg` | Sentinel-2 True Color | Shallow tropical water, coral reefs, open ocean | *"Locate and map all water bodies in this scene."* |
| `optical_sentinel2_south_georgia_coastal.jpg` | Sentinel-2 True Color | Deep ocean water, coastal boundary, rocky terrain | *"Where is the water coastline located?"* |
| `optical_sentinel2_congo_forest.jpg` | Sentinel-2 True Color | Dense tropical rainforest canopy, river corridor | *"Highlight all green vegetation and forest zones."* |
| `optical_sentinel2_minsk_urban.jpg` | Sentinel-2 True Color | Urban city center, road network, water reservoir | *"Detect the urban built-up areas and roads."* |
| `optical_agricultural_monsoon.jpg` | Multispectral Optical | Crop parcels, irrigation channels, vegetative cover | *"Identify agricultural crop fields and vegetation."* |
| `optical_coastal_estuary.jpg` | Multispectral Optical | River delta, sediment outflow, coastal water | *"Analyze the water flow and shoreline boundaries."* |

---

### 2. GeoTIFF Multispectral Files (`data/test_dataset/geotiff/`)
*Real georeferenced TIFF format (.tif) imagery.*

| File Name | Sensor / Band Spec | Ground Coverage | Recommended Query |
| :--- | :--- | :--- | :--- |
| `landsat_rgb.tif` | Landsat-8 3-band GeoTIFF (8-bit) | Coastal estuary, sediment plumes, forest | *"Locate vegetation and water areas in this TIFF."* |
| `landsat_multispectral_urban.tif` | Landsat-8 True-Color GeoTIFF (8-bit) | Coastal wetlands, urban fringe, ocean boundary | *"Map the water bodies and coastal perimeter in this Landsat GeoTIFF."* |
| `sentinel2_sample.tif` | Sentinel-2 Multispectral GeoTIFF (16-bit) | Surface reflectance, agricultural parcels, drainage | *"What is the dominant land cover in this GeoTIFF?"* |
| `sentinel2_cloud_optimized_cog.tif` | Sentinel-2 COG GeoTIFF (16-bit) | Calibrated surface reflectance, coastal & inland water | *"Detect water channels and delineate boundaries in this COG."* |
| `global_earth_observation.tif` | Global True-Color GeoTIFF (8-bit, 3-band) | Continental land masses, ocean basins, atmosphere | *"Analyze water vs land percentages across the global footprint."* |
| `elevation_shade_terrain.tif` | Shaded Relief / DEM GeoTIFF (8-bit) | Mountain ridge lines, topographical valleys | *"Where are the low-elevation drainage basins and shadows located?"* |
| `landcover_aerial_orthophoto.tif` | LandCover.ai Aerial Orthophoto (.tif) | High-resolution rooftops, access roads, structures | *"Detect urban buildings and civil infrastructure in this orthophoto."* |

---

### 3. Synthetic Aperture Radar (SAR) (`data/test_dataset/sar/`)
*Microwave radar backscatter: penetrative, cloud-piercing, specular water reflection.*

| File Name | Polarization / Sensor | Radar Signature | Recommended Query |
| :--- | :--- | :--- | :--- |
| `sentinel1_sar_vh.tif` | Sentinel-1 C-Band SAR (VH GeoTIFF) | Cross-polarized volume scattering radar patch | *"Perform radar backscatter and water detection."* |
| `sentinel1_sar_vv.tif` | Sentinel-1 C-Band SAR (VV GeoTIFF) | Co-polarized surface roughness radar patch | *"Analyze surface roughness and radar intensity."* |
| `sentinel1_sar_hh_polarization.tif` | Sentinel-1 C-Band SAR (HH GeoTIFF) | Co-polarized horizontal backscatter intensity | *"Evaluate radar backscatter intensity and identify low-return water zones."* |
| `sentinel1_sar_hv_polarization.tif` | Sentinel-1 C-Band SAR (HV GeoTIFF) | Cross-polarized microwave volumetric returns | *"Analyze microwave cross-polarization and surface textures."* |
| `sar_sentinel1_dual_polarization.jpg` | Sentinel-1 Composite | False-color dual-pol radar composite | *"Describe radar backscatter patterns across terrain."* |
| `sar_agricultural_monsoon.jpg` | Sentinel-1 SAR | Soil moisture, crop moisture dielectric | *"Evaluate soil moisture and crop radar signature."* |
| `sar_coastal_estuary.jpg` | Sentinel-1 SAR | Specular water reflection vs land dielectric | *"Map water boundaries using radar backscatter."* |

---

### 4. Bi-Temporal Change Detection (`data/test_dataset/bitemporal/`)
*Coregistered pair of the same geographical footprint across two dates ($T_1$ and $T_2$).*

| Epoch T1 (Before) | Epoch T2 (After) | Dynamics | Recommended Query |
| :--- | :--- | :--- | :--- |
| `bitemporal_epoch_t1_before.jpg` | `bitemporal_epoch_t2_after.jpg` | Construction, road expansion, ground transformation | *"What changed between these observation dates?"* |

---

## 🚀 How to Test in SatQuery AI
1. Open the workstation in your browser at `http://localhost:8000`.
2. Drag and drop any image from `data/test_dataset/` into the upload zone (or choose 2 images for change / optical+SAR).
3. Type any question from the table above or speak via the voice microphone button.
4. Watch the agent dynamically analyze the real pixels, localize objects, calibrate confidence, and log the execution trace!
