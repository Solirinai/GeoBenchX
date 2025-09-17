Data used in the experiment can be downloaded [via link.](https://drive.google.com/file/d/10hLDCTMnaMAXyUCxFjUt4iK7tX9XyXZu/view?usp=drive_link)

The archaive contains folder Data with 2 subfolders and bibtex file with citations and links for the data sources.

Folder "Geodata" contains the files listed in GEO_CATALOG and RASTER_CATALOG. Change the GEO_CATALOG_PATH to the folder where you store these files.
Folder "StatData" contains the files listed in DATA_CATALOG. The DATA_CATALOG_PATH to the folder where you store these files.

The data were left as-is after downloading from the data APIs, except for the vector geodataset with country polygons and accumulated snow cover rasters. To make it easier to visually identify issues during development with data filtering or maps, the names of the countries in the geodata were harmonized with the names in the statistical datasets. Raster with snow cover data had a NoData assigned as minimum value, leading to long processing times while using raster processing tools. The NoData value was reassigned in QGIS to address this issue. These steps were not necessary for benchmarking itself. 
