import os
from dotenv import find_dotenv, load_dotenv
import zipfile
from typing import Annotated, Any, Dict, Literal, Union, List, Optional
import random
from pathlib import Path
import ast
import io
import base64

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# from langchain_core.tools import tool
from langchain_core.tools import StructuredTool
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import InjectedState
from langgraph.graph.message import add_messages

from typing_extensions import TypedDict

import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import plotly.express as px
import contextily as ctx
from osgeo import gdal, gdal_array, ogr, osr
from osgeo.gdalconst import *
from shapely.geometry import LineString

from geobenchx.utils import get_dataframe_info


_ = load_dotenv(find_dotenv())

DATA_CATALOG_PATH = os.getenv('STATDATAPATH')
GEO_CATALOG_PATH = os.getenv('GEODATAPATH')
SCRATCH_PATH = os.getenv('SCRATCHPATH')

DATA_CATALOG = {
    "Forest area (sq. km)": "API_AG.LND.FRST.K2_DS2_en_csv_v2_2627",
    "Forest area (% of land area)": "API_AG.LND.FRST.ZS_DS2_en_csv_v2_102",
    "Electric power consumption (kWh per capita)": "API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_2767",
    "Annual freshwater withdrawals, total (billion cubic meters)": "API_ER.H2O.FWTL.K3_DS2_en_csv_v2_8364",
    "Annual freshwater withdrawals, total (% of internal resources)": "API_ER.H2O.FWTL.ZS_DS2_en_csv_v2_7098",
    "Agriculture, value added (% of GDP)": "API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_4614",
    "GDP per capita (current US$)": "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_77536",
    "Labor force, total": "API_SL.TLF.TOTL.IN_DS2_en_csv_v2_4341",
    "Net migration": "API_SM.POP.NETM_DS2_en_csv_v2_91",
    "Fertility rate, births per woman": "API_SP.DYN.TFRT.IN_DS2_en_csv_v2_821",
    "Population, total": "API_SP.POP.TOTL_DS2_en_csv_v2_56",
    "Rural population, total": "API_SP.RUR.TOTL_DS2_en_csv_v2_6835",
    "Greenhouse gases emission, per capita, tons of carbon dioxide-equivalents ":"GHG_emissions_OWID_wide.csv",
    "CO2 emissions per capita, tons":"per-capita-co-emissions_OWID_tons.csv",
    "Incidence of Tuberculosis Disease, 2023, Massachusetts Counties":"Incidence of Tuberculosis Disease 2023 Massachusetts Counties.csv",
    "Incidence of Tuberculosis Disease, 2023, New York State Counties": "NYS TB cases 2023.csv",
    "Rail lines (total route-km)":"API_IS.RRS.TOTL.KM_DS2_en_csv_v2_1937",
    "Regional GDP in departments (provinces) of Peru, constant prices 2007, thousand of soles": "Regional GDPs Peru.csv"
}

GEO_CATALOG = {
    "Countries": "WB_countries_Admin0_10m.shp",
    "Amtrak railway stations": "Amtrak_Stations.shp",
    "Railway lines in Bangladesh":"Bangladesh_gis_osm_railways_free_1.shp",
    "Cities and Towns of the United States, 2014": "bx729wr3020.shp",
    "Railway Network of North America": "North_American_Rail_Network_Lines.shp",
    "Current Wildland Fire Incident Locations, size in acres": "Incidents.shp",
    "Rivers in North America": "northamerica_rivers_cec_2023.shp",
    "Lakes in North America": "northamerica_lakes_cec_2023.shp",
    "USA counties borders": "tl_2024_us_county.shp",
    "USA states borders": "tl_2024_us_state.shp",
    "Earthquakes occurences and magnitude March 15- February 14 2025": "earthquakes_30_days_Feb142025.shp",
    "Rivers in South America": "rivers_samerica_37330.shp",
    "Seaports of Latin America": "Ports_Latin_America.shp",
    "Railways in Brazil": "hotosm_bra_railways_lines_shp.shp",
    "Mineral extraction facilities in Africa": "Africa mineral facilities.shp",
    "Power stations in Africa": "Africa Power Stations.shp",
    "Railways in Africa": "Africa railways.shp",
    "Municipalities of Brazil": "bra_admbnda_adm2_ibge_2020.shp",
    "Regions of Peru":"per_admbnda_adm1_ign_20200714.shp",
    "Provinces of Peru": "per_admbnda_adm2_ign_20200714.shp",
    }

RASTER_CATALOG = {
    "Accumulated snow cover season 2023-2024, USA, inches": "sfav2_CONUS_2023093012_to_2024093012_processed.tif",
    "Accumulated snow cover season 2024-2025, USA, inches": "sfav2_CONUS_2024093012_to_2025052012_processed.tif",
    "Tibetan Plato South Asia flood extent, August 2018": "DFO_4665_From_20180802_to_20180810.tif",
    "Bangladesh population, 2018, people, resolution 1 km":"bgd_pd_2018_1km_UNadj.tif",
    "USA population 2020, people, resolution 1 km": "usa_ppp_2020_1km_Aggregated_UNadj.tif",
    "Chile population, 2020, people, resolution 1 km": "chl_ppp_2020_1km_Aggregated_UNadj.tif",
    "Angola population, 2020, people, resolution 1 km": "ago_ppp_2020_1km_Aggregated_UNadj.tif",
    "Peru, Bolivia, Argentina, Chile flood, February 2018": "DFO_4569_From_20180201_to_20180221.tif",
    "Peru population, 2018, 1 km resolution": "per_ppp_2018_1km_Aggregated_UNadj.tif",
    "Brazil population, 2018, 1 km resolution": "bra_ppp_2018_1km_Aggregated_UNadj.tif",
    "Algeria population density per 1 km 2020, 1 km resolution": "dza_pd_2020_1km_UNadj.tif"    
}

COLORMAPS = {
    "Population": ["YlOrBr", "Oranges", "Reds"],
    "Forest": ["Greens", "YlGn"],
    "Agriculture": ["YlGnBu", "BuGn", "PuBuGn"],
    "Economics": ["Purples", "YlOrRd", "Greys", "OrRd"],
    "Water": ["Blues", "GnBu"],
    "Hazards": ["PuRd", "RdPu"],
    "Environment": ["PuBu", "BuPu"],
    "Population divergent": ["bwr"],
    "Forest divergent": ["BrBG"],
    "Agriculture divergent": ["Spectral"],
    "Economics divergent": ["PRGn"],
    "Water divergent": ["RdYlBu"],
    "Environment divergent": ["PiYG"],
    "Hazards divergent": ["RdGy"],
    "Qualitative any topic": ["tab20c", "Set3"],
}


# Define the state type
class State(TypedDict):
    """Type definition for the graph state"""
    data_store: Dict[str, Any]  # Store for both DataFrames and GeoDataFrames
    image_store: List[Dict[str, Any]] # Store for images to save to the coversation history
    html_store: List[Dict[str, Any]] # Store for html strings to save to the coversation history 
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
    visualize: bool

# Tool function to load statistical data: load_data
def load_data(
    dataset: Annotated[
        Literal[tuple(DATA_CATALOG.keys())],
        "Name of the dataset with statistical data to load. Must be one of: "
        + ", ".join(f"'{k}'" for k in DATA_CATALOG.keys()),
    ],
    output_dataframe_name: Annotated[str, "Name of the new DataFrame to create"],
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Load statistical data from the catalog into a Pandas DataFrame.

    Args:
        dataset: Must be one of the available datasets in DATA_CATALOG
        output_dataframe_name: Name for storing the DataFrame in memory
        state: Current graph state containing dataframes and messages

    Returns:
        State: Status message with DataFrame information
    """
    file_name = DATA_CATALOG[dataset]
    catalog_path = DATA_CATALOG_PATH

    if ".csv" in file_name:
        file_path = os.path.join(catalog_path, file_name)
        stat_dataframe = pd.read_csv(file_path)
    else:
        file_path = os.path.join(catalog_path, file_name + ".zip")
        zipArchive = zipfile.ZipFile(file_path)
        stat_dataframe = pd.read_csv(zipArchive.open(file_name + ".csv"), skiprows=4)

    # Update state with new dataframe
    if "data_store" not in state:
        state["data_store"] = {}
    state["data_store"][output_dataframe_name] = stat_dataframe

    # Add message to state
    info_string, non_empty_info = get_dataframe_info(stat_dataframe)
    result = (
        f"Loaded data and created DataFrame named {output_dataframe_name}.\n"
        f"Description of DataFrame stored in {output_dataframe_name} variable:\n{info_string}\n"
        f"Amount of non-empty values in this dataframe:\n{non_empty_info}"
    )

    return result


# Tool function load_geodata to load data with geometries
def load_geodata(
    geodataset: Annotated[
        Literal[tuple(GEO_CATALOG.keys())],
        "Name of the geodataset with geospatial data to load. Must be one of: "
        + ", ".join(f"'{k}'" for k in GEO_CATALOG.keys()),
    ],
    output_geodataframe_name: Annotated[str, "Name of the new GeoDataFrame to create"],
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Loads vector geospatial data from the catalog into a GeoPandas GeoDataFrame.

    Args:
        geodataset: Must be one of the available geodatasets in GEO_CATALOG
        output_geodataframe_name: Name for storing the GeoDataFrame in memory
        state: Current graph state containing dataframes and messages

    Returns:
        str: Status message with GeoDataFrame information
    """
    file_name = GEO_CATALOG[geodataset]
    geo_catalog_path = GEO_CATALOG_PATH
    geo_path = os.path.join(geo_catalog_path, file_name)
    geodataframe = gpd.read_file(geo_path)

    # Update state with new geodataframe
    if "data_store" not in state:
        state["data_store"] = {}
    state["data_store"][output_geodataframe_name] = geodataframe

    # Add message to state
    info_string, non_empty_info = get_dataframe_info(geodataframe)
    result = (
        f"Loaded geodata and created GeoDataFrame named {output_geodataframe_name}.\n"
        f"Description of GeoDataFrame:\n{info_string}\n"
        f"Amount of non-empty values in this geodataframe:\n{non_empty_info}"
    )

    return result

# Tool function to get path to raster
def get_raster_path(
    rasterdataset: Annotated[
        Literal[tuple(RASTER_CATALOG.keys())],
        "Name of the raster dataset to load. Must be one of: "
        + ", ".join(f"'{k}'" for k in RASTER_CATALOG.keys()),
    ],
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Construct path to raster data from the catalog, handling both regular GeoTIFF and zipped files.

    Args:
        rasterdataset: Must be one of the available raster datasets in GEO_CATALOG
        state: Current graph state containing dataframes and messages

    Returns:
        str: Path to the raster file, formatted for either direct GeoTIFF access or zip archive access
    """
    file_name = RASTER_CATALOG[rasterdataset]
    raster_catalog_path = GEO_CATALOG_PATH
    
    # Convert paths to POSIX format for consistency
    data_folder = str(Path(raster_catalog_path).as_posix())
    file_name = str(Path(file_name).as_posix())
    
    # Check if the file is zipped
    if file_name.endswith('.zip'):
        # Assuming the TIF file inside the ZIP has the same name but with .tif extension
        tif_name = file_name.replace('.zip', '.tif')
        raster_path = f"zip://{data_folder}/{file_name}!/{tif_name}"
    else:
        # Regular GeoTIFF file
        raster_path = os.path.join(data_folder, file_name)
        raster_path = str(Path(raster_path).as_posix())
       
    return f"Constructed path to raster dataset '{rasterdataset}': {raster_path}"

# Tool function to get raster properties
def get_raster_description(
    raster_path: Annotated[str, "Path to the raster file"],
    state: Annotated[dict, InjectedState],
    output_variable_name: Annotated[str, "Name for storing the analysis results"] = None
) -> str:
    """
    Get description of a raster dataset including metadata and basic statistics.
    
    Args:
        raster_path: Path to the raster file to analyze
        state: Current graph state containing analysis results
        output_variable_name: Optional name for storing the analysis results in state
        
    Returns:
        str: Formatted description of raster properties and statistics
    """
    try:
        with rasterio.open(raster_path) as src:
            # Get basic metadata
            metadata = {
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,  # number of bands
                'crs': str(src.crs),
                'transform': src.transform.to_gdal(),
                'nodata': src.nodata,
                'dtype': str(src.dtypes[0])
            }
            
            # Calculate statistics for each band
            stats = []
            for band in range(1, src.count + 1):
                band_data = src.read(band)
                
                # Create mask for valid data (not nodata)
                if src.nodata is not None:
                    mask = band_data != src.nodata
                else:
                    mask = np.ones_like(band_data, dtype=bool)
                
                valid_data = band_data[mask]
                
                if len(valid_data) > 0:
                    band_stats = {
                        'band': band,
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'nan_count': int(np.sum(~np.isfinite(valid_data))),
                        'valid_pixels': int(np.sum(mask))
                    }
                else:
                    band_stats = {
                        'band': band,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'std': None,
                        'nan_count': None,
                        'valid_pixels': 0
                    }
                stats.append(band_stats)
            
            # Store results if output variable name provided
            result = {
                'metadata': metadata,
                'statistics': stats
            }
            
            if output_variable_name:
                if "data_store" not in state:
                    state["data_store"] = {}
                state["data_store"][output_variable_name] = result
            
            # Format description string
            description = [
                f"Raster Dataset: {os.path.basename(raster_path)}",
                f"\nMetadata:",
                f"- Dimensions: {metadata['width']} x {metadata['height']} pixels",
                f"- Number of bands: {metadata['count']}",
                f"- Coordinate System: {metadata['crs']}",
                f"- Data type: {metadata['dtype']}",
                f"- NoData value: {metadata['nodata']}",
                "\nBand Statistics:"
            ]
            
            for band_stats in stats:
                description.extend([
                    f"\nBand {band_stats['band']}:",
                    f"- Valid pixels: {band_stats['valid_pixels']:,}",
                    f"- Range: {band_stats['min']:.6g} to {band_stats['max']:.6g}" if band_stats['min'] is not None else "- Range: No valid data",
                    f"- Mean: {band_stats['mean']:.6g}" if band_stats['mean'] is not None else "- Mean: No valid data",
                    f"- Standard deviation: {band_stats['std']:.6g}" if band_stats['std'] is not None else "- Standard deviation: No valid data",
                    f"- NaN count: {band_stats['nan_count']:,}" if band_stats['nan_count'] is not None else "- NaN count: No valid data"
                ])
            
            if output_variable_name:
                description.append(f"\nFull results stored in variable: '{output_variable_name}'")
            
            return "\n".join(description)
            
    except rasterio.errors.RasterioIOError as e:
        return f"Error opening raster file: {type(e).__name__} : {str(e)}"  
    except Exception as e:
        return f"Unexpected error: {type(e).__name__} : {str(e)}"

#Tool function to overlap rasters and get statistics for the overlapping areas.
def analyze_raster_overlap(
    raster1_path: Annotated[str, "Path to first raster file"],
    raster2_path: Annotated[str, "Path to second raster file"],
    output_variable_name: Annotated[str, "Name for storing the analysis results"],
    state: Annotated[dict, InjectedState],
    resampling_method1: Annotated[str, "Resampling method for first raster (max/min/sum)"] = "max",
    resampling_method2: Annotated[str, "Resampling method for second raster (max/min/sum)"] = "sum",    
    plot_result: Annotated[bool, "Whether to display the result"] = True,
) -> str:
    """
    Analyze overlap between two rasters (raster 1 and raster 2) and calculate statistics for overlapping pixels from raster 2.
    
    Args:
        raster1_path: Path to the first raster file (e.g., flood extent)
        raster2_path: Path to the second raster file (e.g., population)
        output_variable_name: Name for storing the analysis results in state
        resampling_method1: Resampling method for first raster
        resampling_method2: Resampling method for second raster
        state: Current graph state containing analysis results and messages
        plot_result: Whether to display the visualization

    Returns:
        str: Status message with analysis results and statistics
    """
    result = {
        'overlap_exists': False,
        'total_value': 0,
        'statistics': {},
        'error': None
    }
    
    try:
        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []
        
        # Open both rasters
        with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
            # Check CRS
            if src1.crs != src2.crs:
                result['error'] = "Warning: Input rasters have different coordinate systems!"
                return result

            # Calculate overlap bounds
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            overlap_bounds = (
                max(bounds1.left, bounds2.left),
                max(bounds1.bottom, bounds2.bottom),
                min(bounds1.right, bounds2.right),
                min(bounds1.top, bounds2.top)
            )
            
            # Check for overlap
            if not (overlap_bounds[0] < overlap_bounds[2] and overlap_bounds[1] < overlap_bounds[3]):
                result['error'] = "No overlap between the rasters!"
                return result
            
            result['overlap_exists'] = True
            
            # Calculate dimensions and transform
            target_resolution = src1.res[0]
            width = int((overlap_bounds[2] - overlap_bounds[0]) / target_resolution)
            height = int((overlap_bounds[3] - overlap_bounds[1]) / target_resolution)
            target_transform = rasterio.transform.from_bounds(*overlap_bounds, width, height)
            
            # Initialize arrays
            data_1 = np.zeros((height, width), dtype=src1.dtypes[0])
            data_2 = np.zeros((height, width), dtype=src2.dtypes[0])
            
            # Reproject data
            resampling_dict = {
                'max': Resampling.max,
                'min': Resampling.min,
                'sum': Resampling.sum
            }
            
            reproject(
                source=rasterio.band(src1, 1),
                destination=data_1,
                src_transform=src1.transform,
                src_crs=src1.crs,
                dst_transform=target_transform,
                dst_crs=src2.crs,
                resampling=resampling_dict[resampling_method1]
            )
            
            reproject(
                source=rasterio.band(src2, 1),
                destination=data_2,
                src_transform=src2.transform,
                src_crs=src2.crs,
                dst_transform=target_transform,
                dst_crs=src2.crs,
                resampling=resampling_dict[resampling_method2]
            )
            
            # Create and apply mask
            mask = (data_1 > 0) & (data_1 != src1.nodata) & (data_2 != src2.nodata)
            masked_values = data_2[mask]
            
            # Calculate statistics
            if len(masked_values) > 0:
                result['total_value'] = float(np.sum(masked_values))
                result['statistics'] = {
                    'min': float(np.min(masked_values)),
                    'max': float(np.max(masked_values)),
                    'mean': float(np.mean(masked_values)),
                    'std': float(np.std(masked_values)),
                    'count': int(len(masked_values))
                }
            
            # Prepare output raster
            masked_output = data_2.copy()
            masked_output[~mask] = src2.nodata
            
            # Save output
            output_profile = src2.profile.copy()
            output_profile.update({
                'height': masked_output.shape[0],
                'width': masked_output.shape[1],
                'transform': target_transform
            })
                       
            # Plotting the resulting raster if needed 
            if plot_result:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot masked raster
                masked_data = masked_output
                im = ax.imshow(masked_data, cmap='viridis')
                               
                plt.colorbar(im, ax=ax, label='Value')
                plt.title(f'Overlap Analysis: {os.path.basename(raster2_path)} values where {os.path.basename(raster1_path)} > 0')
                
                # Capture the figure as base64 before showing it
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Store the image in the state with metadata
                state["image_store"].append({
                    "type": "map",
                    "description": f"Visualized overlap between {os.path.basename(raster2_path)} and {os.path.basename(raster1_path)}",
                    "base64": img_base64
                })    

                if "visualize" in state and state['visualize']:
                    plt.show()

            # Store results in state
            if "data_store" not in state:
                state["data_store"] = {}
            state["data_store"][output_variable_name] = result
            
            # Prepare return message
            if result['error']:
                return result['error']
            
            if not result['overlap_exists']:
                return "No overlap between the rasters!"
            
            stats = result['statistics']
            result_description = (
                f"Analysis completed successfully.\n"
                f"Total {os.path.basename(raster2_path)} value in areas where {os.path.basename(raster1_path)} > 0: {result['total_value']:,.2f}\n"
                f"Statistics for overlapping pixels:\n"
                f"- Count: {stats['count']:,}\n"
                f"- Mean: {stats['mean']:,.2f}\n"
                f"- Min: {stats['min']:,.2f}\n"
                f"- Max: {stats['max']:,.2f}\n"
                f"- Standard deviation: {stats['std']:,.2f}\n"
                f"Results have been stored in variable '{output_variable_name}'"
            )
            
            return result_description
            
    except rasterio.errors.RasterioIOError as e:
        return f"Error opening raster files: {type(e).__name__} : {str(e)}"
    except Exception as e:
        return f"Unexpected error: {type(e).__name__} : {str(e)}"

#Tool function to get values from raster within certain vector geometrires 
def get_values_from_raster_with_geometries(
    raster_path: Annotated[str, "Path to the raster file to be masked and get values from"],
    geodataframe_name: Annotated[str, "Name of GeoDataFrame with geometries in data_store"],
    output_variable_name: Annotated[str, "Name for storing the analysis results"],
    state: Annotated[dict, InjectedState],
    plot_result: Annotated[bool, "Whether to display the result"] = True,    
) -> str:
    """
    Mask a raster using vector geometries from a GeoDataFrame and calculate statistics for the masked area of the raster.
    
    Args:
        raster_path: Path to the raster file to be masked
        geodataframe_name: Name of GeoDataFrame containing masking geometries
        output_variable_name: Name for storing the analysis results in state
        state: Current graph state containing geodataframes and analysis results
        plot_result: Whether to display the visualization

    Returns:
        str: Status message with analysis results and statistics
    """
    result = {
        'masked_data': None,
        'total_value': 0,
        'statistics': {},
        'error': None
    }
    
    try:
        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []

        # Get geometries from the GeoDataFrame
        geodataframe = state["data_store"].get(geodataframe_name)        
        if geodataframe is None:
            return f"Error: GeoDataFrame '{geodataframe_name}' not found in data store"
        
        # Extract shapes for masking
        shapes = [geom for geom in geodataframe.geometry]

        if not shapes:
            return "Error: No valid geometries found in GeoDataFrame"
        
        # Open and mask the raster
        with rasterio.open(raster_path) as src:
            # Perform the masking operation
            out_image, out_transform = mask(src, shapes, crop=True)
            
            # Store the metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Calculate statistics for non-nodata values
            valid_data = out_image[out_image != src.nodata]

            if len(valid_data) == 0:
                return f"No valid values found in raster after masking with geometries from {geodataframe_name}"            
            elif len(valid_data) > 0:
                result['total_value'] = float(np.sum(valid_data))
                result['statistics'] = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'count': int(len(valid_data))
                }
            
            # Store masked data and metadata
            result['masked_data'] = {
                'data': out_image,
                'metadata': out_meta
            }
            
            # Plotting the resulting raster if needed 
            if plot_result:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot masked raster
                masked_data = result['masked_data']['data'][0]  # Get first band
                im = ax.imshow(masked_data, cmap='viridis')
                
                # Plot masking geometries outline
                geodataframe.boundary.plot(ax=ax, color='red', linewidth=1)
                
                plt.colorbar(im, ax=ax, label='Value')
                plt.title(f'Masked Raster with Total Value: {result["total_value"]:,.2f}')

                # Capture the figure as base64 before showing it
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # Store the image in the state with metadata
                state["image_store"].append({
                    "type": "map",
                    "description": f"Visualized Masked Raster {os.path.basename(raster_path)} ",
                    "base64": img_base64
                }) 

                if "visualize" in state and state['visualize']:
                    plt.show()

            # Store results in state
            if "data_store" not in state:
                state["data_store"] = {}
            state["data_store"][output_variable_name] = result
            
            # Prepare return message
            if result['error']:
                return result['error']
            
            stats = result['statistics']
            result_description = (
                f"Raster masking completed successfully.\n"
                f"Total value in masked area: {result['total_value']:,.2f}\n"
                f"Statistics for masked pixels:\n"
                f"- Count: {stats['count']:,}\n"
                f"- Mean: {stats['mean']:,.2f}\n"
                f"- Min: {stats['min']:,.2f}\n"
                f"- Max: {stats['max']:,.2f}\n"
                f"- Standard deviation: {stats['std']:,.2f}\n"
                f"Results have been stored in variable '{output_variable_name}'"
            )
            
            return result_description
            
    except rasterio.errors.RasterioIOError as e:
        return f"Error opening raster file: {type(e).__name__} : {str(e)}"
    except Exception as e:
        return f"Unexpected error: {type(e).__name__} : {str(e)}"

# Create tool function to merge 2 dataframes, or a dataframe and a geodataframe
def merge_dataframes(
    dataframe_name: Annotated[str, "Name of DataFrame containing statistical data"],
    geodataframe_name: Annotated[str, "Name of GeoDataFrame containing spatial data"],
    statkey: Annotated[str, "Column name in statistical data for joining"],
    geokey: Annotated[str, "Column name in spatial data for joining"],
    output_dataframe_name: Annotated[str, "Name of the merged DataFrame to create"],
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Merge statistical and geospatial dataframes using specified key columns. The resulting merged dataframe will preserve all rows from geodataset, matching with dataset where possible and filling with NaN where no match exists. 

    Args:
        dataframe_name: Name of DataFrame in memory with statistical data
        geodataframe_name: Name of GeoDataFrame in memory with spatial data
        statkey: Column name in statistical data for joining
        geokey: Column name in spatial data for joining
        output_dataframe_name: Name for storing the merged DataFrame
        state: Current graph state containing dataframes and messages

    Returns:
        str: Status message with merge details and dataframe information
    """
    if "data_store" not in state:
        state["data_store"] = {}
        
    dataset = state["data_store"].get(dataframe_name)
    geodataset = state["data_store"].get(geodataframe_name)

    merged_data = geodataset.merge(dataset, left_on=geokey, right_on=statkey, how='left')
    state["data_store"][output_dataframe_name] = merged_data
    
    info_string, non_empty_info = get_dataframe_info(merged_data)
    result = (
        f"Created merged DataFrame named {output_dataframe_name}.\n"
        f"Description of merged DataFrame:\n{info_string}\n"
        f"Amount of non-empty values in this geodataframe:\n{non_empty_info}"
    )
    
    return result

# Create tool function to get unique values in a column of a dataframe for future filters: get_unique_values 
def get_unique_values(
    dataframe_name: Annotated[str, "Name of DataFrame/GeoDataFrame to analyze"],
    column: Annotated[str, "Column name to get unique values from"],
    state: Annotated[dict, InjectedState],
    ) -> str:
    """
    Get unique values from a specified column in a DataFrame/GeoDataFrame.

    Args:
        dataframe_name: Name of DataFrame/GeoDataFrame in memory 
        column: Column name to analyze
        state: Current graph state containing dataframes

    Returns:
        str: List of unique values from a specified column in a DataFrame/GeoDataFrame.
    """
    if "data_store" not in state:
        state["data_store"] = {}
        
    df = state["data_store"].get(dataframe_name)
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
        
    unique_values = df[column].unique().tolist()
    
    result = f"Unique values in column '{column}' of {dataframe_name}:\n{unique_values}"
    return result

# Create a tool function to filter dataframe by a categorical attribute
def filter_categorical(
dataframe_name: Annotated[str, "Name of DataFrame/GeoDataFrame to filter"],
filters: Annotated[str, '''Dictionary of column names and values to filter by or JSON-string representation of this dictionary, for example {"Country": ["Algeria", "Angola"]}'''], # Tool edit to accomodate Gemini's indigestion of complex arguments
# filters: Annotated[Dict[str, Union[str, List[str]] | str], "Dictionary of column names and values to filter by or string representation of this dictionary, for example {'Country': ['Algeria', 'Angola']}"],
output_dataframe_name: Annotated[str, "Name of filtered DataFrame to create"],
state: Annotated[dict, InjectedState],
) -> str:
    """
    Filter DataFrame/GeoDataFrame by categorical values in specified columns.

    Args:
        dataframe_name: Name of DataFrame/GeoDataFrame in memory
        filters: Dict with column names as keys and filter values as values
        output_dataframe_name: Name for storing filtered DataFrame
        state: Current graph state containing dataframes

    Returns:
        str: Filter results and filtered dataframe info
    """
    # To accomodate for Gemini's indigestion of objects as function arguments
    if isinstance(filters, str):
        filters = ast.literal_eval(filters)

    if "data_store" not in state:
        state["data_store"] = {}

    df = state["data_store"].get(dataframe_name)

    if not filters:
        filtered_df = df.copy()
    else:
        filtered_df = df.copy()
        for column, values in filters.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataframe")
                
            if isinstance(values, (list, tuple, set)):
                filtered_df = filtered_df[filtered_df[column].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[column] == values]

    state["data_store"][output_dataframe_name] = filtered_df

    result = (
        f"Created filtered DataFrame named {output_dataframe_name} from DataFrame {dataframe_name}.\n"
        f"Filters applied: {filters}\n" 
        f"Columns in the new dataframe  {output_dataframe_name} are the same as in the original DataFrame {dataframe_name}.\n"
        f"Number of rows in the new {output_dataframe_name} is {len(filtered_df)}, different from the original {len(df)}."
    )

    return result

# Create tool function to filter dataframe/geodataframe by numerical values
def filter_numerical(
   dataframe_name: Annotated[str, "Name of DataFrame/GeoDataFrame to filter"],
   conditions: Annotated[str, "Query string with numerical conditions"],
   output_dataframe_name: Annotated[str, "Name of filtered DataFrame to create"],
   state: Annotated[dict, InjectedState],
) -> str:
    """
    Filter DataFrame/GeoDataFrame using numerical conditions via query method.

    Args:
        dataframe_name: Name of DataFrame/GeoDataFrame in memory
        conditions: Query string (e.g. "col1 > 25 and col2 < 100")
        output_dataframe_name: Name for storing filtered DataFrame
        state: Current graph state containing dataframes

    Returns:
        str: Filter results and filtered dataframe info
    """
    
    try:
        
        if "data_store" not in state:
            state["data_store"] = {}

        df = state["data_store"].get(dataframe_name)

        filtered_df = df.copy()
        filtered_df = filtered_df.query(conditions)
        
        # Store result
        state["data_store"][output_dataframe_name] = filtered_df
        state["data_store"][output_dataframe_name] = filtered_df

        result = (
            f"Created filtered DataFrame named {output_dataframe_name} from DataFrame {dataframe_name}.\n"
            f"Filters applied: {conditions}\n"
            f"Columns in the new dataframe  {output_dataframe_name} are the same as in the original DataFrame {dataframe_name}.\n"
            f"Number of rows in the new {output_dataframe_name} is {len(filtered_df)}, different from the original {len(df)}."
        )
        return result

    except Exception as e:
        return f"Error applying filter: {str(e)}"
    
# Create tool function to identify statistics on a numerical column
def calculate_column_statistics(       
    dataframe_name: Annotated[str, "Name of DataFrame/GeoDataFrame in data_store"],
    column_name: Annotated[str, "Name of column to analyze"],
    output_variable_name: Annotated[str, "Name for storing the statistics results"],
    state: Annotated[dict, InjectedState],
    include_quantiles: Annotated[bool, "Whether to include quantiles (25th, 50th, 75th percentiles)"] = True,
    additional_quantiles: Annotated[str | None, '''Optional additional quantiles to calculate (values between 0 and 1) in JSON-string representation, example, [0.95]''']= None    
    # additional_quantiles: Annotated[List[float]| None, "Optional additional quantiles to calculate (values between 0 and 1), if none required pass empty list"]= None    
) -> str:
    """
    Calculate summary statistics for a numerical column in a DataFrame.

    Args:
        dataframe_name: Name of DataFrame/GeoDataFrame in data store
        column_name: Name of the column to analyze
        output_variable_name: Name for storing the statistics results
        state: Current graph state containing dataframes
        include_quantiles: Whether to include standard quantile (True by default)
        additional_quantiles: Optional list of additional quantiles to calculate

    Returns:
        str: Summary of calculated statistics
    """
    try:
        # To accomodate for Gemini's indigestion of objects as function arguments
        if isinstance(additional_quantiles, str):
            additional_quantiles = ast.literal_eval(additional_quantiles)

        # Get DataFrame from state
        df = state["data_store"].get(dataframe_name)
        if df is None:
            return f"Error: DataFrame '{dataframe_name}' not found in data store"

        # Check if column exists
        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in DataFrame"

        # Extract column data
        column_data = df[column_name]

        # Calculate basic statistics
        stats = {
            'summary': {
                'count': len(column_data),
                'valid_count': column_data.count(),
                'missing_count': column_data.isnull().sum(),
                'min': float(column_data.min()) if not column_data.empty else None,
                'max': float(column_data.max()) if not column_data.empty else None,
                'mean': float(column_data.mean()) if not column_data.empty else None,
                'std': float(column_data.std()) if not column_data.empty else None
            }
        }

        # Add quantiles if requested
        if include_quantiles:
            stats['quantiles'] = {
                'q25': float(column_data.quantile(0.25)),
                'q50': float(column_data.quantile(0.50)),  # median
                'q75': float(column_data.quantile(0.75))
            }

        # Add any additional quantiles
        if additional_quantiles:
            stats['additional_quantiles'] = {
                f'q{int(q*100)}': float(column_data.quantile(q))
                for q in additional_quantiles
            }

        # Store results in state
        if "data_store" not in state:
            state["data_store"] = {}
        state["data_store"][output_variable_name] = stats

        # Format results message
        message_parts = [
            f"Statistics for column '{column_name}' in {dataframe_name}:",
            "\nBasic Statistics:",
            f"- Count: {stats['summary']['count']:,} (Valid: {stats['summary']['valid_count']:,}, Missing: {stats['summary']['missing_count']:,})",
            f"- Range: {stats['summary']['min']:,.2f} to {stats['summary']['max']:,.2f}",
            f"- Mean: {stats['summary']['mean']:,.2f}",
            f"- Standard Deviation: {stats['summary']['std']:,.2f}"
        ]

        # Add quartiles to message if included
        if include_quantiles:
            message_parts.extend([
                "\nQuantile:",
                f"- 25th percentile (Q1): {stats['quantiles']['q25']:,.2f}",
                f"- 50th percentile (Q2/median): {stats['quantiles']['q50']:,.2f}",
                f"- 75th percentile (Q3): {stats['quantiles']['q75']:,.2f}"
            ])

        # Add additional quantiles to message if calculated
        if additional_quantiles:
            message_parts.append("\nAdditional Quantiles:")
            for q in additional_quantiles:
                q_key = f'q{int(q*100)}'
                message_parts.append(f"- {int(q*100)}th percentile: {stats['additional_quantiles'][q_key]:,.2f}")

        message_parts.append(f"\nResults stored in variable '{output_variable_name}'")

        return "\n".join(message_parts)

    except Exception as e:
        return f"Error calculating statistics: {type(e).__name__} : {str(e)}"    

# Create tool funtion to make buffers arounf geometries in a GeoDataFrame
def create_buffer(
    geodataframe_name: Annotated[str, "Name of GeoDataFrame containing spatial data"],
    buffer_size: Annotated[int, "Size of the buffer in meters"],
    output_geodataframe_name: Annotated[str, "Name of the output GeoDataFrame with buffers"],
    state: Annotated[dict, InjectedState],
) -> str:  
    """
    Create buffer zones around geometries in a GeoDataFrame using specified buffer size.

    Args:
        geodataframe_name: Name of the source GeoDataFrame stored in data_store
        buffer_size: Size of the buffer zone in meters
        output_geodataframe_name: Name for storing the resulting GeoDataFrame with buffers
        state: Current graph state containing dataframes and messages

    Returns:
        str: Status message with merge details and dataframe information
    """
    
    geodataframe = state["data_store"].get(geodataframe_name)

    if geodataframe is None:
        return f"Error: GeoDataFrame '{geodataframe_name}' not found in data store"
    
    # Store initial CRS
    crs_init = geodataframe.crs.to_string()
    
    # Reproject to Web Mercator for buffer operation in meters
    geodataframe_reproj = geodataframe.to_crs("EPSG:3857")
    
    # Create buffer
    geodata_buffer = geodataframe_reproj.copy()
    geodata_buffer.geometry = geodataframe_reproj.geometry.buffer(buffer_size)
    
    # Reproject back to original CRS
    geodata_buffer_reproj = geodata_buffer.to_crs(crs_init)
    
    # Store result in data store
    state["data_store"][output_geodataframe_name] = geodata_buffer_reproj
    
    # Generate result message
    info_string, _ = get_dataframe_info(geodata_buffer_reproj)
    result = f"Created {buffer_size}m buffer and stored result in GeoDataFrame '{output_geodataframe_name}'.\nDescription of GeoDataFrame:\n{info_string}"

    return result

# Create make_choropleth_map tool function to plot a geodataframe
def make_choropleth_map(
    dataframe_name: Annotated[str, "Name of GeoDataFrame containing map data"],
    mappingkey: Annotated[str, "Column name to visualize on map"],
    legendtext: Annotated[str, "Title of the legend"],
    colormap: Annotated[
        Literal[tuple(COLORMAPS.keys())],
        "Color scheme for the choropleth map. Must be one of: "
        + ", ".join(f"'{k}'" for k in COLORMAPS.keys()),
    ],
   state: Annotated[dict, InjectedState],    
) -> str:
    """
    Create a choropleth map visualization from a GeoDataFrame.

    Args:
        dataframe_name: Name of GeoDataFrame in memory
        mappingkey: Column to visualize
        legendtext: Legend title
        colormap: Must be one of the available colormap types in COLORMAPS dictionary        
        state: Current graph state containing dataframes    

    Returns:
        str: Map generation status
    """
    if "data_store" not in state:
        state["data_store"] = {}

    # Initialize image store if needed
    if "image_store" not in state:
        state["image_store"] = []        

    geodataframe = state["data_store"].get(dataframe_name)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    max_value = geodataframe[mappingkey].max()
    legend_label = legendtext + ', million' if max_value > 1e6 else legendtext
    formatter = FuncFormatter(lambda x, _: f"{x / 1e6:.1f}" if max_value > 1e6 else f"{x:.1f}")

    geodataframe.plot(
        column=mappingkey,
        cmap=random.choice(COLORMAPS[colormap]),
        legend=True,
        legend_kwds={
            'fmt': '{:.1f}',
            'label': legend_label,
            'orientation': "horizontal",
            'shrink': 0.6
        },
        ax=ax
    )  

    ax.set_xlim(geodataframe.total_bounds[0], geodataframe.total_bounds[2])
    ax.set_ylim(geodataframe.total_bounds[1], geodataframe.total_bounds[3])

    cbar = ax.get_figure().axes[-1]
    cbar.xaxis.set_major_formatter(formatter)

    # Capture the figure as base64 before showing it
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Store the image in the state with metadata
    state["image_store"].append({
        "type": "map",
        "description": f"Generated map visualizing {mappingkey} with {legendtext} legend",
        "base64": img_base64
    }) 

    if "visualize" in state and state['visualize']:
        plt.show()

    result = f"Generated map visualizing {mappingkey} with {legendtext} legend"
    return result

# Create tool function to select points from a GeoDataFrame that overlaps with pixels with needed values in a raster
def filter_points_by_raster_values(
    raster_path: Annotated[str, "Path to the raster file"],
    points_geodataframe_name: Annotated[str, "Name of GeoDataFrame with points in data_store"],
    value_column: Annotated[str, "Name of the column to store raster values"],
    output_geodataframe_name: Annotated[str, "Name for storing the filtered GeoDataFrame"],
    filter_type: Annotated[
        Literal["less", "greater", "between"],
        "Type of threshold filter: 'less' for < threshold1, 'greater' for > threshold1, 'between' for threshold1 <= x <= threshold2"
    ],
    threshold1: Annotated[float, "First threshold value"],
    state: Annotated[dict, InjectedState],     
    threshold2: Annotated[float | None, "Second threshold value (only for 'between' filter type)"] = None,
    plot_result: Annotated[bool, "Whether to display the result"] = True,           
) -> str:
    """
    Sample raster values at point locations and filter points based on threshold conditions.
    
    Args:
        raster_path: Path to the raster file to sample values from
        points_geodataframe_name: Name of GeoDataFrame containing points
        value_column: Name of the column to store sampled raster values
        output_geodataframe_name: Name for storing the filtered GeoDataFrame
        filter_type: Type of threshold filter to apply
        threshold1: Main threshold value
        threshold2: Optional second threshold for 'between' filter
        plot_result: Whether to display the visualization
        state: Current graph state containing geodataframes and results

    Returns:
        str: Status message with filtering results and statistics
    """
    try:
        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []

        # Get points GeoDataFrame from state
        points_gdf = state["data_store"].get(points_geodataframe_name)
        if points_gdf is None:
            return f"Error: GeoDataFrame '{points_geodataframe_name}' not found in data store"

        # Open and process raster
        with rasterio.open(raster_path) as src:
            # Reproject points if needed
            if points_gdf.crs != src.crs:
                points_gdf = points_gdf.to_crs(src.crs)

            # Sample raster values at point locations
            point_values = [
                next(src.sample([(x, y)]))
                for x, y in zip(points_gdf.geometry.x, points_gdf.geometry.y)
            ]
            
            # Store raster values in the GeoDataFrame
            points_gdf[value_column] = [v[0] for v in point_values]
            
            # Apply filtering based on filter_type
            if filter_type == "less":
                filtered_gdf = points_gdf[points_gdf[value_column] < threshold1]
                condition_desc = f"less than {threshold1}"
            elif filter_type == "greater":
                filtered_gdf = points_gdf[points_gdf[value_column] > threshold1]
                condition_desc = f"greater than {threshold1}"
            elif filter_type == "between":
                if threshold2 is None:
                    return "Error: threshold2 must be provided for 'between' filter type"
                filtered_gdf = points_gdf[
                    (points_gdf[value_column] >= threshold1) & 
                    (points_gdf[value_column] <= threshold2)
                ]
                condition_desc = f"between {threshold1} and {threshold2}"
            else:
                return f"Error: Invalid filter_type '{filter_type}'"

            if len(filtered_gdf) == 0:
                return f"No points found matching condition: {value_column} {condition_desc}"

            # Store results in state
            if "data_store" not in state:
                state["data_store"] = {}
            state["data_store"][output_geodataframe_name] = filtered_gdf

            if plot_result:
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot raster
                with rasterio.open(raster_path) as src:
                    raster_data = src.read(1)
                    im = ax.imshow(raster_data, cmap='viridis')
                    
                    # Plot points
                    filtered_gdf.plot(ax=ax, color='red', markersize=50, marker='.')
                    
                    # Add colorbar and title
                    plt.colorbar(im, ax=ax, label=value_column)
                    plt.title(f'Points with {value_column} {condition_desc}\n({len(filtered_gdf)} out of {len(points_gdf)} points)')
                    
                    # Capture the figure as base64 before showing it
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()

                    # Store the image in the state with metadata
                    state["image_store"].append({
                        "type": "map",
                        "description": f"Generated map of points with {value_column} {condition_desc}\n({len(filtered_gdf)} out of {len(points_gdf)} points)",
                        "base64": img_base64
                    }) 

                    # Display the plot
                    if "visualize" in state and state['visualize']:
                        plt.show()     
            
            # Prepare return message
            result_description = (
                f"Point filtering completed successfully.\n"
                f"- Input points: {len(points_gdf)}\n"
                f"- Points {condition_desc}: {len(filtered_gdf)}\n"
                f"- Column '{value_column}' contains sampled raster values\n"
                f"Filtered points have been stored in '{output_geodataframe_name}'"
            )
            
            return result_description
            
    except rasterio.errors.RasterioIOError as e:
        return f"Error opening raster file: {type(e).__name__} : {str(e)}"
    except Exception as e:
        return f"Unexpected error: {type(e).__name__} : {str(e)}"
    
# Tool function to select features by locaiton
def select_features_by_spatial_relationship(
   features_geodataframe_name: Annotated[str, "Name of GeoDataFrame with features to select from"],
   reference_geodataframe_name: Annotated[str, "Name of GeoDataFrame to select by"],
   spatial_predicates: Annotated[
       list[Literal["within", "intersects", "contains", "crosses", "touches", "overlaps"] | str],
       '''List of spatial relationships between features or JSON-string representation of this list, example ["within", "touches"]'''
   ],   
   output_geodataframe_name: Annotated[str, "Name for storing the selected features"],
   state: Annotated[dict, InjectedState],
   plot_result: Annotated[bool, "Whether to display the result"] = True,
) -> str:
   """
   Select features from one GeoDataFrame based on multiple spatial relationships with another.
   Features that satisfy ANY of the specified predicates will be selected (OR logic).
   Automatically handles CRS differences by reprojecting features to match reference CRS.

    Args:
        features_geodataframe_name: Name of GeoDataFrame containing features to select from
        reference_geodataframe_name: Name of GeoDataFrame containing reference geometries
        spatial_predicates: List of spatial predicates to check ('within', 'intersects', 'contains', etc.)
        output_geodataframe_name: Name for storing the selected features GeoDataFrame
        plot_result: Whether to display the visualization
        state: Current graph state containing geodataframes and results

    Returns:
        str: Status message with count of selected features and storage location
   """
   try:
       # Initialize image store if needed
       if "image_store" not in state:
           state["image_store"] = []       

       # Get GeoDataFrames from state
       features_gdf = state["data_store"].get(features_geodataframe_name)
       reference_gdf = state["data_store"].get(reference_geodataframe_name)

       if len(features_gdf) > 150000:
           return f"Error: the number of features in {features_gdf} is too large to process the request."
       elif len(reference_gdf) > 150000:
           return f"Error: the number of features in {reference_gdf} is too large to process the request."

       # To accomodate for Gemini's indigestion of objects as function arguments
       if isinstance(spatial_predicates, str):
           spatial_predicates = ast.literal_eval(spatial_predicates)
       
       if features_gdf is None or reference_gdf is None:
           return "Error: Input GeoDataFrames not found in data store"
       
       # Check CRS and reproject if needed
       if features_gdf.crs != reference_gdf.crs:
           # Reproject features to match reference CRS
           features_gdf = features_gdf.to_crs(reference_gdf.crs)            

       # Initialize an empty GeoDataFrame to store combined results
       selected_features = None

       # Process each predicate and combine results
       for predicate in spatial_predicates:
           # Perform spatial join for current predicate
           current_selection = gpd.sjoin(features_gdf, reference_gdf, predicate=predicate)
           
           if selected_features is None:
               selected_features = current_selection
           else:
               # Combine with previous results (OR logic)
               selected_features = pd.concat([selected_features, current_selection]).drop_duplicates()

       if selected_features is None or len(selected_features) == 0:
           return f"No features found matching any of the spatial predicates: {spatial_predicates}"

       # Store results in state      
       state["data_store"][output_geodataframe_name] = selected_features

       # Create visualization if requested
       if plot_result:
           fig, ax = plt.subplots(figsize=(12, 8))
           # Plot reference geometries in grey
           reference_gdf.plot(ax=ax, alpha=0.5, color='grey', label='Reference')
           # Plot selected features in red
           selected_features.plot(ax=ax, color='red', label='Selected')
           # Add legend and title
           ax.legend()
           plt.title(f'Features matching predicates: {", ".join(spatial_predicates)}')

           # Capture the figure as base64 before showing it
           buf = io.BytesIO()
           fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
           buf.seek(0)
           img_base64 = base64.b64encode(buf.read()).decode('utf-8')
           buf.close()

           # Store the image in the state with metadata
           state["image_store"].append({
               "type": "map",
               "description": f"Generated map of features matching predicates: {', '.join(spatial_predicates)}",
               "base64": img_base64
           })

           if "visualize" in state and state['visualize']:
               plt.show()

       # Prepare return message
       predicates_str = ", ".join(spatial_predicates)
       return (f"Selected {len(selected_features)} features that match ANY of these predicates: {predicates_str}\n"
               f"Results stored in '{output_geodataframe_name}'")
               
   except Exception as e:
       return f"Error: {type(e).__name__} : {str(e)}"
   
# Tool function to calculate length of line features
def calculate_line_lengths(
   geodataframe_name: Annotated[str, "Name of GeoDataFrame with line features"],
   output_variable_name: Annotated[str, "Name for storing results"],
   state: Annotated[dict, InjectedState]
) -> str:
    """
    Calculate lengths of line features in kilometers using appropriate UTM projection.

    Args:
    geodataframe_name: Name of the source GeoDataFrame with line geometries stored in data_store
    output_variable_name: Name for storing the calculation results and projected GeoDataFrame
    state: Current graph state containing geodataframes and results

    Returns:
    str: Status message with UTM zone details, total length, and storage location
    """
    try:
        gdf = state["data_store"].get(geodataframe_name)
        if gdf is None:
            return f"Error: GeoDataFrame '{geodataframe_name}' not found"

        # Calculate UTM zone from centroid
        centroid = gdf.geometry.union_all().centroid
        utm_zone = int(np.floor((centroid.x + 180) / 6) + 1)
        hemisphere = 'N' if centroid.y >= 0 else 'S'
        epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone

        # Project and calculate lengths
        gdf_projected = gdf.to_crs(f'EPSG:{epsg}')
        gdf_projected['length_meters'] = gdf_projected.geometry.length
        total_length_km = gdf_projected.geometry.length.sum() / 1000

        # Store results
        result = {
            'gdf_with_lengths': gdf_projected,
            'total_length_km': total_length_km,
            'utm_zone': utm_zone,
            'epsg': epsg
        }
        state["data_store"][output_variable_name] = result

        return (f"Calculated line lengths in {geodataframe_name} using UTM zone {utm_zone}{hemisphere} (EPSG:{epsg}).\n"
                f"Total length: {total_length_km:,.2f} km\n"
                f"Results stored in '{output_variable_name}'")

    except Exception as e:
        return f"Error: {type(e).__name__} : {str(e)}"   

# Tool function for operations between 2 columns in a (Geo)DataFrame    
def calculate_columns(
   dataframe1_name: Annotated[str, "Name of first DataFrame in data_store"],
   column1_name: Annotated[str, "Name of column from first DataFrame"],
   dataframe2_name: Annotated[str, "Name of second DataFrame in data_store"],
   column2_name: Annotated[str, "Name of column from second DataFrame"], 
   operation: Annotated[
       Literal["multiply", "divide", "add", "subtract"],
       "Mathematical operation to perform between columns"
   ],
   output_column_name: Annotated[str, "Name for the new column with results"],
   output_dataframe_name: Annotated[str, "Name for storing resulting DataFrame"],
   state: Annotated[dict, InjectedState]
) -> str:
   """
   Perform mathematical operations between columns of two DataFrames or columns of the same dataframe.

   Args:
       dataframe1_name: Name of first DataFrame in data_store
       column1_name: Column name from first DataFrame
       dataframe2_name: Name of second DataFrame in data_store 
       column2_name: Column name from second DataFrame
       operation: Type of operation to perform between columns
       output_column_name: Name for the new column containing results
       output_dataframe_name: Name for storing the resulting DataFrame
       state: Current graph state containing dataframes and results

   Returns:
       str: Status message with operation details and results location
   """
   try:
       df1 = state["data_store"].get(dataframe1_name)
       df2 = state["data_store"].get(dataframe2_name)

       if df1 is None or df2 is None:
           return "Error: One or both DataFrames not found in data store"

       if column1_name not in df1.columns or column2_name not in df2.columns:
           return f"Error: Column {column1_name} or {column2_name} not found"

       result_df = df1.copy()
       
       operations = {
           "multiply": lambda x, y: x * y,
           "divide": lambda x, y: x / y,
           "add": lambda x, y: x + y,
           "subtract": lambda x, y: x - y
       }

       result_df[output_column_name] = operations[operation](
           df1[column1_name], 
           df2[column2_name]
       )

       state["data_store"][output_dataframe_name] = result_df

       return (f"Performed {operation} operation between '{column1_name}' and '{column2_name}'.\n"
               f"Results stored in column '{output_column_name}' in DataFrame '{output_dataframe_name}'")

   except Exception as e:
       return f"Error: {type(e).__name__} : {str(e)}"
   
# Tool function to make operations between a column and a numerical value
def scale_column_by_value(
    dataframe_name: Annotated[str, "Name of DataFrame in data_store"],
    column_name: Annotated[str, "Name of column to scale"],
    operation: Annotated[
        Literal["multiply", "divide", "add", "subtract"],
        "Mathematical operation to perform"
    ],
    value: Annotated[float, "Numerical value for scaling"],
    output_column_name: Annotated[str, "Name for the new column with results"],
    output_dataframe_name: Annotated[str, "Name for storing resulting DataFrame"],
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Scale values in a DataFrame column by a numerical value.

    Args:
        dataframe_name: Name of DataFrame to modify
        column_name: Column to apply scaling to
        operation: Type of operation to perform
        value: Numerical value to use in operation
        output_column_name: Name for the new scaled column
        output_dataframe_name: Name for storing the resulting DataFrame
        state: Current graph state containing dataframes and results

    Returns:
        str: Status message with operation details
    """
    try:
        df = state["data_store"].get(dataframe_name)
        if df is None:
            return "Error: DataFrame not found in data store"

        if column_name not in df.columns:
            return f"Error: Column {column_name} not found"

        result_df = df.copy()
        
        operations = {
            "multiply": lambda x: x * value,
            "divide": lambda x: x / value,
            "add": lambda x: x + value,
            "subtract": lambda x: x - value
        }

        result_df[output_column_name] = operations[operation](df[column_name])
        state["data_store"][output_dataframe_name] = result_df

        return (f"Performed {operation} by {value} on column '{column_name}'.\n"
                f"Results stored in column '{output_column_name}' in DataFrame '{output_dataframe_name}'")

    except Exception as e:
        return f"Error: {type(e).__name__} : {str(e)}"

# Tool function to create heatmap    
def make_heatmap(
    geodataframe_name: Annotated[str, "Name of GeoDataFrame containing point data"],
    value_column: Annotated[str, "Column name for intensity values"],
    state: Annotated[dict, InjectedState],
    center_lat: Annotated[float, "Center latitude for the map view"] = None,
    center_lon: Annotated[float, "Center longitude for the map view"] = None,
    zoom_level: Annotated[int, "Initial zoom level (1-20)"] = None,
    radius: Annotated[int, "Radius of influence for each point"] = 10,
    map_style: Annotated[
        Literal["open-street-map", "carto-positron", "carto-darkmatter"],
        "Base map style to use"
    ] = "open-street-map",
    width: Annotated[int, "Width of the map in pixels"] = 1000,
    height: Annotated[int, "Height of the map in pixels"] = 600
) -> str:
    """
    Create an interactive heatmap from point data using Plotly's density_mapbox.

    Args:
        geodataframe_name: Name of the GeoDataFrame with point geometries
        value_column: Column containing values for heatmap intensity
        state: Current graph state containing geodataframes and results
        center_lat: Center latitude for the map view (auto-calculated if None)
        center_lon: Center longitude for the map view (auto-calculated if None)
        zoom_level: Initial zoom level (auto-calculated if None)
        radius: Radius of influence for each point in pixels
        map_style: Style of the base map
        width: Width of the map in pixels
        height: Height of the map in pixels

    Returns:
        str: Status message with visualization details
    """
    try:
        # Initialize html store if needed
        if "html_store" not in state:
            state["html_store"] = []
               
        # Get GeoDataFrame from state
        gdf = state["data_store"].get(geodataframe_name)
        if gdf is None:
            return f"Error: GeoDataFrame '{geodataframe_name}' not found in data store"

        # Validate value column
        if value_column not in gdf.columns:
            return f"Error: Column '{value_column}' not found in GeoDataFrame"

        # Extract coordinates from geometry
        lats = gdf.geometry.y
        lons = gdf.geometry.x

        # Calculate center if not provided
        if center_lat is None:
            center_lat = lats.mean()
        if center_lon is None:
            center_lon = lons.mean()
        
        # Calculate zoom if not provided
        if zoom_level is None:
            # Simple heuristic based on data spread
            lat_range = lats.max() - lats.min()
            lon_range = lons.max() - lons.min()
            max_range = max(lat_range, lon_range)
            # Calculate zoom level based on data range
            # For global views (range > 180 degrees), use zoom level 0-2
            if max_range > 180:
                zoom_level = 1
            else:
                zoom_level = int(max(0, min(10, 8 - np.log2(max_range))))

        # Create density map weighted by value column
        fig = px.density_mapbox(
            gdf,
            lat=lats,
            lon=lons,
            z=value_column,
            radius=radius,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level,
            mapbox_style=map_style
        )

        # Update layout
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        html_string = fig.to_html(include_plotlyjs='cdn', div_id=f"heatmap_{geodataframe_name}")       
      
        state["html_store"].append({
            "type": "interactive_map",
            "description": f"Interactive heatmap from '{geodataframe_name}' showing {value_column}",
            "html": html_string
        })      

        # Prepare result message
        result_parts = [
            f"Created interactive heatmap visualization from '{geodataframe_name}'.",
            f"- Intensity values from column: {value_column}",
            f"- Map centered at: ({center_lat:.2f}, {center_lon:.2f})",
            f"- Zoom level: {zoom_level}",
            f"- Base map style: {map_style}"
        ]

        # Show the plot
        if "visualize" in state and state['visualize']:
            fig.show()

        return "\n".join(result_parts)

    except ImportError as e:
        return f"Error: Required package not found - {type(e).__name__} : {str(e)}"
    except Exception as e:
        return f"Error creating heatmap: {type(e).__name__} : {str(e)}"
    
# Tool function to visualize one or more GeoDataFrames on one map
def visualize_geographies(
    geodataframe_names: Annotated[list[str] | str , '''List of GeoDataFrame names to visualize or JSON-string representation of this list, example ["geodataframe_name_1", "geodataframe_name_2"]'''],
    state: Annotated[dict, InjectedState],
    # layer_styles: Annotated[
    #     List[Dict[str, Union[float, str]]] | None,
    #     "Optional list of style dictionaries for each layer with keys: 'color', 'alpha', 'linewidth', 'label'"
    # ] = None,
    layer_styles: Annotated[str, '''Optional list of style dictionaries as JSON array, each with keys: 'color', 'alpha', 'linewidth', 'label' or JSON-string representation of this list. Example: [{"color": "lightgray", "alpha": 0.5, "label": "Counties"},{"color": "blue", "alpha": 0.7, "label": "Mississippi River"}]'''] = None, # To accomodate Gemins's indigestion of complex arguments
    basemap_style: Annotated[
        Literal["OpenStreetMap", "Carto Positron", "Carto Dark"],
        "Style of the basemap"
    ] = "OpenStreetMap",
    title: Annotated[str, "Title for the map"] = "",
    add_legend: Annotated[bool, "Whether to add a legend"] = True
) -> str:
    """
    Visualize multiple GeoDataFrames as layers over a basemap.

    Args:
        geodataframe_names: List of GeoDataFrame names stored in data_store
        layer_styles: Optional list of style dictionaries for each layer. Each dictionary can contain:
            - color: Color for the geometries
            - alpha: Transparency level (0-1)
            - linewidth: Width of geometry outlines
            - label: Label for the legend
        basemap_style: Style of the basemap
        title: Title for the map
        add_legend: Whether to add a legend
        state: Current graph state containing geodataframes

    Returns:
        str: Status message with visualization details
    """
    try:

        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []
            
        # Default style if none provided
        default_style = {
            'color': 'red',
            'alpha': 0.5,
            'linewidth': 1.0,
            'label': None
        }
        # To accomodate for Gemini's indigestion of objects as function arguments
        if isinstance(geodataframe_names, str):
            geodataframe_names = ast.literal_eval(geodataframe_names)
        # To accomodate for Gemini's indigestion of objects as function arguments
        if isinstance(layer_styles, str):
            layer_styles = ast.literal_eval(layer_styles)

        # If no styles provided, create default styles for each layer
        if layer_styles is None:
            # Generate distinct colors for multiple layers
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
            layer_styles = []
            for i, _ in enumerate(geodataframe_names):
                style = default_style.copy()
                style['color'] = colors[i % len(colors)]
                style['label'] = f'Layer {i+1}'
                layer_styles.append(style)
        
        # Validate number of styles matches number of layers
        if len(layer_styles) != len(geodataframe_names):
            return f"Error: Number of layer styles ({len(layer_styles)}) does not match number of layers ({len(geodataframe_names)})"

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Track bounds for all layers
        all_bounds = []

        # Plot each layer
        for i, gdf_name in enumerate(geodataframe_names):
            # Get GeoDataFrame from state
            geodataframe = state["data_store"].get(gdf_name)
            if geodataframe is None:
                return f"Error: GeoDataFrame '{gdf_name}' not found in data store"

            # Reproject to Web Mercator for basemap compatibility
            gdf_web_mercator = geodataframe.to_crs(epsg=3857)
            
            # Get style for this layer
            style = layer_styles[i]
            style = {**default_style, **style}  # Merge with defaults

            # Plot geometries
            gdf_web_mercator.plot(
                ax=ax,
                color=style['color'],
                alpha=style['alpha'],
                linewidth=style['linewidth'],
                label=style['label']
            )

            # Track bounds
            all_bounds.append(gdf_web_mercator.total_bounds)

        # Calculate overall bounds
        if all_bounds:
            all_bounds = np.array(all_bounds)
            total_bounds = [
                np.min(all_bounds[:, 0]),  # min x
                np.min(all_bounds[:, 1]),  # min y
                np.max(all_bounds[:, 2]),  # max x
                np.max(all_bounds[:, 3])   # max y
            ]
            ax.set_xlim(total_bounds[0], total_bounds[2])
            ax.set_ylim(total_bounds[1], total_bounds[3])

        # Add basemap
        providers = {
            "OpenStreetMap": ctx.providers.OpenStreetMap.Mapnik,
            "Carto Positron": ctx.providers.CartoDB.Positron,
            "Carto Dark": ctx.providers.CartoDB.DarkMatter,
        }
        
        ctx.add_basemap(
            ax,
            source=providers[basemap_style],
            zoom='auto'
        )

        # Set title if provided
        if title:
            plt.title(title)

        # Add legend if requested and labels are provided
        if add_legend and any(style.get('label') for style in layer_styles):
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Remove axes
        ax.set_axis_off()

        # Capture the figure as base64 before showing it
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Store the image in the state with metadata
        state["image_store"].append({
            "type": "map",
            "description": f"Visualized {len(geodataframe_names)} layers",
            "base64": img_base64
        })     

        # Display if visualization is enabled
        if "visualize" in state and state['visualize']:
            plt.show()

        # Prepare status message
        layer_info = []
        for i, gdf_name in enumerate(geodataframe_names):
            gdf = state["data_store"].get(gdf_name)
            layer_info.append(f"- Layer {i+1}: '{gdf_name}' ({len(gdf)} features)")

        return (f"Visualized {len(geodataframe_names)} layers over {basemap_style} basemap:\n" +
                "\n".join(layer_info))

    except Exception as e:
        return f"Error creating visualization: {type(e).__name__} : {str(e)}"

# Tool function to get centroids from polygon features    
def get_centroids(
    geodataframe_name: Annotated[str, "Name of GeoDataFrame containing polygon geometries"],
    output_geodataframe_name: Annotated[str, "Name for storing the GeoDataFrame with centroids"],
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Calculate centroids of polygons in a GeoDataFrame and create a new GeoDataFrame
    where the geometry column is replaced with centroids while preserving all other columns.

    Args:
        geodataframe_name: Name of the source GeoDataFrame stored in data_store
        output_geodataframe_name: Name for storing the resulting GeoDataFrame with centroids
        state: Current graph state containing geodataframes and results

    Returns:
        str: Status message with operation details and GeoDataFrame information
    """
    try:
        # Get the source GeoDataFrame
        geodataframe = state["data_store"].get(geodataframe_name)
        
        if geodataframe is None:
            return f"Error: GeoDataFrame '{geodataframe_name}' not found in data store"
            
        # Create a copy of the GeoDataFrame
        centroids_gdf = geodataframe.copy()
        
        # Replace geometry column with centroids
        centroids_gdf.geometry = geodataframe.geometry.centroid
        
        # Store result in data store
        if "data_store" not in state:
            state["data_store"] = {}
        state["data_store"][output_geodataframe_name] = centroids_gdf
        
        # Generate result message
        info_string, _ = get_dataframe_info(centroids_gdf)
        result = (
            f"Created centroids and stored result in GeoDataFrame '{output_geodataframe_name}'.\n"
            f"Description of GeoDataFrame:\n{info_string}"
        )

        return result
        
    except Exception as e:
        return f"Error calculating centroids: {type(e).__name__} : {str(e)}"
    
# Tool function to plot contour lines from a raster pixels
def plot_contour_lines(
    raster_path: Annotated[str, "Path to the raster file"],
    output_geodataframe_name: Annotated[str, "Name for storing the contour lines GeoDataFrame"],
    interval: Annotated[float, "Interval between contour lines. For snow: 1-5 inches; hilly terrain: 20-50m; mountains: 100-500m. Use get_raster_description for guidance."],
    state: Annotated[dict, InjectedState],
    min_value: Annotated[float | None, "Minimum value for contours (if None, uses raster minimum)"] = None,
    max_value: Annotated[float | None, "Maximum value for contours (if None, uses raster maximum)"] = None,
    plot_result: Annotated[bool, "Whether to display the plot"] = True,
    title: Annotated[str | None, "Title for the plot"] = None
) -> str:
    """
    Create contour lines from raster data with proper coordinate transformation, 
    stores them into a GeoDataFrame for future reuse, and optionally displays the plot.
    
    The contour interval should be chosen based on your data type and value range. For example:
    - For snow depth or rainfall in inches: try 1-5 inch intervals
    - For hilly terrain elevation: try 20-50 meter intervals
    - For mountains: try 100-500 meter intervals
    Use get_raster_description() to analyze your data and get a suggested interval.
    
    Args:
        raster_path: Path to the raster file
        output_geodataframe_name: Name for storing the resulting GeoDataFrame
        interval: Interval between contour lines
        min_value: Minimum value for contours (if None, uses raster minimum)
        max_value: Maximum value for contours (if None, uses raster maximum)
        plot_result: Whether to display the plot
        title: Plot title
        state: Current graph state
        
    Returns:
        str: Status message with operation results
    """
    try:
        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []

        # Open raster with rasterio for better spatial handling
        with rasterio.open(raster_path) as src:
            # Read the data and flip vertically for matplotlib
            data = src.read(1)
            data = np.flipud(data)
            
            # Get the nodata value
            nodataval = src.nodata
            
            # Replace nodata values with nan
            if nodataval is not None:
                data = np.where(data == nodataval, np.nan, data)

            # Determine min and max values
            if min_value is None:
                min_value = np.nanmin(data)
            if max_value is None:
                max_value = np.nanmax(data)

            # Create levels list
            levels = np.arange(min_value, max_value + interval, interval)

            # Generate contours using matplotlib
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            contours = plt.contour(data, levels=levels)
            
            if title:
                plt.title(title)
            plt.colorbar(label='Value')

            # Capture the figure as base64 before showing it
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            # Store the image in the state with metadata
            state["image_store"].append({
                "type": "map",
                "description": f"Created contour lines with interval {interval}.",
                "base64": img_base64
            })             

            # Create empty lists for geometries and values
            geometries = []
            values = []

            # Get raster properties
            height = src.height
            width = src.width
            transform = src.transform

            # Create coordinate arrays
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            # Process each contour level
            for i, collection in enumerate(contours.collections):
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) >= 2:
                        # Transform vertices to geographic coordinates
                        transformed_vertices = []
                        for x, y in vertices:
                            # Scale x and y to raster dimensions
                            x_scaled = x * (width - 1) / (width - 1)
                            y_scaled = (height - 1 - y) * (height - 1) / (height - 1)
                            
                            # Get geographic coordinates using the transform
                            geo_x, geo_y = transform * (x_scaled, y_scaled)
                            transformed_vertices.append((geo_x, geo_y))
                        
                        # Create LineString with transformed coordinates
                        line = LineString(transformed_vertices)
                        geometries.append(line)
                        values.append(contours.levels[i])

            # Create GeoDataFrame with the contour lines
            contour_gdf = gpd.GeoDataFrame(
                {'geometry': geometries, 'value': values},
                crs=src.crs
            )

            # Store in state
            state["data_store"][output_geodataframe_name] = contour_gdf

            if plot_result and "visualize" in state and state["visualize"]:
                plt.show()
            else:
                plt.close()

            # Prepare CRS information for the message
            crs_info = f"\n- CRS: {contour_gdf.crs.to_string()}" if contour_gdf.crs else ""

            return (f"Created contour lines with interval {interval}.\n"
                    f"- Number of contour features: {len(contour_gdf)}\n"
                    f"- Value range: {min_value:.2f} to {max_value:.2f}{crs_info}\n"
                    f"Results stored in GeoDataFrame '{output_geodataframe_name}'")

    except Exception as e:
        return f"Error creating contours: {type(e).__name__} : {str(e)}"
    
# Tool function to plot contour lines from a raster
def generate_contours_display(
    raster_path: Annotated[str, "Path to the raster file"],
    output_filename: Annotated[str, "Name for the output shapefile"],
    contour_interval: Annotated[float, "Interval between contour lines"],
    column_title: Annotated[str, "Name for the column containing contour values, 10 characters maximum"],
    nodataval: Annotated[int, "Value to use for no-data pixels"],
    state: Annotated[dict, InjectedState],    
    output_geodataframe_name: Annotated[str, "Name for storing the resulting GeoDataFrame in state"],
    output_folder: Annotated[str, "Folder path for storing intermediate and output files"] = SCRATCH_PATH,    
    min_value: Annotated[float | None, "Minimum value for contours (if None, uses raster minimum)"] = None,
    max_value: Annotated[float | None, "Maximum value for contours (if None, uses raster maximum)"] = None,
    cleanup_files: Annotated[bool, "Whether to remove temporary files after processing"] = False,
    plot_result: Annotated[bool, "Whether to display the plot"] = True,
    plot_title: Annotated[str | None, "Title for the plot"] = None,
    colormap: Annotated[str, "Matplotlib colormap name"] = 'viridis',
    # figsize: Annotated[tuple[int], "Figure size (width, height) in inches"] = (12, 8),
    figsize: Annotated[str, "Figure size as JSON array [width, height] in inches, e.g. '[12, 8]'"] = "[12, 8]", # To accomodate Gemini's indigestion of objects in arguments
    add_colorbar: Annotated[bool, "Whether to add a colorbar to the plot"] = True,
    plot_background: Annotated[bool, "Whether to display raster as background"] = True
) -> str:
    """
    Generate contour lines from a raster file, optionally plot them, and return as a GeoDataFrame.
    Only generates contours for values between min_value and max_value.
    
    Args:
        raster_path: Path to the raster file
        output_folder: Folder path for storing intermediate files
        output_filename: Name for the output shapefile
        contour_interval: Interval between contour lines
        state: Current graph state
        output_geodataframe_name: Name for storing the resulting GeoDataFrame in state
        min_value: Minimum value for contours (if None, uses raster minimum)
        max_value: Maximum value for contours (if None, uses raster maximum)
        plot_result: Whether to display visualization
        plot_title: Title for the plot
        cleanup_files: Whether to remove temporary files after processing
        colormap: Matplotlib colormap name
        figsize: Figure size as JSON array [width, height] in inches, e.g. '[12, 8]'
        add_colorbar: Whether to add a colorbar to the plot
        plot_background: Whether to display raster as background
        
    Returns:
        str: Status message with operation results
    """
    try:
        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []

        # Validate input parameters
        if not output_folder:
            return "Error: output_folder must be specified"
        if not output_filename:
            return "Error: output_filename must be specified"
        if not os.path.exists(output_folder):
            return f"Error: Output folder does not exist: {output_folder}"

        output_filename = output_filename +'.shp'
        output_shapefile = os.path.join(output_folder, output_filename)
        # To accomodate for Gemini's indigestion of objects as function arguments
        if isinstance(figsize, str):
            figsize = ast.literal_eval(figsize)

        # Open raster dataset
        raster_dataset = gdal.Open(raster_path, GA_ReadOnly)
        if raster_dataset is None:
            return "Error: Could not open raster dataset"

        # Create a temporary raster with filtered values
        temp_raster_path = os.path.join(output_folder, 'temp_filtered.tif')
        
        # Create a copy of the original raster
        driver = gdal.GetDriverByName('GTiff')
        temp_dataset = driver.CreateCopy(temp_raster_path, raster_dataset, 0)
        temp_band = temp_dataset.GetRasterBand(1)
        
        # Read the data into a numpy array
        data = temp_band.ReadAsArray()
        
        # Create mask for values outside the range and set them to nodata
        mask = (data < min_value) | (data > max_value) | (data == nodataval)
        data[mask] = nodataval
        
        # Write the filtered data back to the temporary raster
        temp_band.WriteArray(data)
        temp_band.SetNoDataValue(nodataval)
        temp_band.FlushCache()

        # Get spatial reference from raster
        raster_srs = osr.SpatialReference(wkt=raster_dataset.GetProjection())

        # Create output shapefile      
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # If file exists, remove it first
        if os.path.exists(output_shapefile):
            driver.DeleteDataSource(output_shapefile)

        vector_ds = driver.CreateDataSource(output_shapefile)
        if vector_ds is None:
            return f"Error: Could not create output shapefile at {output_shapefile}"

        contour_layer = vector_ds.CreateLayer('contour', srs=raster_srs)

        # Add fields to the layer
        id_field = ogr.FieldDefn("ID", ogr.OFTInteger)
        contour_field = ogr.FieldDefn(column_title, ogr.OFTReal)
        contour_layer.CreateField(id_field)
        contour_layer.CreateField(contour_field)

        fixed_levels = []

        # Generate contours using the filtered raster
        gdal.ContourGenerate(
            temp_band,
            contour_interval,
            float(min_value),
            fixed_levels,
            nodataval,
            0,
            contour_layer,
            0,
            1
        )

        # Close the datasets
        vector_ds = None
        temp_dataset = None
        raster_dataset = None
        del vector_ds
        del temp_dataset
        del raster_dataset

        # Clean up temporary raster
        if cleanup_files:
            os.remove(temp_raster_path)

        # Read the shapefile into a GeoDataFrame
        contour_gdf = gpd.read_file(output_shapefile)

        # Store in state
        state["data_store"][output_geodataframe_name] = contour_gdf

        # Plot the results if requested
        if plot_result and "visualize" in state and state["visualize"]:
            fig, ax = plt.subplots(figsize=figsize)

            if plot_background:
                # Read and plot the raster data as background
                with rasterio.open(raster_path) as src:
                    raster_data = src.read(1)
                    raster_data = np.ma.masked_equal(raster_data, nodataval)
                    
                    # Get the extent for proper alignment
                    bounds = src.bounds
                    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                    
                    # Plot the raster with transparency
                    plt.imshow(raster_data, extent=extent, cmap='gray', alpha=0.3)

            # Plot contours with different colors based on their values
            contour_plot = contour_gdf.plot(
                column=column_title,
                cmap=colormap,
                linewidth=0.5,
                ax=ax
            )

            # Add colorbar if requested
            if add_colorbar:
                plt.colorbar(
                    plt.cm.ScalarMappable(
                        norm=plt.Normalize(
                            vmin = min_value,
                            vmax = max_value
                        ),
                        cmap=colormap
                    ),
                    ax=ax,
                    label=column_title
                )

            # Add title if provided
            if plot_title:
                plt.title(plot_title)

            # Add some basic styling
            # ax.set_aspect('equal')
            plt.grid(True, linestyle='--', alpha=0.6)

        # Capture the figure as base64 before showing it
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Store the image in the state with metadata
        state["image_store"].append({
            "type": "map",
            "description": f"Created contour lines with interval {contour_interval}.",
            "base64": img_base64
        })      
            
        if plot_result and "visualize" in state and state["visualize"]:
            plt.show()
        else:
            plt.close()

        # Clean up temporary files if requested
        if cleanup_files:
            extensions = [".shp", ".dbf", ".prj", ".shx"]
            for ext in extensions:
                temp_file = output_shapefile.replace(".shp", ext)
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Prepare return message
        crs_info = f"\n- CRS: {contour_gdf.crs.to_string()}" if contour_gdf.crs else ""
        return (f"Generated contour lines with interval {contour_interval}.\n"
                f"- Number of contour features: {len(contour_gdf)}\n"
                f"- Value range: {min_value:.2f} to {max_value:.2f}{crs_info}\n"
                f"- Values stored in column: {column_title}\n"
                f"Results stored in GeoDataFrame '{output_geodataframe_name}'")

    except Exception as e:
        return f"Error generating contours: {type(e).__name__} : {str(e)}" 

# Tool function to create a bivariate choroplate map 
def make_bivariate_map(
    dataframe_name: Annotated[str, "Name of GeoDataFrame containing map data"],
    var1: Annotated[str, "Name of first variable column"],
    var2: Annotated[str, "Name of second variable column"],
    var1_name: Annotated[str, "Display name for first variable"],
    var2_name: Annotated[str, "Display name for second variable"],
    title: Annotated[str, "Title for the map"],
    output_variable_name: Annotated[str, "Name for storing the results"],
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Create a bivariate choropleth map showing relationship between two variables.

    Args:
        dataframe_name: Name of GeoDataFrame in memory
        var1: Column name for first variable
        var2: Column name for second variable
        var1_name: Display name for first variable
        var2_name: Display name for second variable
        title: Title for the map
        output_variable_name: Name for storing the results
        state: Current graph state containing dataframes

    Returns:
        str: Status message with map generation details
    """
    try:
        # Initialize image store if needed
        if "image_store" not in state:
            state["image_store"] = []

        # Get the GeoDataFrame from state
        gdf = state["data_store"].get(dataframe_name)
        if gdf is None:
            return f"Error: GeoDataFrame '{dataframe_name}' not found in data store"

        # Create a deep copy of the dataframe
        gdf_copy = gdf.copy(deep=True)
        
        # Remove rows where either variable is NaN
        mask = gdf_copy[var1].notna() & gdf_copy[var2].notna()
        
        # Calculate quantiles using the entire dataset
        var1_quantiles = pd.qcut(gdf_copy.loc[mask, var1], q=3, labels=['Low', 'Medium', 'High'])
        var2_quantiles = pd.qcut(gdf_copy.loc[mask, var2], q=3, labels=['Low', 'Medium', 'High'])
        
        # Assign quantiles back to the dataframe
        gdf_copy.loc[mask, 'var1_quantile'] = var1_quantiles
        gdf_copy.loc[mask, 'var2_quantile'] = var2_quantiles

        # Create the bivariate color scheme
        colors = {
            ('Low', 'Low'): '#e8e8e8',      # Light gray
            ('Low', 'Medium'): '#e4acac',    # Light red
            ('Low', 'High'): '#c85a5a',      # Dark red
            ('Medium', 'Low'): '#b0d5df',    # Light blue
            ('Medium', 'Medium'): '#ad9ea5',  # Purple
            ('Medium', 'High'): '#985356',    # Dark purple
            ('High', 'Low'): '#64acbe',      # Blue
            ('High', 'Medium'): '#627f8c',    # Dark blue
            ('High', 'High'): '#574249'      # Very dark purple
        }

        # Initialize color column with default white color for NaN values
        gdf_copy['color'] = '#ffffff'
        
        # Update colors for valid data points
        for idx in gdf_copy[mask].index:
            var1_cat = gdf_copy.loc[idx, 'var1_quantile']
            var2_cat = gdf_copy.loc[idx, 'var2_quantile']
            gdf_copy.loc[idx, 'color'] = colors[(var1_cat, var2_cat)]

        # Create figure with two subplots: one for map, one for legend
        fig = plt.figure(figsize=(15, 10))
        
        # Create main map subplot with specific dimensions
        ax_map = fig.add_axes([0.05, 0.1, 1.0, 0.9])  # [left, bottom, width, height]
        
        # Create legend subplot
        ax_legend = fig.add_axes([0.1, 0.1, 0.1, 0.1])  # Position legend on the right

        # Plot the map
        gdf_copy.plot(color=gdf_copy['color'], ax=ax_map, edgecolor='white', linewidth=0.5)
        ax_map.axis('off')

        # Add title
        ax_map.set_title(title, pad=20, size=16)

        # Create legend
        ax_legend.clear()
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        for spine in ax_legend.spines.values():
            spine.set_visible(False)

        # Define legend grid dimensions
        grid_size = 3
        cell_size = 1/grid_size
        
        # Plot legend squares
        for i in range(grid_size):
            for j in range(grid_size):
                var1_cat = ['Low', 'Medium', 'High'][i]
                var2_cat = ['Low', 'Medium', 'High'][j]
                color = colors[(var1_cat, var2_cat)]
                rect = Rectangle((j * cell_size, (2-i) * cell_size), 
                               cell_size, cell_size, 
                               facecolor=color, 
                               edgecolor='white')
                ax_legend.add_patch(rect)

        # Set legend axis limits
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)

        # Add variable labels
        ax_legend.text(1.5 * cell_size, 1.3, var2_name, ha='center', va='bottom')
        ax_legend.text(0, 1.5 * cell_size, var1_name, ha='right', va='center', rotation=90)

        # Add Low/Medium/High labels
        for i, label in enumerate(['Low', 'Medium', 'High']):
            # Labels for var1 (y-axis)
            ax_legend.text(-0.1, (2-i) * cell_size + 0.5 * cell_size, 
                          label, ha='right', va='center')
            # Labels for var2 (x-axis)
            ax_legend.text(i * cell_size + 0.5 * cell_size, -0.1,
                          label, ha='center', va='top')

        # Add note about missing data if there are any
        if (~mask).any():
            ax_legend.text(0.5, -0.3, 
                          f'White areas: missing data\n({(~mask).sum()} features)', 
                          ha='center', va='top', fontsize=8)

        # Store results in state
        if "data_store" not in state:
            state["data_store"] = {}
        
        # Store the visualization result
        result = {
            'figure': fig,
            'map_axis': ax_map,
            'legend_axis': ax_legend,
            'processed_data': gdf_copy
        }
        state["data_store"][output_variable_name] = result

        # Capture the figure as base64 before showing it
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Store the image in the state with metadata
        state["image_store"].append({
            "type": "map",
            "description": f"Created bivariate choropleth map comparing {var1_name} and {var2_name}.",
            "base64": img_base64
        })  

        if "visualize" in state and state['visualize']:
            plt.show()

        # Return success message
        return (f"Created bivariate choropleth map comparing {var1_name} and {var2_name}.\n"
                f"- Total features: {len(gdf_copy)}\n"
                f"- Features with valid data: {mask.sum()}\n"
                f"- Features with missing data: {(~mask).sum()}\n"
                f"Results stored in '{output_variable_name}'")

    except Exception as e:
        return f"Error creating bivariate map: {type(e).__name__} : {str(e)}"
    
# Tool function to mark cases when the task is geospatial in nature but cannot be solved by the agent
def reject_task(
    state: Annotated[dict, InjectedState]
) -> str:
    """
    This tool should be called when the task cannot be solved to return a standardized message indicating that the current task cannot be solved 
    with the available tools and datasets.

    Args:
        state: Current graph state containing dataframes and results

    Returns:
        str: Standardized rejection message
    """
    return "The task is not solvable with the tools and datasets available."

load_data_tool = StructuredTool.from_function(
    func=load_data, 
    name='load_data',
    description='Loads statistical data from the DATA_CATALOG into a Pandas DataFrame. \
        Returns the DataFrame and its description, inlcuding \
        DataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.'
)

load_geodata_tool = StructuredTool.from_function(
    func=load_geodata, 
    name='load_geodata',
    description='Loads vector geospatial data from the GEO_CATALOG into a GeoPandas GeoDataFrame. \
        Returns the GeoDataFrame and its description, inlcuding \
        GeoDataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.'
)

get_raster_path_tool = StructuredTool.from_function(
    func=get_raster_path, 
    name='get_raster_path',
    description='Constructs path to raster data from the catalog, handling both regular GeoTIFF and zipped files. \
        Return path to the raster file in GEO_CATALOG, formatted for either direct GeoTIFF access or zip archive access'
)

get_raster_description_tool = StructuredTool.from_function(
    func=get_raster_description, 
    name='get_raster_description',
    description='Get description of a raster dataset including metadata (driver, width, height, number of bands, coordinate reference system, transform, nodata, data type) and \
        basic statistics (valid pixels, min, max, mean, standard deviation, NaN count by band). \
        Returns formatted description of raster properties and statistics'
)

analyze_raster_overlap_tool = StructuredTool.from_function(
    func=analyze_raster_overlap, 
    name='analyze_raster_overlap',
    description='Analyze overlap between two rasters (raster 1 and raster 2) and \
        calculate statistics (total value, min, max, mean, standard deviation, count) for overlapping pixels from raster 2. \
        Optionally displays a visualization of masked areas. \
        '
)

get_values_from_raster_with_geometries_tool = StructuredTool.from_function(
    func=get_values_from_raster_with_geometries, 
    name='get_values_from_raster_with_geometries',
    description='Mask a raster using vector geometries from a GeoDataFrame and calculate statistics for the masked area of the raster.\
    The tool returns statistics (total value in masked areas, min, max, mean, standard deviation, count) and stores the cropped raster data in the specified variable. \
    Optionally displays a visualization of masked areas. ' 
)

merge_dataframes_tool = StructuredTool.from_function(
    func=merge_dataframes, 
    name='merge_dataframes',
    description='Merge statistical and geospatial dataframes using specified key columns. \
        The resulting merged dataframe will preserve all rows from geodataset, matching with dataset where possible and filling with NaN where no match exists. \
        Can work also on 2 DataFrame or a2 GeoDataFrames.\
        Returns the DataFrame and its description, inlcuding \
        DataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.'
)

get_unique_values_tool = StructuredTool.from_function(
    func=get_unique_values, 
    name='get_unique_values',
    description='Gets unique values from a specified column in a DataFrame/GeoDataFrame. Should be used to clarify the spelling of names of ojects like countries, \
        regions, subregions, continents, etc. before using the filtering tools to avoid missing objects due to using differeing spelling or convention. \
        Returns list of unique values from a specified column in a DataFrame/GeoDataFrame.'
)

filter_categorical_tool = StructuredTool.from_function(
    func=filter_categorical, 
    name='filter_categorical',
    description='Filters DataFrame/GeoDataFrame by categorical values in specified columns. Can be used to select specific countries, subregions, continents from respective columns. \
    To get better results, can be used after the values for filtering are compared with spellings in the selected column using get_unique_values_tool. \
    Returns filtered DataFrame/GeoDataFrame, filters applied, details on columns and rows of the new dataframe.'
)

filter_numerical_tool = StructuredTool.from_function(
    func=filter_numerical, 
    name='filter_numerical',
    description='Filters DataFrame/GeoDataFrame using numerical conditions via query method (e.g. "col1 > 25 and col2 < 100").\
    To identify most suitable values for the filter, calculate_column_statistics_tool can be used before filtering.\
    Returns filtered DataFrame/GeoDataFrame, filters applied, details on columns and rows of the new dataframe.'
)

calculate_column_statistics_tool = StructuredTool.from_function(
    func=calculate_column_statistics, 
    name='calculate_column_statistics',
    description='Calculate summary statistics (count (total, valid, and missing values), min/max range, mean, standard deviation) for a numerical column in a DataFrame/GeoDataFrame.\
        Optionally calculates quantiles (25th, 50th/median, 75th percentiles). Can calculate additional custom quantiles if specified. \
        Returns a formatted summary of the statistics. The tool can be used before applying numerical filters to calculate the optimal filtering thresholds.'
)

create_buffer_tool = StructuredTool.from_function(
    func=create_buffer, 
    name='create_buffer',
    description='Create buffer zones around geometries in a GeoDataFrame using specified buffer size.\
        Stores the result in a new GeoDataFrame with the specified name. Returns the DataFrame and its description, inlcuding \
        DataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.'
)

make_choropleth_map_tool = StructuredTool.from_function(
    func=make_choropleth_map, 
    name='make_choropleth_map',
    description='Creates a choropleth map visualization from a specified column in a GeoDataFrame.'
)

filter_points_by_raster_values_tool = StructuredTool.from_function(
    func=filter_points_by_raster_values, 
    name='filter_points_by_raster_values',
    description='Sample raster values at point locations and filter points based on threshold conditions. \
        Adds these values to a specified column in the GeoDataFrame. Returns filtered GeoDataFrame. Optionally, visualizes the filtered points overlaid on the raster. \
        Also returns text summary with filtering statistics (total points, filtered points, name of the column cotaining sampled values).'
)

select_features_by_spatial_relationship_tool = StructuredTool.from_function(
    func=select_features_by_spatial_relationship, 
    name='select_features_by_spatial_relationship',
    description='Select features from one GeoDataFrame based on multiple spatial relationships with another.\
    Cannot process GeoDataFrame with over 150,000 features. Features that satisfy ANY of the specified predicates will be selected (OR logic).\
    Automatically handles CRS differences by reprojecting features to match reference CRS.\
    Returns flitered GeoDataFrame. Optionally, visualizes the filtered GeoDataFrame over reference GeoDataFrame. Also returns number of selected features, predicates used and name of the filtered dataframe.'
)

calculate_line_lengths_tool = StructuredTool.from_function(
    func=calculate_line_lengths, 
    name='calculate_line_lengths',
    description='Calculates the lengths of line features in a GeoDataFrame in kilometers, using appropriate UTM projections for accurate distance measurements. \
        Returns total lengh of line features in km, GeoDataFrame in UTM projection with column storing features length in m, and a descritive message of the results.'
)

calculate_columns_tool = StructuredTool.from_function(
    func=calculate_columns, 
    name='calculate_columns',
    description='Performs mathematical operations ("multiply", "divide", "add", "subtract") between columns of two DataFrames/GeoDataFrames or columns of the same DataFrame/GeoDataFrame. \
        Returns DataFrame/GeoDataFrame with a column containing the operation result and message about operation, resulting dataframe and column.'
)

scale_column_by_value_tool = StructuredTool.from_function(
    func=scale_column_by_value, 
    name='scale_column_by_value',
    description='Performs a basic mathematical operation (multiply, divide, add, or subtract) between a column\'s values and a specified numeric value. \
        It returns a new DataFrame/GeoDataFrame with the original data plus a new column containing the calculation results as well as and message about operation, resulting dataframe and column.'
)

make_heatmap_tool = StructuredTool.from_function(
    func=make_heatmap, 
    name='make_heatmap',
    description='The make_heatmap tool creates an interactive heatmap visualization from point data in a GeoDataFrame, with intensity based on values from a specified column. \
        It returns a visualization where color intensity represents data density or magnitude, and can optionally store the HTML output. \
        The tool supports customization of map appearance including basemap style, zoom level, and color radius.'
)

visualize_geographies_tool = StructuredTool.from_function(
    func=visualize_geographies, 
    name='visualize_geographies',
    description='Displays multiple GeoDataFrames as map layers with custom styling. \
        It returns a visualization with different geometries (points, lines, polygons) rendered in distinct colors over a basemap, with optional legends and titles. \
        The tool handles coordinate system transformations automatically and supports various basemap styles.'
)

get_centroids_tool = StructuredTool.from_function(
    func=get_centroids, 
    name='get_centroids',
    description='Calculates the geometric center points of polygon features in a GeoDataFrame. \
        It returns a new GeoDataFrame that preserves all attribute columns from the original but replaces the geometry column with centroid points.\
        Also returns description of the GeoDataFrame, inlcuding GeoDataFrame name, number of entries, columns names, number of non-null cells, data types, share of non-empty cells in columns.'
)

plot_contour_lines_tool = StructuredTool.from_function(
    func=plot_contour_lines, 
    name='plot_contour_lines',
    description='creates topographic-style contour lines from raster data at specified intervals. \
        It returns a GeoDataFrame containing line geometries representing areas of equal value (e.g., elevation, temperature), \
        with proper coordinate transformation for mapping. Also returns message with contour interval, number of contour features, range of values, crs, name of teh resulting GeoDataFrame.\
        The tool can also generate an optional visualization of the contours with customizable styling.'
)

generate_contours_display_tool = StructuredTool.from_function(
    func=generate_contours_display, 
    name='generate_contours_display',
    description='Creates contour lines from raster data using GDAL\'s ContourGenerate algorithm. \
        It returns a GeoDataFrame containing line geometries that represent areas of equal value with associated attributes. \
        To get values for the function arguments, call get_raster_description_tool first.\
        Tool supports customization of contour intervals, value ranges. Note, that common nodata values could be -99999, -9999, -32768, -999, -3.4e38  and assign function arguments accordingly.\
        The tool also produces a shapefile that can be optionally saved, optionally can visualize background raster and generated contour lines and customize styling.'
)

make_bivariate_map_tool = StructuredTool.from_function(
    func=make_bivariate_map, 
    name='make_bivariate_map',
    description='Create and displays a bivariate choropleth map showing relationship between two variables.'
)

reject_task_tool = StructuredTool.from_function(
    func=reject_task, 
    name='reject_task',
    description='This tool should be called when the task cannot be solved to return \
        a standardized message indicating that the current task cannot be solved with the available tools and datasets.'
)