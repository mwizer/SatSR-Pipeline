import datetime as dt
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import ee
import geemap
import geopy
import matplotlib.pyplot as plt
import rasterio
import rasterio.transform
from geopy.distance import distance
from geopy.geocoders import Nominatim
from tqdm import tqdm

from satsr_pipeline.modules.data_loader import train_val_test_loader
from satsr_pipeline.modules.divide_data import divide_images
from satsr_pipeline.modules.RegionInfo import RegionInfo

logging.basicConfig(level=logging.INFO, format="SatSR %(levelname)s: %(message)s")

# ToDo: Add metadata for every image. Locations, date, clouds etc.
# ToDo: Can we consider multithreading or multiprocessing for downloading images? Or some other parts of the code, for optimization.
# ToDo: Ad2: I think we can multithread downloading of tiles, but not with whole images bcs of the way we save them... but we can check it later
#       What are most time-consuming parts of the code? Maybe we can optimize them.

__version__ = "1.1.0"


class SatSRError(Exception):
    """Base class for exceptions in this module."""

    pass


# ToDo: Separate properties and static methods to a separate class, that will be inherited by the Pipeline class.
# ToDo: Move content to a main module and display only the usage of the Pipeline class in the main class.
class Pipeline:
    def __init__(
        self,
        data_path: str = "data",
        bands: list = ["B4", "B3", "B2"],
        product: str = "COPERNICUS/S2_SR_HARMONIZED",
    ):
        """
        Initialize the Pipeline class with given parameters.

        Parameters
        ----------
        data_path : str, optional
            The path to the data directory. Default is 'data'.
        bands : list, optional
            The list of bands to be used. Default is ['B4', 'B3', 'B2'] which are equivalents of red, green and blue bands for Sentinel2.
        product : str, optional
            The satellite image product for download. Default is "COPERNICUS/S2_SR_HARMONIZED".
        """
        self.geolocator = Nominatim(
            user_agent="sr_pipeline"
        )  # ToDo: We need to check license

        self
        
        self.title = "image_title"
        self._nomim_loc_name = None
        self._point = None
        self._distance_x = None
        self._distance_y = None
        self._use_geolocator_bbox = False
        self._full_box = None
        self._time_start = None
        self._time_end = None
        self._region_info = None
        self._img_idx = None

        self.data_path = data_path
        self.bands = bands
        self.product = product

        ee.Authenticate()
        ee.Initialize()

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        if not isinstance(value, str):
            raise SatSRError("Path must be a string")
        self._data_path = Path(value)

    @property
    def bands(self):
        return self._bands

    @bands.setter
    def bands(self, value):
        if not isinstance(value, (list, tuple)):
            raise SatSRError("Bands must be a list-like object")
        self._bands = value

    @property
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        if not isinstance(value, str):
            raise SatSRError("Product must be a string")
        self._product = value

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value: Union[tuple, str]):
        if isinstance(value, str):
            self._nomim_loc_name = value
            self._point = self._text2cord(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            self._nomim_loc_name = None
            self._point = geopy.point.Point(value)
        else:
            raise SatSRError("Point must be a string or tuple with two values")

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value: str):
        if isinstance(value, str):
            self._title = value
        else:
            raise SatSRError("Title must be a string")

    @property
    def distance_x(self):
        return self._distance_x

    @distance_x.setter
    def distance_x(self, value: float):
        if not self._is_number(value):
            raise SatSRError("distance_x must be a number")
        self._distance_x = value

    @property
    def distance_y(self):
        return self._distance_y

    @distance_y.setter
    def distance_y(self, value: float):
        if not self._is_number(value):
            raise SatSRError("distance_y must be a number")
        self._distance_y = value
        
    @property
    def use_geolocator_bbox(self):
        return self._use_geolocator_bbox

    @use_geolocator_bbox.setter
    def use_geolocator_bbox(self, value: bool):
        if not isinstance(value, bool):
            raise SatSRError("use_geolocator_bbox must be a boolean")
        if self._nomim_loc_name is None:
            raise SatSRError("use_geolocator_bbox cannot be used without a location name")
        self._use_geolocator_bbox = value

    @property
    def meter_per_pixel(self):
        return self._meter_per_pixel

    @meter_per_pixel.setter
    def meter_per_pixel(self, value: float):
        if not self._is_number(value):
            raise SatSRError("meter_per_pixel must be a number")
        self._meter_per_pixel = value
        
    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, value: Union[str, dt.datetime]):
        if isinstance(value, str):
            self._time_start = dt.datetime.strptime(value, "%Y-%m-%d")
        elif isinstance(value, dt.datetime):
            self._time_start = value
        else:
            raise SatSRError("time_start must be a string or datetime object")

    @property
    def time_end(self):
        return self._time_end

    @time_end.setter
    def time_end(self, value: Union[str, dt.datetime]):
        if isinstance(value, str):
            self._time_end = dt.datetime.strptime(value, "%Y-%m-%d")
        elif isinstance(value, dt.datetime):
            self._time_end = value
        else:
            raise SatSRError("time_end must be a string or datetime object")

    @property
    def img_idx(self):
        return self._img_idx
    
    @img_idx.setter
    def img_idx(self, value: int):
        if isinstance(value, int):
            self._img_idx = value
        else:
            raise SatSRError("img_idx must be a int")

    def setup_point(
        self,
        point: Union[tuple, str],
        time_start: str,
        time_end: str,
        title: str = "image_title",
        meter_per_pixel: float = 5.0,
        distance_x: float = 1000,  # in meters
        distance_y: float = 1000,
        use_geolocator_bbox: bool = False,
    ):
        """Set up the point of interest and related parameters.

        Parameters
        ----------
        point : Union[tuple, str]
            The geographical point of interest. Can be a tuple (latitude, longitude) or a string address.
        time_start : str, optional
            The start time for the data in 'YYYY-MM-DD' format.
        time_end : str, optional
            The end time for the data in 'YYYY-MM-DD' format.
        title : str, optional
            The title of the image. Default is 'image_title'.
        meter_per_pixel : float, optional
            The resolution in meters per pixel. Default is 5.0.
        distance_x : float, optional
            The distance in the x-direction in meters. Default is 1000.
        distance_y : float, optional
            The distance in the y-direction in meters. Default is 1000.
        use_geolocator_bbox : bool, optional
            If True, use the geolocator to get the bounding box. Default is False.
        """
        self.point = point
        self.title = title
        self.distance_x = distance_x
        self.distance_y = distance_y
        self.meter_per_pixel = meter_per_pixel
        self._use_geolocator_bbox = use_geolocator_bbox
        self.full_box = self._create_box()
        self.region_info = RegionInfo(
            box=self.full_box
        )
        self.time_start = time_start
        self.time_end = time_end

    def _text2cord(self, text: str):
        """
        Convert a text address to geographical coordinates.

        Parameters
        ----------
        text : str
            The address in text format.

        Returns
        -------
        geopy.point.Point
            A Point object with the latitude and longitude of the address.
        """
        point = self.geolocator.geocode(text)
        return geopy.point.Point([point.latitude, point.longitude])

    def _create_box(self) -> dict:
        """
        Create a bounding box around the point of interest.

        Returns
        -------
        dict
            A dictionary representing the bounding box with keys 'left', 'right', 'top', and 'bottom'.
        """
        if self._use_geolocator_bbox and self._nomim_loc_name is not None:
            
            polygon = self.geolocator.geocode(self._nomim_loc_name, exactly_one=True, geometry='geojson').raw['geojson']['coordinates'][0]
            min_lon = min(coord[0] for coord in polygon)
            max_lon = max(coord[0] for coord in polygon)
            min_lat = min(coord[1] for coord in polygon)
            max_lat = max(coord[1] for coord in polygon)
            box = {
                "left": min_lon,
                "right": max_lon,
                "top": max_lat,
                "bottom": min_lat,
            }
        
        else:
            _distance_x = distance(meters=self.distance_x / 2)
            _distance_y = distance(meters=self.distance_y / 2)
            box = {
                "left": _distance_x.destination(self.point, 270).longitude,
                "right": _distance_x.destination(self.point, 90).longitude,
                "top": _distance_y.destination(self.point, 0).latitude,
                "bottom": _distance_y.destination(self.point, 180).latitude,
            }
            
            
        return box

    @staticmethod
    def _is_number(value) -> bool:
        """Check if the value is a number."""
        return isinstance(value, (int, float))

    def info(self):
        """
        Print information about the current state of the object.
        """
        logging.info(f"Point: {self.point.longitude, self.point.latitude}")
        logging.info(f"Meter per pixel: {self.meter_per_pixel}")
        logging.info(f"Distance x: {self.distance_x}")
        logging.info(f"Distance y: {self.distance_y}")
        logging.info(f"Box: {self.full_box}")
        logging.info(f"Time range: {self.time_start} - {self.time_end}")

    # ToDo: We can expand plotting and move this to a separate class.
    def plot_image(
        self, file_name: str, path_folder: str = "data/", brightness: float = 1
    ):
        """
        Plot the satellite image with optional brightness adjustment.
        Parameters
        ----------
        file_name : str
            The name of the image file.
        path_folder : str, optional
            The folder path where the image file is located. Default is 'data/satsr'.
        brightness : float, optional
            The brightness adjustment factor. Default is 1.
        """
        path_folder = Path(path_folder)

        with rasterio.open(path_folder / file_name) as src:
            img = src.read()
            meta = src.meta
            bands = src.descriptions

        img = (img - img.min()) / (img.max() - img.min())
        img = img.transpose(1, 2, 0)[:, :, [0, 1, 2]]
        # print(img.max(), img.min(), img.mean())
        # # Apply brightness adjustment if needed
        img = img * brightness
        plt.imshow(img)
        plt.show()

    def _split_region(self, region: ee.Geometry, tile_size: int = 1000):
        """
        Split a region into smaller tiles of specified size.

        Parameters
        ----------
        region : ee.Geometry
            The region to be split.
        tile_size : int, optional
            The size of each tile in meters. Default is 1000.

        Returns
        -------
        list of ee.Geometry
            A list of smaller tiles as ee.Geometry objects.
        """
        coords = region["coordinates"][0]
        min_longitude = min([coord[0] for coord in coords])
        max_longitude = max([coord[0] for coord in coords])
        min_latitude = min([coord[1] for coord in coords])
        max_latitude = max([coord[1] for coord in coords])

        _distance_longitude = distance(meters=tile_size)
        _distance_latitude = distance(meters=tile_size)

        longitude_points = []
        latitude_points = []
        current_longitude = min_longitude
        current_latitude = min_latitude
        
        while (
            current_longitude < max_longitude
        ):
            next_longitude = _distance_longitude.destination(
                geopy.point.Point([current_latitude, current_longitude]), 90
            ).longitude
            longitude_points.append((current_longitude, next_longitude))
            current_longitude = next_longitude

        while (
            current_latitude < max_latitude
        ):
            next_latitude = _distance_latitude.destination(
                geopy.point.Point([current_latitude, current_longitude]), 0
            ).latitude
            latitude_points.append((current_latitude, next_latitude))
            current_latitude = next_latitude

        tiles = []
        for i in range(len(longitude_points)):
            for j in range(len(latitude_points)):
                new_box = {
                    "left": longitude_points[i][0],
                    "right": longitude_points[i][1],
                    "top": latitude_points[j][1],
                    "bottom": latitude_points[j][0],
                }

                tiles.append(
                    ee.Geometry.Polygon(
                        [
                            (new_box["left"], new_box["bottom"]),
                            (new_box["left"], new_box["top"]),
                            (new_box["right"], new_box["top"]),
                            (new_box["right"], new_box["bottom"]),
                        ]
                    )
                )

        return tiles

    def download_image(
        self,
        file_name_base: str = "image",
        tile_size: int = 1000,
        max_cloud: float = 0.3,
        verbose: bool = False,
    ):
        """
        Download satellite images for the specified area of interest.

        Parameters
        ----------
        product : str, optional
            The satellite image product to download. Default is "COPERNICUS/S2_SR_HARMONIZED".
        file_name_base : str, optional
            The base name for the downloaded image files. Default is 'image'.
        tile_size : int, optional
            The size of each tile in meters. Default is 1000.
        max_cloud : float, optional
            The maximum cloud coverage percentage allowed. Default is 0.3.
        verbose : bool, optional
            If True, suppress output messages from geemap. Default is False.
        """
        # ToDo: As I understand we are collecting here the whole location, so the name should be more descriptive.
        #       Additionally, consider to split this method, as it is used in the download_images_list method and not whole operations are needed to be repeated.
        self.data_path.mkdir(parents=True, exist_ok=True)
        aoi = ee.Geometry.Polygon(
            self.region_info.bounding_box_coordinates
        )  # area of interest

        collection = (
            ee.ImageCollection(self.product)
            .filterDate(self.time_start, self.time_end)
            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "not_greater_than", max_cloud)
        )
        image = collection.select(self.bands).median().unmask()
        timestamp = collection.first().get('system:time_start').getInfo()
        timestamp = dt.datetime.fromtimestamp(timestamp / 1000)

        tiles_list = self._split_region(aoi.getInfo(), tile_size)

        # check if there are any images in the folder
        # ToDo: It should check maybe caught idex with regex, this is a bit naive.
        # ToDo: I would change way we name images, so we wont override them. Maybe something with date, location and tiles?
        if self.img_idx is None:
            number_of_images = [
                f for f in os.listdir(self.data_path) if f.startswith(file_name_base)
            ]
            if len(number_of_images) == 0:
                self.img_idx = 0
            else:
                self.img_idx = int(number_of_images[-1].split("_")[-2]) + 1
        else:
            self.img_idx += 1

        # ToDo: Consider flag for turn on-off tqdm.
        for tile_idx, tile in enumerate(
            tqdm(
                tiles_list,
                desc=f"Downloading tiles for {self.img_idx} image",  # ToDo: I would give here the whole name of file
            )
        ):
            output_file = self.data_path / Path(
                f"{file_name_base}_{self.img_idx}_{tile_idx}.tif"
            )
            metadata_file = self.data_path / Path(
                f"metadata_{file_name_base}_{self.img_idx}_{tile_idx}.json"
            )
            if verbose:
                geemap.ee_export_image(
                    image,
                    filename=output_file,
                    scale=self.meter_per_pixel,
                    region=tile,
                    file_per_band=False,
                )
            else:
                with self._suppress_stdout():  # ToDo: I would consider a flag for some debug mode with log file, and collect this output there then. Maybe global flag log_on for class.
                    geemap.ee_export_image(
                        image,
                        filename=output_file,
                        scale=self.meter_per_pixel,
                        region=tile,
                        file_per_band=False,
                    )

            cloud_coverage = collection.filterBounds(tile).first().get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

            metadata = {
                "coordinates": tile['coordinates'],
                "cloud_coverage": cloud_coverage,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

    def download_images_list(
        self,
        points_list: list,
        filename: str = "image_title",
        tile_size: int = 1000,
        max_cloud: float = 0.3,
        time_start: str = "2020-01-01",  # ToDo: What does exactly this dates do?
        time_end: str = "2024-12-31",  # ToDo: I think those are not the dates we wanted to have. We want to take images with timestamps between those dates. So multiple images from different dates can be downloaded.
        # I would rename variables, and add this functionality to other method, that would collect images from different dates as different files.
        meter_per_pixel: float = 5.0,
        distance_x: Union[
            int, float, list
        ] = 1000,  # ToDo: For sure in readme must be explained what the main parameters are doing.
        distance_y: Union[int, float, list] = 1000,
        use_geolocator_bbox: bool = False,
        divide_mode: str = None,
        verbose: bool = False,
    ):
        """
        Download satellite images for a list of points.

        Parameters
        ----------
        points_list : list
            A list of geographical points. Each point can be a tuple (latitude, longitude) or a string address.
        tile_size : int, optional
            The size of each tile in meters. Default is 1000.
        max_cloud : float, optional
            The maximum cloud coverage percentage allowed. Default is 0.3.
        title : str, optional
            The title of the image. Default is 'image_title'.
        meter_per_pixel : float, optional
            The resolution in meters per pixel. Default is 5.0.
        distance_x : Union[int, float, list], optional
            The distance in the x-direction in meters. Can be a single value or a list of values. Default is 1000.
        distance_y : Union[int, float, list], optional
            The distance in the y-direction in meters. Can be a single value or a list of values. Default is 1000.
        use_geolocator_bbox : bool, optional
            If True, use the geolocator to get the bounding box. Default is False.
        time_start : str, optional
            The start time for the data in 'YYYY-MM-DD' format. Default is '2020-01-01'.
        time_end : str, optional
            The end time for the data in 'YYYY-MM-DD' format. Default is '2024-12-31'.
        divide_mode : str, optional
            Divide images into train, validation and test sets. "train-val-test" or "train-val". Default is None.
        verbose : bool, optional
            If True, suppress output messages. Default is False.
        """

        if not isinstance(points_list, (list, tuple)):
            raise SatSRError("Points list must be a list")
        elif len(points_list) == 0:
            raise SatSRError("Points list cannot be empty")

        if not isinstance(distance_x, (int, float, list, tuple)):
            raise SatSRError("Distance x must be a number or a list of numbers")
        elif (
            isinstance(distance_x, (list, tuple))
            and isinstance(distance_y, (list, tuple))
            and (
                len(distance_x) != len(points_list)
                or len(distance_y) != len(points_list)
                or len(distance_x) != len(distance_y)
            )
        ):
            raise SatSRError(
                "Distance x and y must have the same length as points list, have the same length or be a single number"
            )
            
        if use_geolocator_bbox:
            for point in points_list:
                if not isinstance(point, str):
                    raise SatSRError("All points must be a string when using geolocator for bounding box")

        for idx, point in enumerate(
            tqdm(
                points_list, desc="Downloading images for points"
            )  # ToDo: Flag for tqdm. Potential log file for this.
        ):
            if isinstance(distance_x, (list, tuple)):
                dist_x = distance_x[idx]
                dist_y = distance_y[idx]
            else:
                dist_x = distance_x
                dist_y = distance_y

            self.setup_point(
                point,
                time_start=time_start,
                time_end=time_end,
                title=filename,
                meter_per_pixel=meter_per_pixel,
                distance_x=dist_x,
                distance_y=dist_y,
                use_geolocator_bbox=use_geolocator_bbox
            )
            self.download_image(
                file_name_base=filename,
                tile_size=tile_size,
                max_cloud=max_cloud,
                verbose=verbose,
            )
        if divide_mode is not None:
            divide_images(
                divide_mode=divide_mode, download_path=self.data_path, title=filename
            )

    @staticmethod
    @contextmanager
    def _suppress_stdout():
        """ "
        Context manager to suppress standard output.
        """
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                
    def create_data_loaders(self, scale, batch_size, mode, resize, path = None):
        if path is None:
            path = self.data_path
        return train_val_test_loader(path, scale, batch_size, mode, resize)
