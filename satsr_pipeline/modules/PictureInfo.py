import numpy as np
import rasterio
import rasterio.transform


# ToDo: What is self.shape?
# ToDo: Meter_per_pixel and bands are not used in the class. Remove them if not needed.
class PictureInfo:
    def __init__(
        self,
        box: dict[str, float],
        meter_per_pixel: float = 5.0,
        bands: list = ["B4", "B3", "B2"],
    ):
        """
        Initialize the class with given parameters.

        Parameters
        ----------
        box : dict[str, float]
            The bounding box of the area with keys 'left', 'bottom', 'right', 'top'.
        meter_per_pixel : float, optional
            The resolution in meters per pixel. Default is 5.0.
        bands : list, optional
            The list of bands to be used. Default is ['B4', 'B3', 'B2'].
        """
        self.box = box
        self.meter_per_pixel = meter_per_pixel
        self.bands = bands

    @property
    def bounding_box_coordinates(self) -> list[tuple[float, float]]:
        """
        Get the geographical points of the bounding box.

        Returns
        -------
        list of tuple[float, float]
            A list of tuples representing the geographical points (longitude, latitude) of the bounding box.
        """
        return [
            (self.box["left"], self.box["bottom"]),
            (self.box["left"], self.box["top"]),
            (self.box["right"], self.box["top"]),
            (self.box["right"], self.box["bottom"]),
        ]

    @property
    def _box_tuple(self):
        """
        Get the bounding box as a tuple.

        Returns
        -------
        tuple
            A tuple representing the bounding box (left, bottom, right, top).
        """
        return (
            self.box["left"],
            self.box["bottom"],
            self.box["right"],
            self.box["top"],
        )

    @property
    def box_corners(self):
        """
        Get the corners of the bounding box.

        Returns
        -------
        list of tuple
            A list of tuples representing the corners (longitude, latitude) of the bounding box.
        """
        corners = [
            (self.box["left"], self.box["bottom"]),
            (self.box["right"], self.box["bottom"]),
            (self.box["right"], self.box["top"]),
            (self.box["left"], self.box["top"]),
        ]
        return corners

    @property
    def coordinates(self):
        """
        Get the coordinates of the bounding box.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing two numpy arrays:
            - x: The x-coordinates (longitude) of the bounding box.
            - y: The y-coordinates (latitude) of the bounding box.
        """
        y, x = map(range, self.shape)
        affine_transform = rasterio.transform.from_bounds(*self._box_tuple, *self.shape)

        # we need to calculate 2 times in case of not square picture
        x = np.array(rasterio.transform.xy(affine_transform, [0] * len(x), x)[0])
        y = np.array(rasterio.transform.xy(affine_transform, y, [0] * len(y))[1])
        return x, y
