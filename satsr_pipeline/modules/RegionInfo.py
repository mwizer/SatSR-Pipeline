import numpy as np
import rasterio
import rasterio.transform


class RegionInfo:
    def __init__(
        self,
        box: dict[str, float],
    ):
        """
        Initialize the class with given parameters.

        Parameters
        ----------
        box : dict[str, float]
            The bounding box of the area with keys 'left', 'bottom', 'right', 'top'.
        """
        self.box = box

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

