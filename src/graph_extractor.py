from typing import Tuple, Literal, List

import numpy as np

import cv2
from cv2.typing import MatLike, Point
from skimage.morphology import skeletonize


class GraphExtractor:
    def __init__(self):
        pass

    def binearize_image(self, image: MatLike, colorful: bool) -> MatLike:
        """
        Binearize image using standard thresholding, using preprocessing and \\
        post-processing techniques.

        Parameters
        -------
        image: grayscale np.uint8 array of shape (W, H)
        """
        # remove noise
        image = cv2.GaussianBlur(image, (9, 9), 0)
        image = cv2.medianBlur(image, 15)

        # binearization
        if colorful:
            _, image = cv2.threshold(image, 123, 255, cv2.THRESH_BINARY_INV)
        else:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
            bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
            image = cv2.divide(image, bg, scale=255)
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # post-processing
        image = cv2.dilate(image, (3, 3), iterations=4)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, (9, 9), iterations=15)
        return image

    def extend_line(self, pts: np.ndarray, extension_length: int | float) -> np.ndarray:
        """
        Extends line segment in both directions

        Parameters
        -------
        pts: segment's end points of shape (2, 1, 2)
        extension_length: norm of extension in pixels (rounds to int after projection)
        """
        pt1, pt2 = pts[0][0], pts[1][0]
        line_vector = pt2 - pt1
        line_length = np.linalg.norm(line_vector)
        unit_vector = line_vector / line_length
        d = (unit_vector * extension_length).astype(np.int32)

        pt1 -= d
        pt2 += d

        return np.array([[pt1], [pt2]])

    def detect_lines(self, image: MatLike, **kwargs) -> MatLike:
        """
        Detect straight lines on the image and filter out everything \\
        except them from the image.

        Parameters:
        -------
        image: binearized np.uint8 array of shape (width, height)
        kwargs: arguments for line detection algorithm
        """
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, **kwargs)
        processed_image = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(processed_image, (x1, y1), (x2, y2), color=1, thickness=1)
        return processed_image

    def approximate_lines(self,
                          image: MatLike,
                          curvature_thres: float = 0.05,
                          extension_length: int | float = 5) -> Tuple[MatLike, List]:
        """
        Approximate and merge close lines

        Parameters:
        -------
        image: binearized np.uint8 array of shape (width, height)
        curvature_thres: threshold for approximating lines
        extension_length: norm of extension of segments in pixels
        """
        processed_image = np.zeros_like(image)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approx = []
        for contour in contours:
            epsilon = curvature_thres * cv2.arcLength(contour, True)
            appr = cv2.approxPolyDP(contour, epsilon, True)
            appr_extended = self.extend_line(appr, extension_length=extension_length)
            approx.append(appr_extended)
        cv2.polylines(processed_image, approx, isClosed=False, color=1, thickness=1)
        return processed_image, approx

    def build_adjacency_matrix(self, image: MatLike) -> Tuple[MatLike, np.ndarray]:
        """
        Create adjacency matrix from binearized image consisting of \
        intersecting segments: graph edges

        Parameters:
        -------
        image: binearized np.int8 array of shape (W, H)
        """
        # extend lines further so there is more chance they will intersect
        _, approx = self.approximate_lines(image, extension_length=30)

        # create image representation of parsed graph
        output_image = np.zeros((*image.shape, 3), dtype=np.uint8)
        cv2.polylines(output_image, approx, isClosed=False, color=(0, 255, 0), thickness=8)

        # find all intersection points and edges pairs
        intersection_pts = []
        intersecting_edges = []
        intersect_num = np.zeros(len(approx), dtype=int)
        for i in range(len(approx)):
            # add to check on leaf point
            pt1, pt2 = approx[i]
            intersection_pts.append(np.array(pt1[0]))
            intersection_pts.append(np.array(pt2[0]))
            intersecting_edges.append((i, ))
            intersecting_edges.append((i, ))

            # check for intersections
            for j in range(i + 1, len(approx)):
                pt = self.find_intersection(approx[i], approx[j])
                if pt is not None:
                    intersection_pts.append(pt)
                    intersecting_edges.append((i, j))
                    intersect_num[i] += 1
                    intersect_num[j] += 1

        # match similar intersection points into one vertex
        vertices = {}
        for i, pt in enumerate(intersection_pts):
            is_unique = True
            for v in vertices.keys():
                if np.linalg.norm(pt - v) < 50:
                    is_unique = False
                    vertices[v].update(intersecting_edges[i])
                    break
            if is_unique:
                vertices[tuple(pt)] = set(intersecting_edges[i])
                cv2.circle(output_image, pt, radius=20, color=(0, 0, 255), thickness=-1)

        # build adjacency matrix
        adj_matrix = np.zeros((len(vertices), ) * 2, dtype=bool)
        for i, v1_edges in enumerate(vertices.values()):
            for j, v2_edges in enumerate(vertices.values()):
                if len(v1_edges & v2_edges) > 0:
                    adj_matrix[i][j] = True

        return output_image, adj_matrix

    def processing_step(self, image: MatLike, extension_length: int | float = 5) -> MatLike:
        """
        Make a processing step, improving graph representation by \
        adjusting parts of the same edge, approximating curves \
        with straight lines and removing noise from the image

        Parameters
        -------
        image: skeleton image of the graph, np.uint8 array of shape (W, H)
        extension_length: norm of extension of segments in pixels
        """
        image = self.detect_lines(image,
                                  threshold=30,
                                  minLineLength=30,
                                  maxLineGap=30)
        image, _ = self.approximate_lines(image,
                                          curvature_thres=0.06,
                                          extension_length=extension_length)
        return image

    def process_image(self, image: MatLike, steps: int = 2) -> Tuple[MatLike, np.ndarray]:
        """
        Extract graph from the image as adjacency matrix

        Parameters:
        -------
        image: BGR image of shape (W, H, 3)
        steps: number of iterations for approximating graph on the image.\
        The increase of steps usually leads to better results, but can affect \
        performance.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, std = cv2.meanStdDev(hsv_image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = cv2.split(image)[1]

        image = self.binearize_image(image, colorful=(std[1] > 40))
        image = skeletonize(image).astype(np.uint8)

        line_ext = [5, 5, 5]  # line extensions
        for i in range(steps):
            image = self.processing_step(image, line_ext[i])

        return self.build_adjacency_matrix(image)

    @staticmethod
    def find_intersection(line1: np.ndarray, line2: np.ndarray) -> Point | None:
        """
        Function that checks whether two segments are intersecting or not, \
        and returns point of intersection if found any. \\
        Both parameters have shapes (2, 1, 2) where dim 1 represents two points, \\
        and dim 3 represents x and y respectively.
        """
        (x1, y1), (x2, y2) = line1[0][0], line1[1][0]
        (x3, y3), (x4, y4) = line2[0][0], line2[1][0]

        # lines are represented as a*x + b*y = c
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        determinant = a1 * b2 - a2 * b1

        if determinant != 0:
            intersect_x = int((b2 * c1 - b1 * c2) / determinant)
            intersect_y = int((a1 * c2 - a2 * c1) / determinant)

            # chech if the intersection points lies on both segments
            if (min(x1, x2) <= intersect_x <= max(x1, x2) and
                min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and
                min(y3, y4) <= intersect_y <= max(y3, y4)):
                return np.array([intersect_x, intersect_y])
            else:
                return None

        # lines are parallel
        return None
