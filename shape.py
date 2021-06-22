import math
from abc import abstractmethod
import solid as sl
import numpy as np
from numpy import pi

# convert degrees to radians
def deg2rad(degrees: float) -> float:
    return degrees * pi / 180


# convert radians to degrees
def rad2deg(rad: float) -> float:
    return rad * 180 / pi


def cosine(degree):
    return np.cos(deg2rad(degree))


def sine(degree):
    return np.sin(deg2rad(degree))


def execute_actions(obj, actions, action_dict):
    for action in actions:
        for act, param in action.items():
            if act in action_dict:
                act_fun = action_dict[act]
                obj = act_fun(obj, param)
    return obj


class Coordinate:
    @staticmethod
    def translate(coord, xyz):
        return np.add(coord, xyz)

    @staticmethod
    def rotate_x(coord, deg):
        rads = deg2rad(deg)
        t_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rads), -np.sin(rads)],
                [0, np.sin(rads), np.cos(rads)],
            ]
        )
        return np.matmul(t_matrix, coord)

    @staticmethod
    def rotate_z(coord, deg):
        rads = deg2rad(deg)
        t_matrix = np.array(
            [
                [np.cos(rads), -np.sin(rads), 0],
                [np.sin(rads), np.cos(rads), 0],
                [0, 0, 1],
            ]
        )
        return np.matmul(t_matrix, coord)

    @staticmethod
    def rotate_y(coord, rads):
        rads = deg2rad(rads)
        t_matrix = np.array(
            [
                [np.cos(rads), 0, np.sin(rads)],
                [0, 1, 0],
                [-np.sin(rads), 0, np.cos(rads)],
            ]
        )
        return np.matmul(t_matrix, coord)

    @staticmethod
    def _rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    @staticmethod
    def rotate(coord, degs):
        xdeg, ydeg = degs
        yrads = deg2rad(ydeg)
        coord = Coordinate.rotate_x(coord, xdeg)
        y_rot_vec = Coordinate.rotate_x([0, 1, 0], xdeg)
        return np.dot(Coordinate._rotation_matrix(y_rot_vec, yrads), coord)


class Shape:
    def __init__(self):
        self.actions = []
        self.functions = {
            "translate": Shape._translate,
            "rotate_x": Shape._rotate_x,
            "rotate_y": Shape._rotate_y,
            "rotate_z": Shape._rotate_z,
            "rotate": Shape._rotate
        }
        self.coord_functions = {
            "translate": Coordinate.translate,
            "rotate_x": Coordinate.rotate_x,
            "rotate_y": Coordinate.rotate_y,
            "rotate_z": Coordinate.rotate_z,
            "rotate": Coordinate.rotate
        }

    def shape(self):
        shape = self.init_shape()
        return execute_actions(shape, self.actions, self.functions)

    def vertice(self, idx):
        vert = self.init_vertices()[idx]
        return execute_actions(vert, self.actions, self.coord_functions)

    def vertices(self, idxs=None):
        verts = self.init_vertices()
        if idxs is not None:
            verts = [verts[idx] for idx in idxs]
        return [execute_actions(vert, self.actions, self.coord_functions)
                for vert in verts]

    def vertice_shape(self, idx):
        shapes = self.init_vertice_shapes()
        if shapes:
            shape = shapes[idx]
        else:
            coord = self.init_vertices()[idx]
            shape = sl.translate(coord)(self.init_vertice_shape())
        colors = self.vert_colors()
        if colors:
            color = colors[idx]
        else:
            color = [1, 0, 0]
        shape = sl.color(color)(shape)
        return execute_actions(shape, self.actions, self.functions)

    def vertice_shapes(self, idxs=None):
        shapes = self.init_vertice_shapes()
        if shapes:
            idxs = range(len(shapes))
        else:
            if idxs is None:
                idxs = range(len(self.init_vertices()))
        shapes = []
        for idx in idxs:
            shapes.append(self.vertice_shape(idx))
        return sl.union()(shapes)

    def origin(self):
        return execute_actions(self.init_origin(),
                               self.actions,
                               self.coord_functions)

    def translate(self, xyz):
        self.add_action("translate", xyz)
        return self

    def rotate(self, xdeg, ydeg):
        self.add_action("rotate", [xdeg, ydeg])
        return self


    def rotate_x(self, deg):
        self.add_action("rotate_x", deg)
        return self


    def rotate_y(self, deg):
        self.add_action("rotate_y", deg)
        return self


    def rotate_z(self, deg):
        self.add_action("rotate_z", deg)
        return self


    def add_action(self, t: str, args):
        self.actions.append({t: args})

    @abstractmethod
    def init_shape(self):
        """return the intial shape"""
        pass

    def init_vertices(self):
        """return coordinates for the shape's vertices"""
        raise NotImplementedError("this Shape does not implement vertices")

    def init_vertice_shape(self):
        """return the vertice shape"""
        raise NotImplementedError("this Shape does not implement vert shapes")

    def init_vertice_shapes(self):
        return False

    def vert_colors(self):
        """return the vertice shape colors"""
        return False

    def init_origin(self):
        """return the inital coordinate for the shape"""
        raise NotImplementedError("this Shape does not implement an origin")

    @staticmethod
    def _translate(shape, xyz):
        return sl.translate(xyz)(shape)

    @staticmethod
    def _rotate_x(shape, deg):
        return sl.rotate(deg, [1, 0, 0])(shape)

    @staticmethod
    def _rotate_y(shape, deg):
        return sl.rotate(deg, [0, 1, 0])(shape)

    @staticmethod
    def _rotate_z(shape, deg):
        return sl.rotate(deg, [0, 0, 1])(shape)

    @staticmethod
    def _rotate(shape, degs):
        """rotate shape by [x,y]"""
        xdeg, ydeg = degs
        shape = sl.rotate(xdeg, [1, 0, 0])(shape)
        y_rot_vec = Coordinate.rotate_x([0, 1, 0], xdeg)
        return sl.rotate(ydeg, y_rot_vec)(shape)
