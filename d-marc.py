import solid as sl
import numpy as np
from numpy import pi
import os.path as path

KEYSWITCH_WIDTH = 14.4

#TODO: thumb cluster, wall / supports, wrist rest

# TODO: function to allign cols to a chosen row and vert align_cols(row, vert)
# would stretch out some of the more compressed rows

#TODO: function to stagger a column, like raise the pinky column up
# would need to calculate the change in the original z vector, [0,0,1]
# due to rotations to the key so we are going up in relation to the
# direction of the keys. Try only considering Y roatations to start
# considering x rotations may compress the column which we do not want

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


class Rotation:
    def __init__(self, deg, axis):
        self.degree = deg
        self.axis = axis

    def rotate(self, shape):
        return sl.rotate(self.degree, self.axis)(shape)


class KeyCap:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def shape(self):
        return sl.cube([self.width, self.width, self.height], center=True)


class KeyHole:
    def __init__(self, width, thickness, hotswap_socket=None,
                 cap_height=None, switch_height=0, cylinder_segments=100,
                 side="right", vert_colors=None):
        self.rotations = []
        self.width = width
        self.height = width
        self.thickness = thickness
        # keyboard should decide which sockets to use and import them
        self.hotswap_socket = hotswap_socket
        self.cylinder_segments = cylinder_segments
        self.cap_height = cap_height
        self.cap = None
        self.switch_height = switch_height
        self.side = side
        self.vert_colors = vert_colors
        if self.vert_colors is None or len(self.vert_colors) != 4:
            self.vert_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]
        self.vertices = self.origin_vertices()
        self.vert_shapes = self.vertices_shape()
        self.unrotated_verts = self.origin_vertices()
        self.origin = [0, 0, 0]
        self.unrotated_origin = [0, 0, 0]
        self.shape = self.init_keyhole()

    def shape(with_vertices=False):
        return

    # generate a cube for each of the key's current vertices
    # applies the colors in self.vert_colors
    def vertices_shape(self, idxs=[0, 1, 2, 3], rotate=True):
        verts = []
        for idx in idxs:
            vert = self.vertices[idx]
            color = self.vert_colors[idx]
            shape = sl.cube([1,1,self.thickness], center=True)
            if rotate:
                for rot in self.rotations:
                    shape = rot.rotate(shape)
            verts.append(sl.color(color)(sl.translate(vert)(shape)))
        return verts

    # repetative with the above code
    def unrotated_origin_verts(self):
        verts = []
        for idx in range(len(self.unrotated_verts)):
            vert = self.vertices[idx]
            color = self.vert_colors[idx]
            verts.append(sl.color(color)(sl.translate(vert)(sl.cube(1))))
        return sl.union()(*verts)

    def origin_marker(self):
        return sl.color([1, 1, 1])(sl.translate(self.origin)(sl.cube(1)))

    def unrotated_origin_marker(self):
        return sl.color([0, 0, 0])(sl.translate(self.origin)(sl.cube(1)))

    def unrotated_verts_shape(self):
        verts = []
        for idx in range(len(self.unrotated_verts)):
            vert = self.unrotated_verts[idx]
            color = self.vert_colors[idx]
            verts.append(sl.color(color)(sl.translate(vert)(sl.cube(1))))
        return sl.union()(*verts)

    # generates the keyhole shape and places it at the origin
    def init_keyhole(self):
        top_wall = sl.cube([self.width + 3, 1.5, self.thickness],
                           center=True)
        top_wall = sl.translate(
            (0, (1.5 / 2) + (self.height / 2), self.thickness / 2)
        )(top_wall)

        left_wall = sl.cube([1.5, self.height + 3, self.thickness],
                            center=True)
        left_wall = sl.translate(
            ((1.5 / 2) + (self.width / 2), 0, self.thickness / 2)
        )(left_wall)

        side_nub = sl.cylinder(1, 2.75, segments=self.cylinder_segments,
                               center=True)
        side_nub = sl.rotate(rad2deg(pi / 2), [1, 0, 0])(side_nub)
        side_nub = sl.translate((self.width / 2, 0, 1))(side_nub)
        nub_cube = sl.cube([1.5, 2.75, self.thickness], center=True)
        nub_cube = sl.translate(
            ((1.5 / 2) + (self.width / 2), 0, self.thickness / 2)
        )(nub_cube)

        side_nub = sl.hull()(side_nub, nub_cube)

        plate_half1 = top_wall + left_wall + side_nub
        plate_half2 = plate_half1
        plate_half2 = sl.mirror([0, 1, 0])(plate_half2)
        plate_half2 = sl.mirror([1, 0, 0])(plate_half2)

        plate = plate_half1 + plate_half2

        if self.hotswap_socket is not None:
            self.hotswap_socket = sl.translate([0, 0, self.thickness])(
                self.hotswap_socket)
            plate = sl.union()(plate, self.hotswap_socket)

        if self.cap_height is not None:
            cap = sl.cube([self.width, self.width, self.cap_height],
                          center=True)
            self.cap = sl.translate([0, 0,
                                     (
                                         self.thickness + self.switch_height + self.thickness)])(
                cap)

        if self.side == "left":
            plate = sl.mirror([-1, 0, 0])(plate)

        return plate

    # get the vertices when the key is at the origin / first created
    # without any rotations applied
    def origin_vertices(self):
        hw = (self.width + self.thickness) / 2
        hh = (self.width + self.thickness) / 2
        th = self.thickness / 2
        # th = 0
        verts = np.array([
            [-hw + 1, hh - 1, th],
            [hw - 1, hh - 1, th],
            [-hw + 1, -hh + 1, th],
            [hw - 1, -hh + 1, th]
        ])
        # if self.cap_height is not None:
        #     for idx in range(len(verts)):
        #         verts[idx] = verts[idx] - [0,0,(
        #                               self.thickness + self.switch_height + self.thickness)]
        return verts

    def translate(self, xyz, move_unrot=True):
        for idx in range(len(self.vertices)):
            self.vertices[idx] += xyz
            self.vert_shapes[idx] = sl.translate(xyz)(self.vert_shapes[idx])

        if move_unrot:
            for idx in range(len(self.unrotated_verts)):
                self.unrotated_verts[idx] += xyz
        if self.cap is not None:
            self.cap = sl.translate(xyz)(self.cap)
        self.origin += xyz
        self.shape = sl.translate(xyz)(self.shape)

    def rotate_about_x(self, degree):
        for idx in range(len(self.vertices)):
            self.vertices[idx] = self.rotate_around_x(self.vertices[idx],
                                                      degree)
        self.origin = self.rotate_around_x(self.origin, degree)
        self.shape = sl.rotate(degree, [1, 0, 0])(self.shape)
        if self.cap is not None:
            self.cap = sl.rotate(degree, [1, 0, 0])(self.cap)
        self.rotations.append(Rotation(degree, [1,0,0]))


    def rotate_about_y(self, degree):
        for idx in range(len(self.vertices)):
            self.vertices[idx] = self.rotate_around_y(self.vertices[idx],
                                                      degree)
        self.origin = self.rotate_around_y(self.origin, degree)
        self.shape = sl.rotate(degree, [0, 1, 0])(self.shape)
        if self.cap is not None:
            self.cap = sl.rotate(degree, [0, 1, 0])(self.cap)
        self.rotations.append(Rotation(degree, [0,1,0]))

    def translate_to_vert(self, vert_number):
        vert = self.vertices[vert_number]
        self.translate(-vert)

    def rotate_around_vert(self, vert_num, degree, axis):
        self.translate_to_vert(vert_num)
        if axis == 'x':
            self.rotate_about_x(degree)
        elif axis == 'y':
            self.rotate_about_y(degree)

    def move_to_unrotated_vert(self, vert_num):
        vert = self.vertices[vert_num]
        unrot_vert = self.unrotated_verts[vert_num]
        move_to = unrot_vert - vert
        self.translate(move_to, move_unrot=False)

    @staticmethod
    def rotate_around_x(position, angle):
        angle = deg2rad(angle)
        t_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
        return np.matmul(t_matrix, position)

    @staticmethod
    def rotate_around_y(position, angle):
        angle = deg2rad(angle)
        t_matrix = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        return np.matmul(t_matrix, position)

    def get_shape(self, with_verts=False):
        if with_verts:
            shape = sl.union()(self.shape,
                               self.vertices_shape(),
                               self.unrotated_verts_shape(),
                               self.origin_marker(),
                               self.unrotated_origin_marker())
            if self.cap is not None:
                return sl.union()(shape, self.cap)
            return shape

        return self.shape

    def save_to_file(self, file_path, with_verts=False):
        shape = self.get_shape(with_verts=with_verts)
        sl.scad_render_to_file(shape, file_path)


class Keyboard:
    def __init__(self, nrows, ncols, row_degree, col_degree, tent_angle,
                 tilt_angle, cap_height, switch_height, keyhole_width=14.4, \
                 keyhole_thickness=4):
        self.nrows = nrows
        self.ncols = ncols
        self.cap_height = cap_height
        self.switch_height = switch_height
        self.row_degree = row_degree
        self.col_degree = col_degree
        self.tent_angle = tent_angle
        self.tilt_angle = tilt_angle
        self.row_degrees = [x + tent_angle for x in np.linspace(0, row_degree,
                                                                nrows)]
        self.col_degrees = [x + tilt_angle for x in np.linspace(0, col_degree,
                                                                ncols)]

        self.keyhole_width = keyhole_width
        self.keyhole_thickness = keyhole_thickness
        self.row_radius, self.col_radius = self.calc_radius()
        self.shape = None
        self.key_array = self.init_key_array()

    def init_key_array(self):
        rows = []
        for row in range(self.nrows):
            rows.append([None for x in range(self.ncols)])
        return rows

    def calc_key_pos(self, row_degree, col_degree):
        x = self.col_radius * cosine(col_degree)
        y = self.row_radius * cosine(row_degree)
        z = self.row_radius * sine(row_degree) + self.col_radius * sine(
            col_degree)
        return x, y, z

    def keys_hull(self):
        keys = []
        for row in self.key_array:
            for key in row:
                keys.append(key.get_shape())
        return sl.hull()(*keys)

    def place_key(self, row, col):
        row_degree, col_degree = self.get_key_degrees(row, col)
        x_rot, y_rot = rotation_degree(row_degree), rotation_degree(col_degree)
        x, y, z = self.calc_key_pos(row_degree, col_degree)

        # y = -(self.nrows - row) * (self.keyhole_width +
        #                            self.keyhole_thickness+
        #                            self.switch_height)
        # x = (self.ncols - col) * (self.keyhole_width +
        #                           self.keyhole_thickness +
        #                           self.switch_height)

        key = KeyHole(self.keyhole_width, self.keyhole_thickness,
                      cap_height=self.cap_height,
                      switch_height=self.switch_height)

        # if (x_rot >= 0):
        #     key.translate(-key.vertices[3])
        # elif x_rot < 0:
        #     key.translate(-key.vertices[0])

        key.rotate_about_x(x_rot)
        key.rotate_about_y(y_rot)

        if (x_rot >= 0):
            key.move_to_unrotated_vert(3)
        elif (x_rot < 0):
            key.move_to_unrotated_vert(1)

        key.translate([x, y, z])
        self.key_array[row][col] = key

    # 0 for origin, 1 for red, 2 for green, 3 for blue, 4 for purple
    def get_key_marker_pos(self, marker_num):
        key_pos = np.zeros((3, self.nrows, self.ncols))
        for row in range(self.nrows):
            for col in range(self.ncols):
                key = self.key_array[row][col]
                if marker_num == 0:
                    marker = key.origin
                else:
                    marker = key.vertices[marker_num - 1]
                x, y, z = marker[0], marker[1], marker[2]
                key_pos[0, row, col] = x
                key_pos[1, row, col] = y
                key_pos[2, row, col] = z
        return key_pos

    def align_column_marker(self, marker):
        key_pos = self.get_key_marker_pos(marker)
        key_x, key_z = key_pos[0], key_pos[2]
        key_x = key_x.T
        min_x = np.transpose(key_x.min(axis=1))
        min_x = min_x.reshape(1, 6).T
        x_shift = key_x - min_x
        for row in range(self.nrows):
            for col in range(self.ncols):
                key = self.key_array[row][col]
                key.translate([x_shift[col][row], 0, 0])

    def place_keys(self):
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.place_key(row, col)

    def offset_row(self, nrow, offset):
        for key in self.key_array[nrow]:
            key.translate(offset)

    def get_key_degrees(self, row, col):
        return self.row_degrees[row], self.col_degrees[col]

    def calc_radius(self):
        row_len = (self.keyhole_width) * self.nrows
        col_len = (self.keyhole_width) * self.ncols
        row_radius = row_len / deg2rad(self.row_degree)
        col_radius = col_len / deg2rad(self.col_degree)
        if self.cap_height is not None:
            row_radius += self.cap_height + self.switch_height
            col_radius += self.cap_height + self.switch_height - 30
        return row_radius, col_radius

    def get_shape(self, with_markers=False):
        shapes = []
        for row in self.key_array:
            for key in row:
                if key is not None:
                    shapes.append(key.get_shape(with_verts=with_markers))
        return sl.union()(*shapes)

    def move_keys_to_unrot(self, vert):
        for row in self.key_array:
            for key in row:
                key.move_to_unrotated_vert(vert_num=vert)

    def create_hull_cube(self, xyz):
        return sl.translate(xyz)(sl.cube(self.keyhole_thickness))

    def col_key_connector(self, row1, row2, col):
        verts = []
        key1 = self.key_array[row1][col]
        key2 = self.key_array[row2][col]
        verts.append(key1.vertices_shape(idxs=[0, 1]))
        verts.append(key2.vertices_shape(idxs=[2, 3]))
        if col < self.ncols-1:
            verts.append(self.key_array[row1][col+1].vertices_shape(idxs=[1]))
            verts.append(self.key_array[row2][col+1].vertices_shape(idxs=[3]))
        return sl.hull()(*verts)

    def row_key_connector(self, col1, col2, row):
        key1 = self.key_array[row][col1]
        key2 = self.key_array[row][col2]
        verts1 = key1.vertices_shape(idxs=[0,2])
        verts2 = key2.vertices_shape(idxs=[1,3])
        return sl.hull()(verts1, verts2)

    def col_key_connectors(self):
        hull = []
        for col in range(self.ncols):
            for row in range(self.nrows-1):
                hull.append(self.col_key_connector(row, row+1, col))
        return sl.union()(*hull)

    def row_key_connectors(self):
        hull = []
        for row in range(self.nrows):
            for col in range(self.ncols - 1):
                hull.append(self.row_key_connector(col, col+1, row))
        return hull

    def allign_keys(self, key1, key2):
        pass

    def allign_rows(self, row1, row2):
        pass

    def shape_to_file(self, filepath, with_markers=False):
        shape = self.get_shape(with_markers=with_markers)
        con = [self.row_key_connectors(), self.col_key_connectors()]
        sl.scad_render_to_file(sl.union()(shape, con),
                               filepath)
        # sl.scad_render_to_file(self.keys_hull(), filepath)

    def shell_block(self, x_steps, y_steps):
        row_degs = [x + self.tent_angle for x in np.linspace(0, self.row_degree,
                                                             x_steps)]
        col_degs = [x + self.tilt_angle for x in np.linspace(0, self.col_degree,
                                                             y_steps)]
        width = self.row_radius / x_steps
        height = self.col_radius / y_steps
        blocks = []
        for row_deg in row_degs:
            for col_deg in col_degs:
                x_rot, y_rot = rotation_degree(row_deg), rotation_degree(
                    col_deg)
                block = sl.cube([4 * width, 2 * height, self.keyhole_thickness],
                                center=True)
                x, y, z = self.calc_key_pos(row_deg, col_deg)
                block = sl.rotate(x_rot, [1, 0, 0])(block)
                block = sl.rotate(-y_rot, [0, 1, 0])(block)
                x, y, z = self.calc_key_pos(row_deg, col_deg)
                block = sl.translate([x, y, z])(block)
                blocks.append(block)
        return sl.union()(*blocks)


def rotation_degree(degree):
    if degree < 180:
        return 90.0 - degree
    return degree - 270.0


if __name__ == '__main__':
    plate_file = path.join("..", "src", r"hot_swap_plate.stl")
    socket = sl.import_(plate_file)
    key = KeyHole(14.4, 4, cap_height=8, switch_height=4, hotswap_socket=socket)
    # key.translate(-key.vertices[0])
    key.rotate_about_x(-50)
    # key.translate(key.vertices[0])
    key.rotate_about_y(40)
    # print(key.vertices)
    key.save_to_file(r"single_key.scad", with_verts=True)
    keyboard = Keyboard(4, 6, 100, 50, 200, 20, 8, 4)
    # print(keyboard.row_degrees)
    # print(keyboard.col_degrees)
    # print(keyboard.calc_radius())
    keyboard.place_keys()
    # keyboard.move_keys_to_unrot(0)
    keyboard.align_column_marker(4)
    keyboard.offset_row(2, [1, 1, 7])
    keyboard.offset_row(0, [2, 1, -2])
    keyboard.offset_row(1, [-2, 2, 0])
    # print(keyboard.get_key_marker_pos(0))

    # shell = keyboard.shell_block(50, 50)
    # sl.scad_render_to_file(sl.union()(keyboard.get_shape(), shell),
    #                        "shell.scad")

    keyboard.shape_to_file(r"keyboard.scad", with_markers=True)
