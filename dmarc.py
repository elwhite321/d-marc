from copy import deepcopy
import solid as sl
import numpy as np
from numpy import pi
import os.path as path
from shape import Shape, rad2deg
from joint import Joint


KEYSWITCH_WIDTH = 14.4


# TODO: thumb cluster, wall / supports, wrist rest

# TODO: function to allign cols to a chosen row and vert align_cols(row, vert)
# would stretch out some of the more compressed rows

# TODO: function to stagger a column, like raise the pinky column up
# would need to calculate the change in the original z vector, [0,0,1]
# due to rotations to the key so we are going up in relation to the
# direction of the keys. Try only considering Y roatations to start
# considering x rotations may compress the column which we do not want

class KeyCapShape(Shape):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def init_shape(self):
        return sl.cube(self.dims, center=True)

    def init_origin(self):
        return np.array([0, 0, 0])

    def init_vertices(self):
        w, l, h = self.dims
        half_width = w / 2
        half_len = l / 2
        half_height = h / 2
        adj = .5
        return np.array([
            [half_width - adj, half_len - adj, -half_height + adj],
            [half_width - adj, -half_len + adj, -half_height + adj],
            [-half_width + adj, half_len - adj, -half_height + adj],
            [-half_width + adj, -half_len + adj, -half_height + adj],
            [half_width - adj, half_len - adj, half_height - adj],
            [half_width - adj, -half_len + adj, half_height - adj],
            [-half_width + adj, half_len - adj, half_height - adj],
            [-half_width + adj, -half_len + adj, half_height - adj]
        ])

    def init_vertice_shape(self):
        return sl.cube([1, 1, 1], center=True)

    def vert_colors(self):
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]


class SocketSkirt(Shape):
    def __init__(self, dims, switch_width, axis, deg):
        super().__init__()
        self.dims = dims

        x_trans = switch_width / 2 + dims[0] / 2 - 1
        self.rotate_y(deg)
        self.translate([x_trans, 0, -1])
        rot_deg = 90 * axis
        self.rotate_z(rot_deg)

    def init_vertices(self):
        w, l, h = self.dims
        hw, hl, hh = w / 2, l / 2, h / 2
        adj = 1
        return [
            [hw - adj, hl - adj, -adj],
            [hw - adj, -hl, -adj],
            [-hw, -hl, -adj],
            [-hw, hl-adj, -adj]
        ]

    def init_vertice_shape(self):
        return sl.cube([1, 1, self.dims[2]])

    def vert_colors(self):
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]

    def init_origin(self):
        return [0, 0, 0]

    def init_shape(self):
        return sl.cube(self.dims, center=True)


def SocketJoint(joint, key, axis, flip=False, trans=[0,0,0], hull=None):
    rot_deg = 90*axis
    if flip:
        joint.rotate_z(90)
    x_trans = key.width / 2 + joint.dim[0] / 2 + key.thickness/4 + .5
    joint.translate(np.add([x_trans, 0, 0], trans))
    joint.rotate_z(rot_deg)
    joint.actions += key.actions
    if hull:
        sock_axis_verts = [(1,3), (0,1), (0,2), (0,2)]
        joint_axis_verts = [3,3,0,3]
        if hull == "top":
            jhull = joint.top()
        elif hull=="bottom":
            jhull = joint.bottom()
        else:
            return joint, None
        hull = sl.difference()(sl.hull()(
            jhull.vertice_shape(joint_axis_verts[axis]),
            key.vertice_shapes(idxs=sock_axis_verts[axis])), joint.shape())

    return joint, hull

class SocketArea(Shape):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def init_shape(self):
        return sl.color([0,0,1], alpha=0.4)(sl.cube(self.dim, center=True))


class SwitchSocket(Shape):
    def __init__(self, dim, hot_swap=None, skirt=(0, 0, 0, 0),
                 cylinder_segments=100, side="right"):
        super().__init__()
        self.width, self.height, self.thickness = dim
        self.skirts = skirt
        self.cylinder_segments = cylinder_segments
        self.side = side
        self.hotswap_socket = hot_swap

    def init_shape(self):
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
            self.hotswap_socket = sl.translate((0, 0, self.thickness))(
                self.hotswap_socket)
            plate = sl.union()(plate, self.hotswap_socket)

        if self.side == "left":
            plate = sl.mirror([-1, 0, 0])(plate)

        return plate

    def init_origin(self):
        return np.array([0, 0, 0])

    def init_vertices(self):
        hw = (self.width + self.thickness) / 2
        hh = (self.width + self.thickness) / 2
        th = self.thickness / 2
        adj = 1
        return np.array([
            [-hw + adj, hh - adj, th],
            [hw - adj, hh - adj, th],
            [-hw + adj, -hh + adj, th],
            [hw - adj, -hh + adj, th]
        ])

    def init_vertice_shape(self):
        return sl.cube([1, 1, self.thickness - 1], center=True)

    def vert_colors(self):
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]]

    def area(self):
        w,l,h = self.width, self.height, self.thickness
        # this area is flush with the top of the socket
        # area = SocketArea([w, l, h*2])

        area = SocketArea([w, l, h*3])
        area.actions = self.actions

        swap_area = SocketArea([w+h+.5,10,4.5])
        swap_area.translate(np.add(-self.init_vertices()[3], [(w)/2+.5,-4.4,
                                                              -.25]))
        swap_area.actions += self.actions

        return sl.union()(area.shape(), swap_area.shape())

    @staticmethod
    def face_vertice_idxs(axis):
        face_vertices = [(0,1), (1,3), (2,3), (0,2)]
        return face_vertices[axis]

    def face_vertices(self, axis):
        return self.vertices(idxs=self.face_vertice_idxs(axis))

    def face_vertice_shapes(self, axis):
        return self.vertice_shapes(idxs=self.face_vertice_idxs(axis))

class KeysGrid(Shape):
    def __init__(self, nrows, ncols, row_deg, col_deg, row_deg_steps,
                 col_deg_steps, cap_dim, switch_dim, switch_height, spacing=1,
                 hot_swap=True, row_degs=None, col_degs=None):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.spacing = spacing
        self.row_deg = row_deg
        self.col_deg = col_deg
        self.cap_dim = cap_dim
        self.switch_dim = switch_dim
        self.switch_height = switch_height
        self.skirts = None
        self.skirt_hulls = None
        self.hot_swap = None
        if hot_swap:
            self.hot_swap = self.hotswap_socket()

        self.row_degs = row_degs
        if row_degs is None or len(row_degs) != nrows:
            self.row_degs = self._deg_steps(row_deg, nrows, row_deg_steps)

        self.col_degs = col_degs
        if col_degs is None or len(col_degs) != ncols:
            self.col_degs = self._deg_steps(col_deg, ncols, col_deg_steps)
        self.caps = self._init_array(nrows, ncols)
        self.switches = self._init_array(nrows, ncols)
        self.skirts = []
        self._place_caps_and_switches()

    @staticmethod
    def _init_array(rows, cols):
        return [[None for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _deg_steps(deg, steps, step_size):
        return [deg + step * step_size for step in range(steps)]

    def _align_xyz(self, cap1, cap2, axis):
        if axis == 1:
            verts1, verts2 = (0, 2, 4, 6), (1, 3, 5, 7)
        else:
            verts1, verts2 = (0, 1, 4, 5), (2, 3, 6, 7)
        coords1 = np.array(cap1.vertices(idxs=verts1))
        coords2 = np.array(cap2.vertices(idxs=verts2))
        # get the closest vertice
        vert_idx = np.argmin(np.abs(np.subtract(coords1[:, axis],
                                                coords2[:, axis])))
        vec = np.subtract(coords1[vert_idx], coords2[vert_idx])
        # if axis == 0:
        #     vec[axis] = vec[axis] - self.spacing
        return vec

    def _align_caps(self, cap1, cap2, axis):
        """translates cap2 to cap1"""
        xyz = self._align_xyz(cap1, cap2, axis)
        cap2.translate(xyz)

    def align_row(self, row):
        cap = self.caps[row][0]
        switch = self.switches[row][0]
        for col in range(1, self.ncols):
            if row != 0:
                # align the columns
                xyz = self._align_xyz(self.caps[(row - 1)][col - 1], cap, 0)
                cap.translate(xyz)
                switch.translate(xyz)
            align_cap = self.caps[row][col]
            align_switch = self.switches[row][col]
            xyz = self._align_xyz(cap, align_cap, 1)
            align_cap.translate(xyz)
            align_switch.translate(xyz)
            cap = align_cap
            switch = align_switch
        if row != 0:
            # align the columns
            xyz = self._align_xyz(self.caps[(row - 1)][col], cap, 0)
            cap.translate(xyz)
            switch.translate(xyz)

    def _switch_hull(self, switch1, switch2, axis):
        if axis == 1:
            verts1, verts2 = (0, 1), (2, 3)
        else:
            verts1, verts2 = (1, 3), (0, 2)
        vert_shapes1 = switch1.vertice_shapes(idxs=verts1)
        vert_shapes2 = switch2.vertice_shapes(idxs=verts2)
        return sl.hull()(vert_shapes1, vert_shapes2)

    def _switch_row_hulls(self, row):
        hulls = []
        switch = self.switches[row][0]
        for col in range(1, self.ncols):
            align_switch = self.switches[row][col]
            if row != 0:
                hulls.append(self._switch_hull(self.switches[row - 1][col - 1],
                                               switch, 0))
            hulls.append(self._switch_hull(switch, align_switch, 1))
            switch = align_switch
        if row != 0:
            hulls.append(self._switch_hull(self.switches[row - 1][col],
                                           switch, 0))
        return hulls

    def switch_hulls(self):
        return [sl.union()(self._switch_row_hulls(row)) for row in range(
            self.nrows)]

    def _place_cap_and_switch(self, row, col):
        xyz = [row * self.cap_dim[0],
               col * self.cap_dim[1],
               self.cap_dim[2] * col]
        cap = KeyCapShape(self.cap_dim)
        switch = SwitchSocket(self.switch_dim, hot_swap=self.hot_swap)
        switch.translate([0, 0, -self.switch_height])
        cap.rotate(self.col_degs[col], self.row_degs[row])
        switch.rotate(self.col_degs[col], self.row_degs[row])
        cap.translate(xyz)
        switch.translate(xyz)
        self.caps[row][col] = cap
        self.switches[row][col] = switch

    def _place_caps_and_switches(self):
        [[self._place_cap_and_switch(row, col) for col in range(self.ncols)] for
         row in
         range(self.nrows)]
        [self.align_row(row) for row in range(self.nrows)]

    def init_shape(self, with_caps=False):
        shapes = []
        if with_caps:
            for row in self.caps:
                for cap in row:
                    shapes.append(cap.shape())
        for row in self.switches:
            for switch in row:
                shapes.append(switch.shape())
                shapes.append(switch.vertice_shapes())

        for row in self.skirts:
            for skirt in row:
                if skirt:
                    shapes.append(skirt.shape())
                    shapes.append(skirt.vertice_shapes())

        if self.skirt_hulls:
            shapes += self.skirt_hulls
        return sl.union()(shapes, self.switch_hulls())

    def init_vertices(self):
        verts = [self.switches[0][col].vertice(2) for col in range(self.ncols)]
        verts += [self.switches[row][-1].vertice(0) for row in
                  range(
                      self.nrows)]
        verts += [self.switches[-1][col].vertice(1) for col in
                  range(self.ncols-1, -1, -1)]
        verts += [self.switches[row][0].vertice(3) for row in
                  range(self.nrows-1, -1, -1)]
        return verts

    def init_vertice_shape(self):
        return self.switches[0][0].init_vertice_shape()

    @staticmethod
    def hotswap_socket():
        plate_file = path.join("..", "src", r"hot_swap_plate.stl")
        return sl.import_(plate_file)

    def add_skirt(self, row, col, axis, deg):
        switch = self.switches[row][col]
        skirt = SocketSkirt([10, 17, 2], self.switch_dim[0], axis, deg)
        skirt.actions += switch.actions
        return skirt

    def add_skirts(self, skirts):
        self.skirts = skirts
        deg = 50
        for axis in range(len(skirts)):
            for idx in range(len(skirts[axis])):
                if skirts[axis][idx]:
                    if axis == 0:
                        skirt = self.add_skirt(self.nrows - 1, idx, axis, deg)
                    elif axis == 1:
                        skirt = self.add_skirt(idx,self.ncols-1,axis,deg)
                    elif axis == 2:
                        skirt = self.add_skirt(0, idx, axis, deg)
                    elif axis == 3:
                        skirt = self.add_skirt(idx,0,axis,deg)
                    else:
                        raise ValueError(f"invalid axis {axis}")
                    self.skirts[axis][idx] = skirt
                if idx != 0 and self.skirts[axis][idx-1]:
                    if axis in [0,3]:
                        self.hull_skirts(self.skirts[axis][idx-1], skirt)
                    else:
                        self.hull_skirts(skirt, self.skirts[axis][idx - 1])
        skirt1, skirt2 = self.skirts[2][0], self.skirts[3][0]
        if skirt1 and skirt2:
            self.hull_skirts(skirt1, skirt2)
        skirt1, skirt2 = self.skirts[3][-1], self.skirts[0][0]
        if skirt1 and skirt2:
            self.hull_skirts(skirt1, skirt2)
        skirt1, skirt2 = self.skirts[0][-1], self.skirts[1][-1]
        if skirt1 and skirt2:
            self.hull_skirts(skirt1, skirt2)
        skirt1, skirt2 = self.skirts[1][0], self.skirts[2][-1]
        if skirt1 and skirt2:
            self.hull_skirts(skirt1, skirt2)



    def hull_skirts(self, skirt1, skirt2):
        vert1 = skirt1.vertice_shapes(idxs=(0,3))
        vert2 = skirt2.vertice_shapes(idxs=(1,2))
        if self.skirt_hulls is None:
            self.skirt_hulls = sl.hull()(vert1, vert2)
        else:
            self.skirt_hulls = sl.union()(self.skirt_hulls,
                                          sl.hull()(vert1, vert2))

    def add_action(self, t: str, args):
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.switches[row][col].add_action(t, args)
                self.caps[row][col].add_action(t, args)

    def base(self, thickness):
        verts = self.vertices()
        for vert in verts:
            vert[2] = 0
        return sl.linear_extrude(thickness)(sl.polygon(verts))


class Base(Shape):
    def __init__(self, init_verts):
        super().__init__()
        for idx in range(len(init_verts)):
            init_verts[idx][2] = 0
        self.init_verts = init_verts

    def init_shape(self):
        return sl.linear_extrude(1.5)(sl.polygon(self.init_verts))

    def init_vertices(self):
        return self.init_verts

    def init_vertice_shape(self):
        return sl.translate([-3,-3,0])(sl.cube([3,3,1.5]))


class Keyboard(Shape):
    def __init__(self, grid, thumbs):
        super().__init__()
        self.grid = grid
        self.thumbs = thumbs
        self.grid_joints = []
        self.thumb_joints = []
        self.base_joints = []
        self.con_joint = None
        self.align_thumbs()
        self.zero_z()
        self.add_base_joints()
        self.add_joints()


    def _hull_1(self):
        x, y, z = self.grid.vertice(self.grid.ncols + 1)

        hpost = sl.cube([1, 2, 1])
        vpost = sl.cube([1, 1, z + 1])
        return sl.union()(sl.translate([x, y + 1, z])(hpost),
                          sl.translate([x, y + 3, 0])(vpost))

    def align_thumbs(self):
        gpoint = self.grid.ncols
        tmid = 2 * self.thumbs.ncols + self.thumbs.nrows + int(
            self.thumbs.nrows / 2) - 1
        gcoord = self.grid.vertice(gpoint)
        tcoord = self.thumbs.vertice(tmid)
        x, y, z = np.subtract(gcoord, tcoord)
        self.thumbs.translate([x-5, y, z])

    def zero_z(self):
        verts = self.vertices()
        verts = [vert[2] for vert in verts]
        min_vert = np.min(verts)
        self.translate([0,0,-min_vert+5])

    def add_joints(self):
        joint1 = Joint([14.4, 14.4, 8])
        joint1.translate([0,0,-2])
        joint2 = Joint([14.4, 8, 4], deg=90)
        grid_sock = deepcopy(self.grid.switches[0][-1])
        thumb_sock = deepcopy(self.thumbs.switches[-1][0])

        # drop this socket down so the keycap doesnt hit it
        joint1, grid_hull1 = SocketJoint(joint1, grid_sock, 1, trans=[0,0,-1.5],
                                        hull="top")
        joint2, grid_hull2 = SocketJoint(joint2, grid_sock, 2,
                                         hull="top")

        grid_sock_shape = self.grid.switches[0][-1].shape()
        grid_sock_area = self.grid.switches[0][-1].area()
        thumb_sock_area = thumb_sock.area()

        bot1 = joint1.bottom()
        bot2 = joint2.bottom()

        thumb_hull1 = sl.hull()(thumb_sock.vertice_shapes(idxs=(1,3)),
                               bot1.vertice_shape(0))
        thumb_hull1 = sl.difference()(thumb_hull1, grid_sock_shape,
                                      joint2.shape(), grid_sock_area,
                                      thumb_sock_area)

        thumb_hull2 = sl.hull()(thumb_sock.vertice_shapes(idxs=(2,3)),
                                bot2.vertice_shape(2))
        thumb_hull2 = sl.difference()(thumb_hull2, joint2.shape(),
            thumb_hull1, thumb_sock_area, grid_sock_area, grid_sock_shape)

        joint2_bot_shape = sl.difference()(bot2.shape(),
                                           thumb_sock_area, grid_sock_area,
                                           grid_sock_shape)

        self.grid_joints += [joint1.top().shape(), joint2.top().shape(),
                             grid_hull1, grid_hull2]
        self.thumb_joints += [bot1.shape(), joint2_bot_shape,
                              thumb_hull1, thumb_hull2]

        # the socket gets messed up if we don't do this. Don't know why
        self.grid.switches[0][-1] = grid_sock


    def add_grid_skirts(self):
        self.grid.add_skirts([
            [True for _ in range(self.grid.ncols)],
            [True if row > 0 else False for row in range(self.grid.nrows)],
            [True if col < (self.grid.ncols-1) else False  for col in range(
                self.grid.ncols)],
            [True for _ in range(self.grid.nrows)]
        ])

    def add_thumb_skirts(self):
        self.thumbs.add_skirts([
            [True if col > 0 else False for col in range(self.thumbs.ncols)],
            [True for _ in range(self.thumbs.nrows)],
            [True for _ in range(self.thumbs.ncols)],
            [True if row < (self.thumbs.nrows-1) else False for row in range(
                self.thumbs.nrows)],
        ])

    def add_action(self, t: str, args):
        self.grid.add_action(t, args)
        self.thumbs.add_action(t, args)

    def init_shape(self):
        return sl.union()(self.grid.shape(), self.thumbs.shape(),
                          self.grid_joints, self.thumb_joints,
                          self.base_joints,
                          self.base().shape())

    def init_vertices(self):
        return self.grid.init_vertices() + self.thumbs.init_vertices()

    def init_vertice_shape(self):
        return self.grid.init_vertice_shape()

    def base(self):
        verts = self.grid.vertices(idxs=range(self.grid.nrows-1)) + \
            self.thumbs.vertices(idxs=range(
                self.thumbs.nrows+2*self.thumbs.ncols+1)) + \
            self.grid.vertices(idxs=range(self.grid.nrows+3,
                                          len(self.grid.init_vertices()) ))
        return Base(verts)

    def add_base_joints(self):
        self.add_base_joint(self.grid, 0, 0, 2)
        self.add_base_joint(self.grid, -1, 0, 2)
        self.add_base_joint(self.grid, -1, -1, 0)
        self.add_base_joint(self.thumbs, 0, -1, 0)
        self.add_base_joint(self.thumbs, -1, -1, 0)
        self.add_base_joint(self.thumbs, 0, 0, 2)


    def base_joint_deg_movement(self, grid, col, axis):
        col_deg = grid.col_degs[col]
        if axis == 0 and col_deg >= -20:
            return [0, 8, 0]
        if axis == 2 and col_deg <=20:
            return [0, -5, 0]
        return [0,0,0]


    def add_leg(self, grid, row, col, axis):
        key = deepcopy(grid.switches[row][col])
        face_verts = key.face_vertices(axis)
        key_joint = Joint([14.4, 14.4, 5])
        key_joint = SocketJoint(key_joint, )

        base_joint = Joint([14.4, 14.4, 5]).rotate_x(90).rotate_z(90)



    def add_base_joint(self, grid, row, col, axis):
        base = self.base()
        key = deepcopy(grid.switches[row][col])
        face_verts = key.face_vertices(axis)

        #TODO: add legs if joint is to tall
        h = min(face_verts[0][2], face_verts[1][2])/2
        print("min height", h)
        #TODO: move away from keygrid if rotation angle is too steep
        trans = np.add(face_verts[0], face_verts[1]) / 2
        joint = Joint([14.4,10,h])
        trans[2] = joint.height()/2
        trans = np.asarray(np.add(trans,
                self.base_joint_deg_movement(grid, col, axis)))
        joint.translate(trans)

        top = joint.top()
        bot = joint.bottom()

        key_area = key.area()
        # key_shape = key.shape()

        base_verts = np.asarray(base.vertices())
        base_dist = np.sum((base_verts - trans)**2, axis=1)
        base_vert_idxs = np.argsort(base_dist)[:4]


        base_hull = sl.hull()(base.vertice_shapes(idxs=base_vert_idxs),
                              bot.vertice_shape(4))
        switch_hull = sl.hull()(key.face_vertice_shapes(axis),
                                top.vertice_shape(-1))
        switch_hull = sl.difference()(switch_hull, key_area,
                                      bot.shape())

        top_shape = sl.difference()(top.shape(), key_area)

        self.base_joints += [bot.shape(), base_hull]
        self.grid_joints += [switch_hull, top_shape]

        sl.scad_render_to_file(sl.union()(joint.shape(), self.base().shape(),
                                          key.shape(), key.area(),
                                          top.vertice_shapes(),
                                          base_hull,
                                          switch_hull, key.vertice_shapes()),
                               "base_joints.scad")


if __name__ == '__main__':
    keygrid = KeysGrid(4, 6, 20, 50, -30, -6,
                       [17, 17, 1], [14.4, 14.4, 4], 10,
                       row_degs=[20, -10, -40, -80])

    thumb_cluster = KeysGrid(2, 3, -20, -40, -30, -20,
                             [17, 19.5, 1], [14.4, 14.4, 4], 10)
    thumb_cluster.rotate_z(10)

    keyboard = Keyboard(keygrid, thumb_cluster)

    # thumb_cluster.rotate_z(25)
    # thumb_cluster.rotate_y(30)

    # skirt = SocketSkirt([10, 17, 2], 14.4, 0, 30)
    # socket = SwitchSocket([14.4, 14.4, 4])
    sl.scad_render_to_file(sl.union()(keyboard.shape()),
                           r"keycap.scad")
    #
    joint = Joint([8,14.4,4])
    joint.translate([0,0,(4 + 4/3 + .01)/2])

    sl.scad_render_to_file(joint.shape(), r"joint.scad")