from copy import deepcopy
import solid as sl
from shape import Shape

class JointBottom(Shape):
    def __init__(self, dim, tol=0.01):
        super().__init__()
        self.dim = dim
        self.tol = tol

    def notch(self):
        w, l, h = self.dim
        tol = self.tol
        div = 3
        poly_verts = [
            [(-w / div) + tol*2, (h / div * 1.8) - tol / 2],
            [(w / div) - tol*2, (h / div * 1.8) - tol / 2],
            [(w / (div*2)) - tol, -self.tol], [(-w / (div*2)) + tol, -self.tol],
        ]
        notch = sl.polygon(poly_verts)
        notch = sl.rotate(90, [1, 0, 0])(sl.linear_extrude(l)(notch))
        return sl.translate([0, l / 2, 0])(notch)

    def bot(self):
        w,l,h = self.dim
        return sl.translate([0, 0, -(h - h / 3) / 2 - .01])(
            sl.cube([w, l, h - h / 3 - self.tol/2], center=True))

    def init_shape(self):
        return sl.union()(self.notch(), self.bot())

    def init_vertice_shapes(self):
        w,l,h = self.dim
        tol = self.tol

        h = h - h / 3

        adj = .05

        face0 = sl.translate([0, l/2-adj, -h / 2 - tol])(
            sl.cube([w, .1, h-tol], center=True))
        face1 = sl.translate([w/2-adj, 0, -h / 2 - tol])(
            sl.cube([.1, l, h-tol], center=True))
        face2 = sl.translate([0, -l/2+adj, -h / 2 - tol])(
            sl.cube([w, .1, h-tol], center=True))
        face3 = sl.translate([-w/2+adj, 0, -h / 2 - tol])(
            sl.cube([.1, l, h-tol], center=True))
        bot_face = sl.translate([-w/2,-l/2,-h])(sl.cube([w,l,1.5]))

        return [face0, face1, face2, face3, bot_face]


class JointTop(Shape):
    def __init__(self, dim, tol=.01):
        super().__init__()
        self.dim = dim
        self.tol = tol

    def notch(self):
        w, l, h = self.dim
        div = 3
        poly_verts = [
            [(-w / div) - self.tol, (h / div * 1.8) + self.tol],
            [(w / div) + self.tol, (h / div * 1.8) + self.tol],
            [(w / (div*2)) + self.tol*2, -.01], [(-w / (div*2)) - self.tol*2,
                                                -.01],
        ]
        notch = sl.polygon(poly_verts)
        notch = sl.rotate(90, [1, 0, 0])(sl.linear_extrude(l + .1)(notch))
        return sl.translate([0, (l + .1) / 2, 0])(notch)

    def top(self):
        w,l,h = self.dim
        return sl.translate([0, 0, h/2+self.tol/2])(sl.cube([w, l,
                                                             h-self.tol/2],
                                                 center=True))

    def init_shape(self):
        return sl.difference()(self.top(), self.notch())

    def init_vertice_shapes(self):
        w,l,h = self.dim
        tol = self.tol
        vh = h - (h / 3 * 1.8)
        # mh = h - (h/4) + .25
        mh = h - vh/2 + .1
        adj = .05

        face0 = sl.translate([0, l/2-adj, mh])(
            sl.cube([w, .1, vh-tol], center=True))
        face1 = sl.translate([w/2-adj, 0, mh])(
            sl.cube([.1, l, vh-tol], center=True))
        face2 = sl.translate([0, -l/2+adj, mh])(
            sl.cube([w, .1, vh-tol], center=True))
        face3 = sl.translate([-w/2+adj, 0, mh])(
            sl.cube([.1, l, vh-tol], center=True))
        top_face = sl.translate([-w/2,-l/2,self.dim[2]-.1])(sl.cube([w,l,.2]))

        return [face0, face1, face2, face3, top_face]

#TODO: make joint height dim the actual total height of the joint
class Joint(Shape):
    def __init__(self, dim, tol=.1, deg=None):
        super().__init__()
        self.dim = dim
        self.tol = tol
        self.b = JointBottom(dim, tol=tol)
        self.t = JointTop(dim, tol=tol)
        if deg:
            self.rotate_z(deg)
            self.translate([-self.dim[0]/3 + .5,0,0])

    def init_shape(self):
        return sl.union()(self.b.shape(), self.t.shape())

    def init_vertice_shapes(self):
        return self.b.init_vertice_shapes() + self.t.init_vertice_shapes()

    def bottom(self):
        bot = deepcopy(self.b)
        bot.actions = self.actions
        return bot

    def top(self):
        top = deepcopy(self.t)
        top.actions = self.actions
        return top

    def height(self):
        h = self.dim[2]
        return h + h / 3 - self.tol/2

    def init_origin(self):
        return [0,0,0]


if __name__ == '__main__':
    w, l, h = 7, 5, 3
    poly_verts = [
        [-w / 3, h / 3], [w / 3, h / 3],
        [w / 6, -.01], [-w / 6, -.01],
    ]


    top = sl.translate([0,0,h/2])(sl.cube([w,l,h], center=True))
    poly = sl.polygon(poly_verts)
    poly = sl.rotate(90, [1,0,0])(sl.linear_extrude(l+.1)(poly))
    poly = sl.translate([0, (l+.1)/2, 0])(poly)

    top = sl.difference()(top, poly)

    tol = .01
    poly_verts = [
        [(-w / 3)+tol, (h / 3) - tol/2], [(w / 3) - tol, (h / 3) - tol/2],
        [(w / 6) - tol, -.01], [(-w / 6) +tol, -.01],
    ]
    notch = sl.polygon(poly_verts)
    notch = sl.rotate(90, [1, 0, 0])(sl.linear_extrude(l)(notch))
    notch = sl.translate([0, l/2, 0])(notch)
    bot = sl.translate([0,0,-(h-h/3)/2-tol])(sl.cube([w,l,h-h/3], center=True))
    bot = sl.union()(bot, notch)

    cube = sl.translate([0, 0, -10])(sl.cube([5,5,5], center=True))

    joint = sl.union()(bot, top, cube)

    joint = Joint([14.4, 14.4, 5])
    joint.rotate_x(90).rotate_z(90)

    sl.scad_render_to_file(sl.union()(joint.bottom().shape(),
                                      joint.top().shape(),
                                      joint.vertice_shapes()),
                           "joint.scad")