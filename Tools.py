from pyproj import Proj, transform, Transformer


def coordinate_change(x, y, before_type, after_type):

    before_type = int(before_type.split(":")[1])
    after_type = int(after_type.split(":")[1])

    transformer = Transformer.from_crs(before_type,after_type,always_xy=True)

    x_, y_ = transformer.transform(x,y)
    # proj_1 = Proj(init=before_type)
    # proj_2 = Proj(init=after_type)
    #
    # x_, y_ = transform(proj_1, proj_2, x, y)

    return x_, y_

