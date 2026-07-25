import math
import sys
from pyproj import Transformer, CRS
from pathlib import Path

# Usage cmd: python3 gps_calc2.py 42.410812 -83.497912

constants_path = Path(__file__).parent.parent / "constants/bounding_boxes.txt"
output_path = "results.txt"

def get_spcs_transformer():
    to_spcs = Transformer.from_crs("EPSG:4326", "EPSG:6498", always_xy=True)
    from_spcs = Transformer.from_crs("EPSG:6498", "EPSG:4326", always_xy=True)

    #to_spcs = Transformer.from_crs("EPSG:4326", "EPSG:6418", always_xy=True) 
    #from_spcs = Transformer.from_crs("EPSG:6418", "EPSG:4326", always_xy=True)     if we were in Huntsville, AL
    return to_spcs, from_spcs

def convert_to_spcs(latlon, transformer):
    x, y = transformer.transform(latlon[1], latlon[0])
    return (x, y)

def convert_from_spcs(spcs, transformer):
    lon, lat = transformer.transform(spcs[0], spcs[1])
    return (lat, lon)

def convert_to_dec_degrees(coord1):
    coord1str = coord1
    result_coord = []
    for i in range(0, 2):
        if (coord1str[0] == " "):
            coord1str = coord1str[1:]
        mult_by_neg_one = False
        separator = "°"
        index = coord1str.find(separator)
        D = coord1str[:index]
        coord1str = coord1str.replace(D, '')
        coord1str = coord1str[1:]
        D = float(D)
        separator = "'"
        index = coord1str.find(separator)
        M = coord1str[:index]
        coord1str = coord1str.replace(M, '')
        coord1str = coord1str[1:]
        M = float(M)
        separator = '"'
        index = coord1str.find(separator)
        S = coord1str[:index]
        coord1str = coord1str.replace(S, '')
        coord1str = coord1str[1:]
        S = float(S)
        direction = coord1str[0]
        coord1str = coord1str[1:]

        if ((direction == "S") or (direction == "W")):
            mult_by_neg_one = True
        result = D + M/60 + S/3600
        if (mult_by_neg_one):
            result *= -1

        result_coord.append(result)
    return result_coord

def rotate_point(pivot, point, theta):
    dx = point[0] - pivot[0]
    dy = point[1] - pivot[1]
    x_rot = dx * math.cos(theta) - dy * math.sin(theta)
    y_rot = dx * math.sin(theta) + dy * math.cos(theta)
    return (x_rot + pivot[0], y_rot + pivot[1])

def unrotate_point(pivot, rotated_point, theta):
    return rotate_point(pivot, rotated_point, -theta)

def calc_theta(tl, tr):
    return math.atan2(tr[1] - tl[1], tr[0] - tl[0])

def rotate_rectangle(tl, tr, bl, br, theta):
    tl_rot = rotate_point(tl, tl, theta)
    tr_rot = rotate_point(tl, tr, theta)
    bl_rot = rotate_point(tl, bl, theta)
    br_rot = rotate_point(tl, br, theta)
    return tl_rot, tr_rot, bl_rot, br_rot

def calc_dimensions(tl, tr, bl, br):
    width = (math.hypot(tr[0] - tl[0], tr[1] - tl[1]) + 
             math.hypot(br[0] - bl[0], br[1] - bl[1])) / 2
    height = (math.hypot(bl[0] - tl[0], bl[1] - tl[1]) + 
              math.hypot(br[0] - tr[0], br[1] - tr[1])) / 2
    return width, height

############################################################################################
#func/main code separator
############################################################################################

# #northville high school field coordinates
# field_tl = """42°24'40.09"N 83°29'53.86"W"""
# field_tr = """42°24'40.24"N 83°29'51.74"W"""
# field_br = """42°24'36.70"N 83°29'51.28"W"""
# field_bl = """42°24'36.55"N 83°29'53.41"W"""

# tl = convert_to_dec_degrees(field_tl)
# tr = convert_to_dec_degrees(field_tr)
# br = convert_to_dec_degrees(field_br)
# bl = convert_to_dec_degrees(field_bl)

rand_lat = sys.argv[1]
rand_lon = sys.argv[2]

rand_latlon = []
rand_latlon.append(rand_lat)
rand_latlon.append(rand_lon)

tl = []
tr = []
br = []
bl = []

with open(constants_path, "r") as f:
    for i, line in enumerate(f):
        if i == 0:
            line_contents = line.strip().split()
            tl.append(float(line_contents[0]))
            tl.append(float(line_contents[1]))
        elif i == 1:
            line_contents = line.strip().split()
            tr.append(float(line_contents[0]))
            tr.append(float(line_contents[1]))
        elif i == 2:
            line_contents = line.strip().split()
            br.append(float(line_contents[0]))
            br.append(float(line_contents[1]))
        elif i == 3:
            line_contents = line.strip().split()
            bl.append(float(line_contents[0]))
            bl.append(float(line_contents[1]))
        else:
            print("Read in the first 4 elements")
            break

# print(tl, tr, br, bl)
#calc SPCS transformers
to_spcs, from_spcs = get_spcs_transformer()

#convert to SPCS coordinates
tl_spcs = convert_to_spcs(tl, to_spcs)
tr_spcs = convert_to_spcs(tr, to_spcs)
br_spcs = convert_to_spcs(br, to_spcs)
bl_spcs = convert_to_spcs(bl, to_spcs)

#calcu theta w top left and top right corners
theta = calc_theta(tl_spcs, tr_spcs)
# print("theta: ", math.degrees(theta), " degrees")

#calc dimensions
width, height = calc_dimensions(tl_spcs, tr_spcs, bl_spcs, br_spcs)
# print("field width: ", width, " meters")
# print("field height: ", height, " meters")

tl_rect, tr_rect, bl_rect, br_rect = rotate_rectangle(tl_spcs, tr_spcs, bl_spcs, br_spcs, theta)

#convert back to lat/lon
tl_final = convert_from_spcs(tl_rect, from_spcs)
tr_final = convert_from_spcs(tr_rect, from_spcs)
br_final = convert_from_spcs(br_rect, from_spcs)
bl_final = convert_from_spcs(bl_rect, from_spcs)
# print("final coordinates: ")
# print(tl_final)
# print(tr_final)
# print(br_final)
# print(bl_final)



# #test w random point
# field_rand = """42°24'38.61"N 83°29'51.52"W"""
# rand_latlon = convert_to_dec_degrees(field_rand)

rand_spcs = convert_to_spcs(rand_latlon, to_spcs)

# print("orig point: ", rand_latlon)

#rotate by theta
rand_rotated = rotate_point(tl_spcs, rand_spcs, theta)
rand_rotated_latlon = convert_from_spcs(rand_rotated, from_spcs)
with open(output_path, "a") as f:
    f.write(f"{rand_rotated_latlon[0]} {rand_rotated_latlon[1]}\n")
# print("rotated by theta: ", rand_rotated_latlon)

#unrotate back
rand_back = unrotate_point(tl_spcs, rand_rotated, theta)
rand_back_latlon = convert_from_spcs(rand_back, from_spcs)
# print("unrotated back: ", rand_back_latlon)