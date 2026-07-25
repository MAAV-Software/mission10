import subprocess

locations_path = "sample_locations.txt"
output_path = "results.txt"

with open(output_path, "w") as f:
    pass

mine_bounds = []
with open(locations_path, "r") as f:
    for i, line in enumerate(f):
        line_contents = line.strip().split()
        args = []
        args.append("python3")
        args.append("gps_calc2.py")
        args.append(line_contents[0])
        args.append(line_contents[1])
        mine_bounds.append((line_contents[2], line_contents[3], line_contents[4], line_contents[5], line_contents[6]))
        subprocess.run(args)

converted_coords = []       
with open(output_path, "r") as f:
    for i, line in enumerate(f):
        line_contents = line.strip().split()
        converted_lat = line_contents[0]
        converted_lon = line_contents[1]
        converted_coords.append((converted_lat, converted_lon))

with open(output_path, "w") as f:
    for i, bound in enumerate(mine_bounds):
        print(bound)
        f.write(f"{converted_coords[i][0]} {converted_coords[i][1]} {mine_bounds[i][0]} {mine_bounds[i][1]} {mine_bounds[i][2]} {mine_bounds[i][3]} {mine_bounds[i][4]}\n")


    