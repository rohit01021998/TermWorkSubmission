import random

def add_traffic(file_path, num_new_cars,new_file_path):
    # Read the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the last traffic entry
    last_traffic_index = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Traffic."):
            last_traffic_index = i
            break

    if last_traffic_index == -1:
        raise ValueError("No traffic information found in the file.")

    last_traffic_num = int(lines[last_traffic_index].split('.')[1])

    # Extract the last road location
    last_road_location = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(f"Traffic.{last_traffic_num}.Init.Road"):
            last_road_location = float(lines[i].split('=')[1].strip().split()[0])
            break

    if last_road_location is None:
        raise ValueError("No Init.Road information found for the last traffic object.")

    # Prepare new traffic entries
    new_entries = []
    for i in range(1, num_new_cars + 1):
        new_traffic_num = last_traffic_num + i
        new_velocity = random.uniform(60, 80)
        random_increment = random.randint(10, 20)
        new_road_location = last_road_location + (i * random_increment)  # Spawning ahead of the last car

        new_entry = [
            f"Traffic.{new_traffic_num}.ObjectKind = Movable\n",
            f"Traffic.{new_traffic_num}.ObjectClass = Car\n",
            f"Traffic.{new_traffic_num}.Name = T{new_traffic_num:02}\n",
            f"Traffic.{new_traffic_num}.Info = Sports Car\n",
            f"Traffic.{new_traffic_num}.Movie.Geometry = 3D/Vehicles/BMW_5_2017_Blue.mobj\n",
            f"Traffic.{new_traffic_num}.Color = 0.7 0.0 0.0\n",
            f"Traffic.{new_traffic_num}.Basics.Dimension = 4.93 1.85 1.27\n",
            f"Traffic.{new_traffic_num}.Basics.Offset = 0.2 0.0\n",
            f"Traffic.{new_traffic_num}.Basics.Fr12CoM = 2.6\n",
            f"Traffic.{new_traffic_num}.Init.Orientation = 0.0 0.0 0.0\n",
            f"Traffic.{new_traffic_num}.RCSClass = RCS_Car\n",
            f"Traffic.{new_traffic_num}.DetectMask = 1 1\n",
            f"Traffic.{new_traffic_num}.Route = 0 1\n",
            f"Traffic.{new_traffic_num}.Init.Road = {new_road_location:.2f}   3.75\n",
            f"Traffic.{new_traffic_num}.Init.v = {new_velocity:.2f}\n",
            f"Traffic.{new_traffic_num}.FreeMotion = 0\n",
            f"Traffic.{new_traffic_num}.UpdRate = 200\n",
            f"Traffic.{new_traffic_num}.Motion.Kind = 4Wheel\n",
            f"Traffic.{new_traffic_num}.Motion.mass = 1600\n",
            f"Traffic.{new_traffic_num}.Motion.I = 700 2700 3000\n",
            f"Traffic.{new_traffic_num}.Motion.Overhang = 0.83  1.1\n",
            f"Traffic.{new_traffic_num}.Motion.Cf = 1.6e5\n",
            f"Traffic.{new_traffic_num}.Motion.Cr = 1.7e5\n",
            f"Traffic.{new_traffic_num}.Motion.C_roll = 2.0e5\n",
            f"Traffic.{new_traffic_num}.Motion.D_roll = 2.0e4\n",
            f"Traffic.{new_traffic_num}.Motion.C_pitch = 4.6e5\n",
            f"Traffic.{new_traffic_num}.Motion.D_pitch = 4.6e4\n",
            f"Traffic.{new_traffic_num}.Motion.SteerCtrl.Ang_max = 40.0\n",
            '''f"Traffic.{new_traffic_num}.Man.TreatAtEnd = FreezePos\n",
            f"Traffic.{new_traffic_num}.Man.N = 3\n",
            f"Traffic.{new_traffic_num}.Man.0.Limit = t_abs 74.0\n",
            f"Traffic.{new_traffic_num}.Man.0.LongDyn = auto 130\n",
            f"Traffic.{new_traffic_num}.Man.1.Limit = t_abs 82.0\n",
            f"Traffic.{new_traffic_num}.Man.1.LongDyn = auto 130\n",
            f"Traffic.{new_traffic_num}.Man.1.LatDyn = y -3.75\n",
            f"Traffic.{new_traffic_num}.Man.2.Limit = s_abs 5000.00\n",
            f"Traffic.{new_traffic_num}.Man.2.LongDyn = auto 130\n",
            f"Traffic.{new_traffic_num}.AutoDrv.AxMin_CC = -2.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.AxMin_ACC = -10.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.AxMax = 2.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.AyMax = 4.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.p = 0.41\n",
            f"Traffic.{new_traffic_num}.AutoDrv.SafeStopDist = 3.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.PreviewDist = 200.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.DesrTGap = 1.5\n",
            f"Traffic.{new_traffic_num}.AutoDrv.TauD = 4.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.TauV1 = 2.6\n",
            f"Traffic.{new_traffic_num}.AutoDrv.TauV2 = 1.3\n",
            f"Traffic.{new_traffic_num}.AutoDrv.vRef12 = 1.0\n",
            f"Traffic.{new_traffic_num}.AutoDrv.Cautious = 0.5\n",'''
            f"Traffic.{new_traffic_num}.LVD.AxMax = 6.5\n",
            f"Traffic.{new_traffic_num}.LVD.vMax = 250\n",
            f"Traffic.{new_traffic_num}.LVD.Pmax = 250\n",
            f"Traffic.{new_traffic_num}.LVD.tBuildUp = 0.2\n"
        ]

        new_entries.extend(new_entry)


    # Update the total number of vehicles
    total_vehicles = last_traffic_num + num_new_cars + 1
    for i in range(len(lines)):

        if lines[i].startswith("Traffic.N"):
            lines[i] = f"Traffic.N = {total_vehicles}\n"
            break

    # Insert new entries into the file
    with open(new_file_path, 'w') as file:
        for i, line in enumerate(lines):
            file.write(line)
            if i == last_traffic_index + 1:  # Insert new entries immediately after the last traffic entry
                file.writelines(new_entries)

# Example usage
add_traffic(r'traffic\base_file', 20, r'traffic\output_file')
