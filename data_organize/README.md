# Cirrus data reorganization

Using creat_dict.py to creat name_dict.json based on the order of the data from two LiDAR sensors. (This is not perfect, but name order is the only information I can use to align the data.)

Using link_data.py to move all data from batches (1 to 7) into a joint path named data_all with soft links.

Using link_point.py to move the Uniform pattern data based on the name pairs in name_dict.json.