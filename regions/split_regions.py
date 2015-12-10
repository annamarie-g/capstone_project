import json 

def split_dictionary(locations, keys):
    sub_dict = {k:locations[k] for k in keys}
    return sub_dict 


    return dict_subset 

if __name__=='__main__':
    with open('../locations.json') as fp: 
       locations = json.load(fp) 

    new_england = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont']
    mid_atlantic = ['New Jersey', 'New York', 'Pennsylvania']
    east_north_central = ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin']
    west_north_central = ['Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska','North Dakota', 'South Dakota']
    south_atlantic = ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina','Virginia', 'District of Columbia', 'West Virginia']
    east_south_central = ['Alabama', 'Kentucky', 'Mississippi', 'Tennessee']
    west_south_central = ['Arkansas' , 'Louisiana', 'Oklahoma', 'Texas']
    mountain_west = ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah',  'Wyoming'] 
    pacific_west = ['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington'] 

    
    with open('new_england.json', 'wb') as fp:
        new_england_locations = split_dictionary(locations, new_england)
        json.dump(new_england_locations, fp)

    with open('mid_atlantic.json', 'wb') as fp:
        mid_atlantic_locations = split_dictionary(locations, mid_atlantic)
        json.dump(mid_atlantic_locations, fp)

    with open('east_north_central.json', 'wb') as fp:
        east_north_central_locations = split_dictionary(locations, east_north_central)
        json.dump(east_north_central_locations, fp)
        
    with open('west_north_central.json', 'wb') as fp:
        west_north_central_locations = split_dictionary(locations, west_north_central)
        json.dump(west_north_central_locations, fp)
        
    with open('south_atlantic.json', 'wb') as fp:
        south_atlantic_locations = split_dictionary(locations, south_atlantic)
        json.dump(south_atlantic_locations, fp)
        
    with open('east_south_central.json', 'wb') as fp:
        east_south_central_locations = split_dictionary(locations, east_south_central)
        json.dump(east_south_central_locations, fp)

    with open('west_south_central.json', 'wb') as fp:
        west_south_central_locations = split_dictionary(locations, west_south_central)
        json.dump(west_south_central_locations, fp)

    with open('mountain_west.json', 'wb') as fp:
        mountain_west_locations = split_dictionary(locations, mountain_west)
        json.dump(mountain_west_locations, fp)

    with open('pacific_west.json', 'wb') as fp:
        pacific_west_locations = split_dictionary(locations, pacific_west)
        json.dump(pacific_west_locations, fp)
        



