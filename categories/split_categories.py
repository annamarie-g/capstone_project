import json 

def split_dictionary(locations, keys):
    sub_dict = {k:locations[k] for k in keys}
    return sub_dict 


    return dict_subset 

if __name__=='__main__':
    with open('../categories.json') as fp: 
       categories = json.load(fp) 
    
    category_splits = { 'appliances' : ['app', 'ppd'], 'art_and_crafts': ['ard', 'art'], 'antiques' :  ['atd', 'atq'], 'baby_and_kids' : ['bab', 'bad'],'barter' : ['bar'] ,'bicycle_parts' : ['boa', 'bdp'],'business' : ['bfd', 'bfs'],'bicycles' : ['bid', 'bik'],'books_and_mags' : ['bkd', 'bks'] ,'boats' : ['boa', 'bod'],'boat_parts' : ['bpd', 'bpo'] ,'collectibles' : ['cbd','clt'],'clothing_and_accessories' : ['cld', 'clo'],'cars_and_trucks' : ['ctd', 'cto'],'electronics' : ['eld', 'ele'],'cds_dvds_vhs' : ['emd', 'emq'],'general' : ['fod', 'for'],'furniture' : ['fud', 'fuo'],'garage_sales' : ['gms'],'farm_and_garden' : ['grd', 'grq'],'health_and_beauty' : ['hab', 'had'],'household_items' : ['hsd', 'hsh'],'heavy_equipment' : ['hvd', 'hvo'],'jewelry' : ['jwd', 'jwl'],'materials' : ['mad', 'mat'],'motorcycles' : ['mpd', 'mpo'],'instruments' : ['msd', 'msg'],'photo_video' : ['phd', 'pho'],'auto_parts_general' : ['ptd', 'pts'],'rec_vehicles' : ['rvd', 'rvs'],'computer_parts' : ['sdp', 'sop'],'sporting_goods' : ['sgd', 'spo'],'atvs_utvs' : ['snd', 'snw'],'computers' : ['syd', 'sys'],'toys_and_games' : ['tad', 'tag'],'tickets' : ['tid', 'tix'],'tools' : ['tld', 'tls'],'video_gaming' : ['vgd', 'vgm'],'wanted' : ['wad', 'wan'] ,'wheels_and_tires' : ['wtd', 'wto'],'free_stuff' : ['zip']}

    for category in category_splits.keys():
        with open('{}.json'.format(category), 'wb') as fp:
            split  = split_dictionary(categories, category_splits[category])
            json.dump(split, fp)

        



