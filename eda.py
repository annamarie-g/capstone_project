import modeling as md


def get_data():
    with open('d.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    #df = pd.concat([training_data[0], training_data[1]], axis=1)
    #df = df.ix[df['category_code'] == 'm4w', :]
    target = df.pop('age')
    return df, target

if __name__=='__main__':
    
