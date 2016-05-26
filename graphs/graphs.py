import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
sns.set(style = 'darkgrid', context='talk')


def read_data(): 
    sns.set_context('talk', font_scale=1.5)
    df = pd.read_pickle('dataframe_for_eda.pkl')
    return df 

def age_histogram(df_age):
    age_counts = df_age.groupby('age').age.count()

    y = age_counts.values
    x = [int(age) for age in age_counts.index]

    f, ax = plt.subplots(1,1, figsize=(12,8))
    sns.barplot(x,y, palette=sns.dark_palette('#008080', reverse=True, n_colors=60), linewidth=0)
    ax.set_ylabel('Postings')
    ax.set_xlabel('')
    ax.set_title('Histogram of Postings by Age')
    x_ticks = [0]
    x_ticks.extend(range(2,95, 5))
    x_ticklabels = ['']
    x_ticklabels.extend(range(20,95,5))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    #need to fix xlabels
    sns.despine(bottom=True, right=True)
    sns.plt.xlim(-1, 90)
    for i,p  in enumerate(ax.patches):
        height = p.get_height()
        if ((i+18) % 5 == 0) and (i+18 < 70): 
            ax.text(p.get_x()-1, height + 4, i+18, fontsize=18)

    plt.show()

def num_images_by_age(df_age):  
    sns.set_context('talk', font_scale=1.5)
    df_age_images = df_age.ix[df_age['age']<66, :]
    age_means = df_age_images.groupby('age').num_images.mean()
    age_bins = [18]
    age_bins.extend(list(range(25,66,5)))
    bin_labels = ['18-24'] 
    bin_labels.extend(["{} - {}".format(i, i+4) for i in range(25,61,5)])
    df_age_images['hist_age_group'] = pd.cut(df_age_images.ix[:,'age'], age_bins, right = False, labels= bin_labels)
    age_means  = df_age_images.groupby('hist_age_group').num_images.mean()
    y = age_means.values
    x = age_means.index
    f, ax = plt.subplots(1,1, figsize=(16,8))
    ax.set_ylabel('Average Number of Images')
    sns.barplot(x,y, palette=sns.dark_palette('#008080', reverse=True, n_colors=10), linewidth=0)
    ax.set_xlabel('')
    ax.set_title('Average Number of Images by Age')
    for item in ax.get_xticklabels():
        item.set_rotation(15)
    sns.plt.ylim(0.0,0.75)
    sns.despine(bottom=True, right=True)
    plt.show()

def post_length_by_age(df_age):
    sns.set_context('talk', font_scale=1.5)
    df_age['post_length'] = df_age['total_text'].apply(len)
    df_age_length = df_age.ix[df_age['age']<66, :]
    age_means = df_age_length.groupby('age').post_length.mean()
    age_bins = [18]
    age_bins.extend(list(range(25,66,5)))
    bin_labels = ['18-24'] 
    bin_labels.extend(["{} - {}".format(i, i+4) for i in range(25,61,5)])
    df_age_length['hist_age_group'] = pd.cut(df_age_length.ix[:,'age'], age_bins, right = False, labels= bin_labels)
    age_means  = df_age_length.groupby('hist_age_group').post_length.mean()
    y = age_means.values
    x = age_means.index
    f, ax = plt.subplots(1,1, figsize=(16,8))
    ax.set_ylabel('Post Length (characters)')
    sns.barplot(x,y,palette=sns.dark_palette('#008080', reverse=True, n_colors=10), linewidth=0)
    ax.set_xlabel('')
    ax.set_title('Average Post Length by Age')
    for item in ax.get_xticklabels():
        item.set_rotation(15)
    sns.plt.ylim(0, 650)
    sns.despine(bottom=True, right=True, trim=True)
    plt.show()

def posts_by_category(df): 
    sns.set_context('talk', font_scale=1.5)
    cat_counts = df.groupby('category_code').url.count()
    x = cat_counts.index
    y = cat_counts.values
    f, ax = plt.subplots(1,1, figsize=(10,8))
    ax.set_ylabel('Postings')
    sns.barplot(x,y,palette=sns.light_palette('#008080', reverse=True, n_colors=10), linewidth=0)
    ax.set_xlabel('')
    ax.set_title('Postings by Category')
    for item in ax.get_xticklabels():
        item.set_rotation(15)
    sns.despine(bottom=True, right=True, trim=True)

    percentage = [np.round((float(y_)*100/sum(y)),2) for y_ in y] 
    for i,p  in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x(), p.get_height()+ 10, '{}%'.format(percentage[i]), fontsize=20)
    plt.show()


def post_length_by_category(df):
    sns.set_context('talk', font_scale=1.5)
    df['post_length'] = df['total_text'].apply(len)
    cat_means  = df.groupby('category_code').post_length.mean()
    y = cat_means.values
    x = cat_means.index
    f, ax = plt.subplots(1,1, figsize=(10,8))
    ax.set_ylabel('Post Length (characters)')
    sns.barplot(x,y,palette=sns.dark_palette('#008080', reverse=True, n_colors=10), linewidth=0)
    ax.set_xlabel('')
    ax.set_title('Average Post Length by Category')
    for item in ax.get_xticklabels():
        item.set_rotation(15)
    #sns.plt.ylim(350, 700)
    sns.despine(bottom=True, right=True, trim=True)
    plt.show()
    


def age_hist_by_category(df_age):
    pass


def age_dataframe(df): 
    df_age = df.ix[df['age'].notnull(), :] 
    df_age = df_age.ix[df_age['age']<91, :] 
    df_age = df_age.ix[df_age['age']>17, :] 
    return df_age


if __name__ == '__main__':
    df = read_data()
    df_age = age_dataframe(df) 
    #age_histogram(df)
