import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from kneed import KneeLocator
import re
from utils import plot_distributions, find_acceptable_price

# Read the data
calendar = pd.read_csv('data/calendar.csv')
listing = pd.read_csv('data/listings.csv')

##############################################
### 1: Get basic understanding of the data ###
##############################################
calendar['date'].max(), calendar['date'].min() # Exactly one year, starting from Dec-22, ending by Dec-23
calendar['listing_id'].value_counts() # ~16K listings, well populated, no imbalance
listing['id'].value_counts() # One record per ID, Amount of unique ID's is almost the same as in calendar. Very good.

###################################################################
### 2: Compare price distribution btw available & not available ###
###################################################################
# Prepare adjusted price
calendar['adjusted_price'] = calendar['adjusted_price'].fillna("0")
calendar['adjusted_price'] = calendar['adjusted_price'].apply(lambda x: "".join(c for c in x if (c.isdecimal() or c=='.')))
calendar['adjusted_price'] = pd.to_numeric(calendar['adjusted_price'])

# Treat outliers & logarithmic scaling
q_25, q_75 = calendar['adjusted_price'].quantile(0.25), calendar['adjusted_price'].quantile(0.75)
IQR = q_75 - q_25
lower_limit, upper_limit = q_25 - (IQR * 1.5), q_75 + (IQR * 1.5)
calendar['adjusted_price_capped'] = calendar['adjusted_price'].apply(lambda x: upper_limit if x>upper_limit else x)
calendar['adjusted_price_capped'] = calendar['adjusted_price_capped'].apply(lambda x: lower_limit if x<lower_limit else x)
calendar['adjusted_price_capped_log'] = np.log(calendar['adjusted_price_capped']+1)

# Draw
x = calendar['adjusted_price_capped_log'][calendar['available']=="t"]
y = calendar['adjusted_price_capped_log'][calendar['available']=="f"]
plot_distributions(x,y)

#######################################
### 3.0: Map between hosts & places ###
#######################################
map_hosts_places = listing[['id', 'host_id']].copy()

#######################################
##### 3.1: Clustering: hosts data #####
#######################################
### 3.1.1: Data preparation
# Host exp in days
listing['last_scraped'] = pd.to_datetime(listing['last_scraped'])
listing['host_since'] = pd.to_datetime(listing['host_since'])
listing['host_exp_in_days'] = (listing['last_scraped'] - listing['host_since']) / np.timedelta64(1, 'D')
listing['host_exp_in_days'] = listing['host_exp_in_days'].fillna(0)

# One-hot encoding for chosen categorical features
listing['host_response_time'] = listing['host_response_time'].fillna('unknown_host_resp_time')
host_response_time = pd.get_dummies(listing['host_response_time'].str.lower())

# Adjust acceptance and response rate
listing['host_response_rate'] = listing['host_response_rate'].str.replace(r'%', r'.0').astype('float')
listing['host_response_rate'] = listing['host_response_rate'].fillna(listing['host_response_rate'].mean())
listing['host_acceptance_rate'] = listing['host_acceptance_rate'].str.replace(r'%', r'.0').astype('float')
listing['host_acceptance_rate'] = listing['host_acceptance_rate'].fillna(listing['host_acceptance_rate'].mean())

# Correct true/false columns
listing['host_is_superhost'] = listing['host_is_superhost'].fillna('f').replace({'t': 1, 'f': 0})
listing['host_has_profile_pic'] = listing['host_is_superhost'].fillna('f').replace({'t': 1, 'f': 0})
listing['host_identity_verified'] = listing['host_is_superhost'].fillna('f').replace({'t': 1, 'f': 0})

# Final host table
host_data = listing[['host_id', 'host_exp_in_days', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                     'host_listings_count', 'host_has_profile_pic', 'host_identity_verified']]
host_data = pd.concat([host_data, host_response_time], axis=1)
host_data = host_data.groupby('host_id', as_index=False).mean()
host_data = host_data.fillna(0)

### 3.1.2: Scaling, clustering, storing the results
# Scale the data
min_max_scaler_host = preprocessing.MinMaxScaler()
X = min_max_scaler_host.fit_transform(host_data.iloc[:,1:])

sse = []
range_to_discover = range(2, 10)
for k in range_to_discover:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Find best value
kl = KneeLocator(range_to_discover, sse, curve="convex", direction="decreasing")
print(f'Best number of clusters per hosts is {kl.elbow}')

# Run clustering with best cluster to fit and remember the results
kmeans = KMeans(n_clusters=kl.elbow)
kmeans.fit(X)
host_data['label_host'] = kmeans.labels_
host_cluster_description = host_data.iloc[:,1:].groupby(by='label_host').mean()
host_cluster_description = host_cluster_description.join(host_data['label_host'].value_counts())
host_cluster_description.to_csv('outputs/host_clusters.csv')

# Join host clusters to host/place mapping
map_hosts_places = map_hosts_places.merge(host_data[['host_id', 'label_host']], on='host_id')

###############################
### 3.2: Clustering: places ###
###############################
# One-hot encoding for categorical features
listing['room_type'] = listing['room_type'].fillna('unknown_room_type')
listing['room_type'] = listing['room_type'].str.lower() + '_rt'
room_type = pd.get_dummies(listing['room_type'].str.lower())

# Amenities (extracting few features that seems important for me)
listing['is_wifi'] = listing['amenities'].apply(lambda x: 1 if 'wifi' in re.sub('[^a-zA-Z]+', '', x.lower()) else 0)
listing['is_air_cond'] = listing['amenities'].apply(lambda x: 1 if 'condition' in re.sub('[^a-zA-Z]+', '', x.lower()) else 0)
listing['amenities_count'] = listing['amenities'].apply(lambda x: len(eval(x)))

# Fill na with 0's
listing['bedrooms'].fillna(0, inplace=True)
listing['beds'].fillna(0, inplace=True)
listing['reviews_per_month'].fillna(0, inplace=True)
listing['review_scores_rating'] = listing['review_scores_rating'].fillna(0)
listing['review_scores_accuracy'] = listing['review_scores_accuracy'].fillna(0)
listing['review_scores_cleanliness'] = listing['review_scores_cleanliness'].fillna(0)
listing['review_scores_checkin'] = listing['review_scores_checkin'].fillna(0)
listing['review_scores_communication'] = listing['review_scores_communication'].fillna(0)
listing['review_scores_location'] = listing['review_scores_location'].fillna(0)
listing['review_scores_value'] = listing['review_scores_value'].fillna(0)

# When was first/last review
listing['first_review'] = listing['first_review'].fillna(listing['last_scraped'][listing['first_review'].isna()])
listing['first_review'] = pd.to_datetime(listing['first_review'])
listing['last_review'] = listing['last_review'].fillna(listing['last_scraped'][listing['last_review'].isna()])
listing['last_review'] = pd.to_datetime(listing['last_review'])
listing['first_review_in_days'] = (listing['last_scraped'] - listing['first_review']) / np.timedelta64(1, 'D')
listing['last_review_in_days'] = (listing['last_scraped'] - listing['last_review']) / np.timedelta64(1, 'D')

# instant_bookable
listing['instant_bookable'] = listing['instant_bookable'].replace({'t': 1, 'f': 0})

# Final place table
place_table = listing[['accommodates', 'bedrooms', 'beds', 'is_wifi', 'is_air_cond', 'amenities_count',
                       'number_of_reviews', 'review_scores_rating', 'calculated_host_listings_count',
                       'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review_in_days', 'last_review_in_days',
                       'calculated_host_listings_count_private_rooms', 'reviews_per_month', 'id']]

place_table = pd.concat([room_type, place_table], axis=1)

# Clustering & scaling
min_max_scaler_place = preprocessing.MinMaxScaler()
X = min_max_scaler_place.fit_transform(place_table.iloc[:,:-1])

sse = []
range_to_discover = range(2, 10)
for k in range_to_discover:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Find best value
kl = KneeLocator(range_to_discover, sse, curve="convex", direction="decreasing")
print(f'Best number of clusters per places is {kl.elbow}')

# Run clustering with best cluster to fit and remember the results
kmeans = KMeans(n_clusters=kl.elbow)
kmeans.fit(X)
place_table['label_place'] = kmeans.labels_
place_table = place_table.drop(['id'], axis=1)
place_cluster_description = place_table.groupby(by='label_place').mean()
place_cluster_description = place_cluster_description.join(place_table['label_place'].value_counts())
place_cluster_description.to_csv('outputs/place_clusters.csv')

# Join host clusters to host/place mapping
map_hosts_places = map_hosts_places.merge(place_table[['id', 'label_place']], on='id')
map_hosts_places['label_final'] = map_hosts_places['label_host'].astype(str) + '_' + map_hosts_places['label_place'].astype(str)
map_hosts_places.rename({'id':'listing_id'}, axis=1, inplace=True)

###############################################
#### 6: Find acceptable prices per cluster ####
###############################################
calendar = calendar.merge(map_hosts_places, on='listing_id')

acceptable_price_per_cluster = {}
for cluster in calendar['label_final'].unique():
    avail = calendar['adjusted_price_capped_log'][(calendar['available'] == "t") & (calendar['label_final'] == cluster)]
    occup = calendar['adjusted_price_capped_log'][(calendar['available'] == "f") & (calendar['label_final'] == cluster)]
    acceptable_price_euro, _ = find_acceptable_price(avail, occup)
    avail = calendar['adjusted_price_capped'][(calendar['available'] == "t") & (calendar['label_final'] == cluster)]
    occup = calendar['adjusted_price_capped'][(calendar['available'] == "f") & (calendar['label_final'] == cluster)]
    plot_distributions(avail, occup, acceptable_price_euro, f'Cluster {cluster}')
    acceptable_price_per_cluster[cluster] = acceptable_price_euro

result = pd.DataFrame.from_dict(acceptable_price_per_cluster, orient='index', columns=['acceptable_price'])
result.index.name = 'cluster_host_place'
result.to_csv('outputs/acceptable_price_recommendations.csv')