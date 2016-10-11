#-*- coding:utf-8 -*-
import fb
import json
import numpy as np
from sklearn.cluster import KMeans


class FbEvent:
    """Facebook data collecting and processing class"""
    def __init__(self, token='', user_threshold=1000, event_threshold=1000):
        self.facebook = fb.graph.api(token)
        self.user_threshold = user_threshold
        self.event_threshold = event_threshold
        self.all_users = dict()
        self.all_events = dict()
        self.event_user = dict()

    def get_event_from_query(self, keyword="USC"):
        """get event list of a keyword query"""
        events = self.facebook.get_query_object(keyword=keyword)
        if 'data' in events.keys():
            return events['data']
        return []

    def get_event_detail(self, event_id):
        """get event detail from a event, which include description, location name, location detail, attending list"""
        events = self.facebook.get_object(cat="single", id=event_id, fields=['description', 'name', 'venue', 'attending'])
        return events.get('description', None), events.get('name', None), \
               events.get('venue', None), events.get('attending', None)

    def grasp_user_event_info(self, query_list):
        """get facebook events and users from a keyword query list till the scale of the data is large enough ,
         and skip those events,
         whose description, location info is missing """
        for query in query_list:
                events = self.get_event_from_query(query)
                for event in events:
                    if len(self.all_users) > self.user_threshold and len(self.all_events) > self.event_threshold:
                        return
                    desc, loc_name, location, users = self.get_event_detail(event['id'])
                    if event['id'] not in self.all_events.keys() and desc is not None and loc_name is not None\
                            and location is not None and 'longitude' in location.keys() and users is not None:
                        self.all_events[event['id']] = {'description': desc, 'loc_name': loc_name, 'location': location}
                        self.event_user[event['id']] = []
                        for user in users['data']:
                            self.event_user[event['id']].append(user['id'])
                            if user['id'] not in self.all_users.keys():
                                self.all_users[user['id']] = user

    def write_into_file(self, event_file='event.json', user_file='user.json', event_user_file='event_user.json'):
        """write all data into files"""
        with open(event_file, 'w') as f:
            f.write(json.dumps(self.all_events))
        with open(user_file, 'w') as f:
            f.write(json.dumps(self.all_users))
        with open(event_user_file, 'w') as f:
            f.write(json.dumps(self.event_user))

        return

    def file_transform(self, event_file='event.json', user_file='user.json', event_user_file='event_user.json'):
        """read files and transform data into rating matrix and clustering location with kmeans method"""
        with open(event_file, 'r') as f:
            self.all_events = json.load(f)
        with open(user_file, 'r') as f:
            self.all_users = json.load(f)
        with open(event_user_file, 'r') as f:
            self.event_user = json.load(f)

        #construct user_id map and event_map from files
        user_map = dict();user_id_map = dict()
        event_map = dict();event_id_map = dict()
        all_event_id = [a[0] for a in sorted(self.all_events.items(), key=lambda c:c[0])]
        for i in range(len(all_event_id)):
            event_map[all_event_id[i]] = i
            event_id_map[i] = all_event_id[i]
        all_user_id = [a[0] for a in sorted(self.all_users.items(), key=lambda c:c[0])]
        for i in range(len(all_user_id)):
            user_map[all_user_id[i]] = i
            user_id_map[i] = all_user_id[i]

        #construct rating matrix with event_user data
        R = np.mat(np.zeros((len(self.all_events), len(self.all_users))))
        for event_id, user_list in self.event_user.items():
            for user_id in user_list:
                R[event_map[event_id], user_map[user_id]] = 1

        #clustering events with location and return cluster label
        all_event_location = [[a[1]['location']['longitude'], a[1]['location']['latitude']]
                              for a in sorted(self.all_events.items(), key=lambda c:c[0])]
        feature = np.mat(all_event_location)
        clf = KMeans(n_clusters=8)
        s = clf.fit(feature)

        # self.skip_sparse_item(R, event_id_map, user_id_map, event_file, user_file, event_user_file)
        return R, clf.labels_

    def skip_sparse_item(self, R, event_id_map, user_id_map, event_file, user_file, event_user_file):
        #skip those users, who attend at most 2 events
        attend_cnt = np.sum(R, axis=0)
        all_users = dict()
        event_user = dict()
        for i in range(R.shape[0]):
            event_user[event_id_map[i]] = []
            for j in range(R.shape[1]):
                if attend_cnt[0, j] > 2 and R[i, j] == 1:
                    all_users[user_id_map[j]] = self.all_users[user_id_map[j]]
                    event_user[event_id_map[i]].append(user_id_map[j])
        self.all_users = all_users
        self.event_user = event_user
        for event_id, user_list in self.event_user.items():
            if len(user_list) == 0:
                del self.event_user[event_id]
                del self.all_events[event_id]
        self.write_into_file(event_file, user_file, event_user_file)


def test_facebook_api(token):
    facebook = fb.graph.api(token)
    object1 = facebook.get_object(cat="single", id='1702801646707359', fields=['name', 'friends', 'events'])
    print object1
    object2 = facebook.get_object(cat="multiple", ids=['me', '1622298331340599'])
    print object2
    object3 = facebook.get_object(cat="single", id='1137489669671017', fields=['place'])
    print object3
    object4 = facebook.get_query_object(keyword='USC')
    print object4


def test_fb_event(token):
    """test FbEvent class"""
    user_threshold = 1000
    event_threshold = 1000
    query_list = ['USC', 'UCLA', 'Stanford', 'MIT',
                  'Harvard', 'UCB', 'Princeton', 'Cambridge']
    event_file = 'event.json'
    user_file = 'user.json'
    event_user_file = 'event_user.json'
    fb = FbEvent(token, user_threshold, event_threshold)
    # fb.grasp_user_event_info(query_list)
    # fb.write_into_file(event_file, user_file, event_user_file)
    R, label = fb.file_transform(event_file, user_file, event_user_file)
    return R, label

if __name__ == '__main__':
    token = "EAACEdEose0cBAICWQhyVRPUF3tb9ZCMZCK72OJ1HdY0CbZAI6D" \
            "WaA4yxAZAntTVP4XZC7wnsNZC0PmDLZAEBeKwI8WTBVuvHX1vtn" \
            "ZBhpEaZCIpZBgO77Y0zpGbJEGCsK3sWkSNWSCoOOVa97C1Kj0T0ZBSRWZBybFBUxOeQ6W7GKrxtOwZDZD"
    # test_facebook_api(token)
    R, label = test_fb_event(token)
    print R
    print label
    
