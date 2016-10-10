#-*- coding:utf-8 -*-
import urllib2,urllib,httplib
import json
 
 
class xBaiduMap:
    """analytical geography address or location into a tuple of(latitude, longitude)"""
    def __init__(self, key='iLardjqG75kxoCiKlMD8DHjebCNF4Fpg'):
        self.host = 'http://api.map.baidu.com'
        self.path = '/geocoder?'
        self.param = {'address': None, 'output': 'json ', 'key': key, 'location': None, 'city': None}
       
    def getLocationByAddress(self, address, city=None):
        rlt = self.geocoding('address', address, city)
        if rlt != None:
            l = rlt['result']
            if isinstance(l, list):
                return None
            return l['location']['lat'], l['location']['lng']

    def getLocation(self, address):
        rlt = self.geocoding('location', address)
        if rlt != None:
            l = rlt['result']
            if isinstance(l, list):
                return None
            return l['location']['lat'], l['location']['lng']

    def geocoding(self, key, value, city=None):
        """encode request params and get result from BaiduMap"""
        if key == 'location':
            if 'city' in self.param:
                del self.param['city']
            if 'address' in self.param:
                del self.param['address']
             
        elif key == 'address':
            if 'location' in self.param:
                del self.param['location']
            if city == None and 'city' in self.param:
                del self.param['city']
            else:
                self.param['city'] = city
        self.param[key] = value
        request_url = self.host + self.path + urllib.urlencode(self.param)
        r = urllib.urlopen(request_url)
        rlt = json.loads(r.read())
        if rlt['status'] == 'OK':
            return rlt
        else:
            print "Decoding Failed"
            return None