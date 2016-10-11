try:
  import httplib
  import requests
except:
        import http.client as httplib
try:
        from urllib import parse
except:
        import urllib
try:
	import simplejson as json
except:
	import json


#create and return instance that connect to facebook graph
def create():
    con = httplib.HTTPSConnection('graph.facebook.com')
    con.set_tunnel('127.0.0.1', '1080')
    return con


#Sends request to facebook graph
#Returns the facebook-json response converted to python object
def send_request(req_cat, con, req_str, kwargs):
        try:
                kwargs= parse.urlencode(kwargs)    #python3x
        except:
                kwargs= urllib.urlencode(kwargs)   #python2x
        proxies ={'http':'http://127.0.0.1:1080','https':'http://127.0.0.1:1080'}
        host = 'https://graph.facebook.com'
        req_str = host +req_str
        # con.request(req_cat, req_str, kwargs)      #send request to facebook graph
        html = requests.get(req_str, proxies=proxies)
        res = html.content	   #read response
        t=type(res)
        if type(res) == t:
                res=bytes.decode(res)
        return json.loads(res)                     #convert the response to python object
