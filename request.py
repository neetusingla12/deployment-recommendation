# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:39:54 2020

@author: Neetu
"""

import requests

url = 'http://localhost:5000/news-recom'
r = requests.post(url)

print(r.json())
