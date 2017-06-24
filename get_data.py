import json
import urllib2
import urllib
# import requests

import re
import csv

from lxml import html


class Scrapy():
    def __init__(self):
        self.urls = []
        self.domain = 'https://openi.nlm.nih.gov/'
        self.regex = re.compile(r"var oi = (.*);")
        self.final_data = {}
        self.path = 'image/'
        self.img_idx = 0
        self.img_des=[]
    # The total number of the images is 7450,
    # then we get 75 pages if every page shows 100 images
    def get_urls(self):
        for i in range(0, 75):
            url = 'https://openi.nlm.nih.gov/gridquery.php?q=lung&it=x,xg&sub=x&m=%d&n=%d' % ( 100 * i, 100 + 100 * i)
            self.urls.append(url)
    def download(self, link):
        self.img_idx += 1
        req = urllib2.Request(link)
        response = html.fromstring(urllib2.urlopen(req).read().decode('UTF-8'))
        
        
        
        div = response.xpath('//table[@class="masterresultstable"]//div[@class="meshtext-wrapper-left"]')
        # div = response.xpath('//div[@id="imageClassM"]//b[@class="fidings"]/text()')

        if div != []:
            div = div[0]
        else:
            return

        typ = div.xpath('.//strong/text()')[0]
        items = div.xpath('.//li/text()')
        img = response.xpath('//img[@id="theImage"]/@src')[0]

        self.final_data[self.img_idx] = {
            'typ': typ,
            'items': items,
            'img': self.domain + img
        }

        urllib.urlretrieve(self.domain+img, self.path+str(self.img_idx)+".png")

        with open('image/data_new.json', 'w') as f:
            json.dump(self.final_data, f)

        print(self.img_idx, self.final_data[self.img_idx])
    def get_content(self):
        #url is reached 
        for url in self.urls:
            req = urllib2.Request(url)
            response = html.fromstring(urllib2.urlopen(req).read().decode('UTF-8'))
            script = response.xpath('//script[@language="javascript"]/text()')[0]

            json_string = self.regex.findall(script)[0]
            json_data = json.loads(json_string)

            links = [self.domain + x['nodeRef'] for x in json_data]
            for link in links:
                try:
                    self.download(link)
                except:
                    print("download fail! " + link)
        #        self.img_des.append([self.img_idx,self.get_Img(self.get_Html(link))[0]])
        #self.write_Out(self.img_des)
    def get_Html(self,url):
        page = urllib.urlopen(url)
        ht = page.read()
        return ht
    def get_Img(self,ht):
        reg = r'<div id="other"><b>ABSTRACT.+?</div>'
        imgre = re.compile(reg)
        imglist = re.findall(imgre,ht)
        imglist.append("None")
        return imglist      
    def write_Out(self,descript):
        csvfile = file('image/csv_test.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(descript)
    
if __name__ == '__main__':
    s = Scrapy()
    s.get_urls()
    s.get_content()
