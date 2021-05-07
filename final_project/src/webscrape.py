#!/usr/bin/env python3
"""
########### Import libs ##########
"""
from urllib.request import urlopen
from urllib.request import urlretrieve
import re
#import urllib2
from bs4  import BeautifulSoup 
import os


#main function
def main():
    
    soup = BeautifulSoup(urlopen("https://www.csgodatabase.com/weapons/").read(), features="html.parser")
    div_elements = soup.find_all("div", attrs = {"class" : "weaponBox"})
    
    weapon_links = []
    weapon_names = []
    
    for element in div_elements:
        #Get link to weapon's skin
        a_element = element.find("a")
        link = "https://www.csgodatabase.com/" + a_element.get("href")
        
        #Get the weapon's name
        weapon_name = a_element.getText().replace(" ", "_")
        
        #Make dir in data with weapon's name
        filepath = os.path.join("..", "data", "weapon_model", weapon_name)
        os.makedirs(filepath,  exist_ok = True)
        
        #Open link to weapon's skins
        soup = BeautifulSoup(urlopen(link).read(), features="html.parser")
        #get div elements from site
        skins_div = soup.find_all("div", attrs = {"class" : "skin-box-container"})
        
        for skin_element in skins_div:
            for element in skin_element:
                img_element = skin_element.find("img")
                img_link = img_element.get("src")
            
            try:
                skin_name = re.findall(r"(?!.*/).*png", img_link)[0]
                prrm -r
                image_path_model = os.path.join(filepath, skin_name)
                urlretrieve(img_link, image_path_model)  
                
                image_path_all = os.path.join("..", "data", "all_weapons", skin_name)
                urlretrieve(img_link, image_path_all)
                
            except UnicodeEncodeError:
                pass
            
#Define behaviour when called from command line
if __name__ == "__main__":
    main()