#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import unicode_literals

import re
from optparse import OptionParser, OptionGroup
import shutil
import os
import difflib
import glob
import operator
import numpy as np
import face_recognition



def parse_commandline():
    parser = OptionParser()
    
    usage = "python3 asc_pipeline_npy.py [options]"
    version = "1.0"
    description = "%prog finds similar faces in Astrodatabank"
    epilog = "Written by Svetlana Stoma / @astroloxplore (2022)"
    parser = OptionParser(usage=usage, description=description,
                          version="%prog "+version, epilog=epilog)
 
    parser.add_option("-s", "--score",  dest="similarity_score", type="float",\
    help="similarity score threshold, where =0 is the same image, default is (0.0-0.6] - considered similar, and > 0.6 - dissimilar")
    parser.set_defaults(similarity_score=0.60)
    
    
    # get the options:
    (options, args) = parser.parse_args()

    # check for any leftover command line arguments:
    if len(args):
        warning("Ignoring additional arguments "+str(args))
    
    # clean up (recommended):
    del(parser)
    return options




def main():

    # get command line options
    options = parse_commandline()
    
    
    
   
    # ------------------- compare input images to ones in DB -----------------------
    
    # collect images in the present folder
    images = glob.glob('*.JPG') + glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.JPEG') + glob.glob('*.png') + glob.glob('*.PNG')

    nr_input_images = len(images)
    
    print('Detected input images:', images)

    name = images[0].split('.')[0]
    
    adb_npy = 'ADB_images_npy/images_' # database of npy matrices of faces
    
    f = open('face_similarity_outputs.txt', 'w') # to store similarity scores

    for i in images:
        
        i_name = i.split('.')[0] # remove the image extension
        image_to_match = face_recognition.load_image_file(i)
        
        try:
            image_to_match_encodings = face_recognition.face_encodings(image_to_match)[0] # first face if several detected
            
            for letter in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']:
                
                adb_im_fold = adb_npy+letter+'/'
                
                adb_images = glob.glob(adb_im_fold+'*.JPG.npy') + glob.glob(adb_im_fold+'*.jpg.npy') + glob.glob(adb_im_fold+'*.jpeg.npy') + glob.glob(adb_im_fold+'*.JPEG.npy') + glob.glob(adb_im_fold+'*.png.npy') + glob.glob(adb_im_fold+'*.PNG.npy')
            
                
                for adb_image in adb_images:
                
                    saved_encodings = np.load(adb_image)
                    
                    face_distance = face_recognition.face_distance([saved_encodings], image_to_match_encodings)
                    
                    adb_image_name = adb_image.split('/')[-1].split('.')[0]
                    
                    f.write(i_name+' '+adb_image_name+' '+str(float(face_distance[0]))+'\n')
            
        except IndexError:
            print('Could not read a face from input photo', i)     
        
    f.close()
    
    



    # ------------------- read created face_similarity_outputs.txt and select best matches -----------------------
    
    try:
        outputs = open('face_similarity_outputs.txt').read().splitlines()
    except IOError:
        print("ERROR: cannot open or read outputs.txt file")
        exit(-1)


    top_image_id_score = {}


    for rec in outputs:
    
        record = rec.split(' ')
        
        id_name = record[0]
        adb_name = record[1]
        score = float(record[2])
        
        if score < options.similarity_score:
            
            id_split = adb_name.split('_')
            
            try:
                x = int(id_split[-1]) # check if last item is a number
                adb_name = "_".join(id_split[:-1])
            except ValueError:
                pass

            if adb_name in top_image_id_score:
                first_score = top_image_id_score[adb_name]
                if float(first_score) > score: # then update
                    top_image_id_score[adb_name] = score
            else:    
                top_image_id_score[adb_name] = score
                
            

    # ----------------- best match -------------------
    matched_id_list = list(set(top_image_id_score.keys()))
    sorted_top_image_ids = sorted(top_image_id_score, key=top_image_id_score.get)
    print('\nMost similar IDs:')
    for x in sorted_top_image_ids[0:10]:
        print(x)



    
    
        
if __name__ == "__main__":
    main()
    


