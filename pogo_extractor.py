# standard lib
import argparse, pprint, math, os, shutil, re

# math libs
import numpy as np

# image processing
import cv2
import pytesseract

# plotting
import matplotlib.pyplot as plt

# prog bar
from tqdm import tqdm

from utils import fix_name, ivs_to_cp

def plot_img(img, min_=0, max_=255):
    plt.imshow(img, cmap='gray', vmin=min_, vmax=max_)

def f_by_f(f_name):
    vidcap = cv2.VideoCapture(f_name)
    success,image = vidcap.read()
    count = 0
    while success:
        yield count, image
        success, image = vidcap.read()
        count += 1
        
def get_nframes(f_name):
    cap = cv2.VideoCapture(f_name)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_bboxes(frame):
    name = frame[400:600, 200:600, :]
    
    cp = frame[50:150, 180:475, :]
    
    ivs = frame[900:1150, 50:300, :]
    attack_bar = ivs[78:94, 25:220]
    defense_bar = ivs[135:151, 25:220]
    hp_bar = ivs[192:208, 25:220]
    
    return name, cp, attack_bar, defense_bar, hp_bar

def to_grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def binarize(np_arr, threshold=100):
    np_arr = to_grayscale(np_arr)
    sel = (np_arr - np_arr.min()) > threshold
    return 1 - sel

def get_blurr(np_arr):
    #gray = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
    #return np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))

    as_gray = to_grayscale(np_arr)
    
    # finds the difference between adjacement
    # pixels. eg
    # [0.1, 0.2] 
    # [0.1, 0.3] 
    # 
    # dx =   [[0. , 0.1],
    #        [0. , 0.1]]
    #
    # dy =   [[0.1, 0.1],
    #        [0.2, 0.2]]

    (dx, dy) = np.gradient(as_gray)

    # combine distance together into 
    # an array of distances
    gnorm = np.sqrt(dx**2 + dy**2)
    return np.average(gnorm) # average the result


def iv_to_int(np_arr):
    FULL_BAR_BASE = 382
    MAX_IV_VAL = 15

    np_arr = np_arr[8:10]
    np_arr = binarize(np_arr)
    return math.floor((np.sum(np_arr) / FULL_BAR_BASE) * MAX_IV_VAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', help='input filename', type=str)
    parser.add_argument('--video_start', help='frame at which the recording starts', type=int, default=20)
    parser.add_argument('--debug', help='save intermittent frames and results', action='store_true')
    args = parser.parse_args()

    f_name = args.fname
    video_start = args.video_start
    debug = args.debug

    non_alphabetical_re = re.compile("[^a-zA-Z]")
    non_numeric_re = re.compile("[^0-9]")

    if not os.path.exists(f_name):
        raise Exception("File not found", f_name)

    total_frames = get_nframes(f_name)
    extracted_pokemon = []
    prev_att, prev_def, prev_hp = -1, -1, -1
   
    if debug: 
        print("Writing debugging output to:", os.path.join(os.getcwd(), 'debug_output'))
        if os.path.exists('debug_output'):
            shutil.rmtree('debug_output')
        
        os.mkdir('debug_output')
        os.mkdir('debug_output/attack_vals')
        os.mkdir('debug_output/defense_vals')
        os.mkdir('debug_output/hp_vals')
        os.mkdir('debug_output/name_vals')
        os.mkdir('debug_output/cp_vals')


    with tqdm(total=total_frames) as pbar:
        for (count,frame) in f_by_f(f_name):
            pbar.set_description("Processing Frame %i" % count)
            pbar.update(1)
            
            if count < video_start:
                continue
            
            blurr = get_blurr(frame)
            if blurr < 4:
                continue

            name, cp, attack_bar, defense_bar, hp_bar = get_bboxes(frame)
            att, def_, hp = iv_to_int(attack_bar), iv_to_int(defense_bar), iv_to_int(hp_bar)

            # pytesseract is by far the slowest part of this loop
            # and typically we see the same pokemon many times in the same
            # loop, so to save time if we have seen these exact IV's 
            # in the previous frames, we skip
            if att == prev_att and def_ == prev_def and hp == prev_hp:
                continue

            if debug:
                plt.imsave(os.path.join('debug_output', 'attack_vals', 'att_{count}_{blurr}_{att}.jpg'.format(blurr=blurr, count=count, att=att)), attack_bar)
                plt.imsave(os.path.join('debug_output', 'defense_vals', 'def_{count}_{blurr}_{def_}.jpg'.format(blurr=blurr, count=count, def_=def_)), defense_bar)
                plt.imsave(os.path.join('debug_output', 'hp_vals', 'hp_{count}_{blurr}_{hp}.jpg'.format(blurr=blurr, count=count, hp=hp)), hp_bar)

            poke_name = pytesseract.image_to_string(name)
            poke_cp = pytesseract.image_to_string(cp)

            poke_name = non_alphabetical_re.sub("", poke_name).replace("HP", "")
            poke_cp = non_numeric_re.sub("", poke_cp)

            poke_name = fix_name(poke_name)

            if debug:
                plt.imsave(os.path.join('debug_output', 'name_vals', 'name_{name}.jpg'.format(name=poke_name)), name)
                plt.imsave(os.path.join('debug_output', 'cp_vals', 'cp_{cp}.jpg'.format(cp=poke_cp)), cp)
            
            if poke_cp == '' or poke_name == '':
                continue 

            possible_cp = ivs_to_cp(poke_name, hp, att, def_)

            # implies the name did not match any in our dataset
            if len(possible_cp) == 0:
                continue 

            if int(poke_cp) not in possible_cp:
                continue

            extracted_pokemon += [
                {
                    'name' : poke_name,
                    'cp' : poke_cp,
                    'attack' : att,
                    'defense' : def_,
                    'hp' : hp
                }
            ]

            prev_att, prev_def, prev_hp = att, def_, hp

    print("EXTRACTED POKEMON")
    pprint.pprint(extracted_pokemon)
