import pickle as pk
import numpy as np
import os, time
import matplotlib.pyplot as plt
import random

color_lib = {
    "default": ['E0E000', 'FF8600', 'FF0000', 'B21212', '1485CC', '07378A', '7003CC', '320F73'],
    "binary": ['FF0000', '00FF00']
}

def rest():
    while True:
        time.sleep(10)

class color_generator:
    def __init__(self, lib="default"):
        self.map = {}
        self.ptr = 0
        if not lib == "binary":
            lib = "default"
        self.lib = color_lib[lib]

    def generate_color(self, key=None):
        if key is None:
            key = random.random()
        if key not in self.map.keys():
            if self.ptr >= len(self.lib):
                self.ptr = 0
            self.map[key] = self.lib[self.ptr]
            self.ptr += 1
        return '#' + self.map[key]

def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print "Pickle file cannot be opened."
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print 'load_pickle failed once, trying again'
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()

def make_dir():
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_plot(X, Y, title="", X_label="x", Y_label = "y", fn="figure", USE_GRID=True, USE_AXIS="on",
    X_LIM=None, Y_LIM=None, color=None):
    fig, ax = plt.subplots()

    for i in range(len(Y)):
        # cg = color_generator()
        # color = cg.generate_color()
        ax.plot(X[i], Y[i], linewidth=2.0) # , color=color
        ax.grid(USE_GRID)
        plt.title(title)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)
    plt.axis(USE_AXIS)
    plt.show()
    # plt.savefig(fn+".png")