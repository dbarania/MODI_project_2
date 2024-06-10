import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

def load_data(file_path:str)->np.ndarray:
    return np.loadtxt(file_path)

def divide_data(data:np.ndarray)->list[np.ndarray]:
    return np.vsplit(data,2)

def save_fig(fig:Figure,filename:str)->None:
    img_directory = "images"
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
    path = os.path.join(img_directory,filename)
    fig.savefig(path,dpi=400)

def sum_of_squared_error(y_data:np.ndarray,y_predicted:np.ndarray)->float:
    diff = np.power(y_predicted-y_data,2)
    return np.sum(diff)

def setup_figure(title:str,x_label:str="$u$",y_label:str="$y$",x_limits:tuple=(-1.1,1.1),y_limits:tuple=(-2.5,8))->tuple[Figure,Axes]:
    def comma_formatter(x, pos):
        return str(x).replace('.', ',')

    fig,ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_xlim(x_limits)

    ax.set_ylabel(y_label)
    ax.set_ylim(y_limits)

    if title:
        print("TITLE")
        ax.set_title(title)
    
    ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    ax.grid(True )
    return fig,ax


def add_plot(ax:Axes,X,Y,format:str,label:str="")->None:
    ax.plot(X,Y,format,label=label)