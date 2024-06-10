import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from utils import *

u_lin = np.linspace(-1,1,100)


def least_squares_linear(data:np.ndarray)->np.ndarray:
    Y = data[:,1]
    U = data[:,0]
    a = np.ones(100)
    M = np.vstack((a,U)).T
    return np.linalg.inv(M.T @ M)@M.T@Y

def least_squares_polynomial(data:np.ndarray,N:int=3)->np.ndarray:
    Y = data[:,1]
    U = data[:,0]
    M = np.array([np.power(U,i) for i in range(N+1)]).T
    return np.linalg.inv(M.T @ M)@M.T@Y


def eval_polynomial(w:np.ndarray,input:np.ndarray=u_lin)->np.ndarray:
    return np.polyval(np.flip(w),input)


def exercise_1a(learn_data:np.ndarray,verify_data:np.ndarray,save:bool=True)->None:
    fig_learn,ax_learn = setup_figure("","$u$","$y$")
    add_plot(ax_learn,learn_data[:,0],learn_data[:,1],'o')
    fig_verify,ax_verify = setup_figure("","$u$","$y$")

    add_plot(ax_verify,verify_data[:,0],verify_data[:,1],'o')
    fig_learn.show()
    fig_verify.show()
    if save:
        save_fig(fig_learn,"1a_learning.png")
        save_fig(fig_verify,"1a_verify.png")


def exercise_1b(learn_data:np.ndarray,verify_data:np.ndarray,save:bool=True):
    f = open("1b_linear.log",'w')
    w = least_squares_linear(learn_data)
    f.write(f"w returned: {w}\n")
    
    y_characteristic = eval_polynomial(w)

    fig_characteristic, ax_characteristic = setup_figure("")
    add_plot(ax_characteristic,u_lin,y_characteristic,'-')
    fig_characteristic.show()

    y_learn_predict = eval_polynomial(w,learn_data[:,0])
    error_learn_predict = sum_of_squared_error(learn_data[:,1],y_learn_predict)

    fig_learn,ax_learn = setup_figure("")
    add_plot(ax_learn,learn_data[:,0],learn_data[:,1],'o',"Obiekt")
    add_plot(ax_learn,learn_data[:,0],y_learn_predict,'r.',"Model")

    y_verify_predict = eval_polynomial(w,verify_data[:,0])
    error_verify_predict = sum_of_squared_error(learn_data[:,1],y_verify_predict)

    fig_verify,ax_verify = setup_figure("")
    add_plot(ax_verify,verify_data[:,0],verify_data[:,1],'o',"Obiekt")
    add_plot(ax_verify,verify_data[:,0],y_verify_predict,'.',"Model")

    fig_learn.show()
    fig_verify.show()
    f.write(f"For train data error: {error_learn_predict}\n")
    f.write(f"For verify data error: {error_verify_predict}\n")

    ax_characteristic.legend()
    ax_learn.legend()
    ax_verify.legend()
    if save:
        save_fig(fig_characteristic,"1b_characteristic.png")
        save_fig(fig_learn,"1b_learning.png")
        save_fig(fig_verify,"1b_verify.png")
    plt.close(fig_characteristic)
    plt.close(fig_learn)
    plt.close(fig_verify)
    f.close()

def exercise_1c(learn_data:np.ndarray,verify_data:np.ndarray,save:bool=True):
    f = open("1c_polynomial.log",'w')
    for i in range(2,13,1):
        w = least_squares_polynomial(learn_data,i)
        f.write(f"For {i} degree returned {w}\n")

        y_characteristic = eval_polynomial(w)
        fig_characteristic, ax_characteristic = setup_figure("")
        add_plot(ax_characteristic,u_lin,y_characteristic,'-')
        fig_characteristic.show()

        y_learn_predict = eval_polynomial(w,learn_data[:,0])
        error_learn_predict = sum_of_squared_error(learn_data[:,1],y_learn_predict)

        fig_learn,ax_learn = setup_figure("")
        add_plot(ax_learn,learn_data[:,0],learn_data[:,1],'o',"Obiekt")
        add_plot(ax_learn,learn_data[:,0],y_learn_predict,'r.',"Model")

        y_verify_predict = eval_polynomial(w,verify_data[:,0])
        error_verify_predict = sum_of_squared_error(learn_data[:,1],y_verify_predict)

        fig_verify,ax_verify = setup_figure("")
        add_plot(ax_verify,verify_data[:,0],verify_data[:,1],'o',"Obiekt")
        add_plot(ax_verify,verify_data[:,0],y_verify_predict,'.',"Model")

        fig_learn.show()
        fig_verify.show()
        f.write(f"Polynomial degree: {i}\t For train data error: {error_learn_predict}\n")
        f.write(f"Polynomial degree: {i}\t For verify data error: {error_verify_predict}\n")
        ax_characteristic.legend()
        ax_learn.legend()
        ax_verify.legend()
        if save:
            save_fig(fig_characteristic, f"1c_characteristic_deg{i}.png")
            save_fig(fig_learn, f"1c_learning_deg{i}.png")
            save_fig(fig_verify, f"1c_verify_deg{i}.png")
        plt.close(fig_characteristic)
        plt.close(fig_learn)
        plt.close(fig_verify)
    f.close()