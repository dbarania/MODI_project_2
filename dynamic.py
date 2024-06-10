import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from utils import *

def least_squares_dynamic_linear(data:np.ndarray,N:int = 1)->np.ndarray:
    Y = data[N:,1]
    M = []
    y = data[:,1]
    u = data[:,0]
    
    for i in range(N):
        M.append(y[i:-N+i])  # Append slices of y to M
    for i in range(N):
        M.append(u[i:-N+i])  # Append slices of u to M
    
    M = np.array(M).T  # Convert M to a NumPy array and transpose to get the correct shape

    return np.linalg.inv(M.T @ M) @ M.T @ Y

def get_row(p:int,N:int,y:np.ndarray,u:np.ndarray):
    row = []
    for i in range(1,N+1):
        row.append(y[p-i])
        row.append(u[p-i])
    return np.array(row)

def dynamic_least_squares(data:np.ndarray,N:int=1,poly_degree:int=1)->np.ndarray:
    Y = data[N:,1]
    y = data[:,1]
    u = data[:,0]
    M = []
    for i in range(N,data[:,1].size):
        row_base = get_row(i,N,y,u)
        row = np.concatenate([np.power(row_base, n) for n in range(1, poly_degree+1)])
        M.append(row)
    M = np.array(M)
    return np.linalg.inv(M.T @ M) @ M.T @ Y


def eval_model_linear(w:np.ndarray,data:np.ndarray,N:int=1,recursive:bool = False):
    u = data[:,0]
    if not recursive:
        M = []
        y = data[:,1]
    
        for i in range(N):
            M.append(y[i:-N+i])  # Append slices of y to M
        for i in range(N):
            M.append(u[i:-N+i])  # Append slices of u to M
        
        M = np.array(M).T 
        return M@w
    else:
        a,b = np.hsplit(w,2)
        y = np.hstack((data[:N,1],np.zeros(data[:,1].size-N)))
        for i in range(N, data[:,1].size):
            y[i] = y[i-N:i]@a+u[i-N:i]@b
        return y

def eval_dyn_model(w:np.ndarray,data:np.ndarray,N:int,poly_degree:int,recursive:bool):
    u = data[:,0]
    if not recursive:
        M = []
        y = data[:,1]
        for i in range(N,data[:,1].size):
            row_base = get_row(i,N,y,u)
            row = np.concatenate([np.power(row_base, n) for n in range(1, poly_degree+1)])
            M.append(row)
        M = np.array(M)
        return np.hstack((y[:N],M@w))
    else:
        y = np.hstack((data[:N,1],np.zeros(data[:,1].size-N)))
        for i in range(N,data[:,1].size):
            row_base = get_row(i,N,y,u)
            row = np.concatenate([np.power(row_base, n) for n in range(1, poly_degree+1)]) 
            y[i] = row@w
        return y

def exercise_2a(learn_data:np.ndarray,verify_data:np.ndarray,save:bool=True)->None:
    fig_learn,ax_learn = setup_figure("","$k$","$y,u$",(-1,2001),(-2,6.5))
    k_learn = range(0,learn_data[:,0].size)
    add_plot(ax_learn,k_learn,learn_data[:,1],'b-',label="Wyjście")
    add_plot(ax_learn,k_learn,learn_data[:,0],'r-',label="Sygnał sterujący")

    fig_verify,ax_verify = setup_figure("","$u$","$y$",(-1,2001),(-2,6.5))
    k_verify = range(0,verify_data[:,0].size)
    add_plot(ax_verify,k_verify,verify_data[:,1],'b-',label="Wyjście")
    add_plot(ax_verify,k_verify,verify_data[:,0],'r-',label="Sygnał sterujący")
    fig_learn.show()
    fig_verify.show()
    if save:
        save_fig(fig_learn,"2a_learning.png")
        save_fig(fig_verify,"2a_verify.png")


def exercise_2b(learn_data:np.ndarray,verify_data:np.ndarray,save:bool=True,recurssive:bool=False):
    f = open(f"2b_dynamic_{recurssive}.log",'w')

    for N in [1,2,3]:
        w = dynamic_least_squares(learn_data,N,1)
        f.write(f"For N {N}, w {w}\n")
        y_learn_predict = eval_dyn_model(w,learn_data,N,1,recurssive)
        error_learn_predict = sum_of_squared_error(learn_data[:,1],y_learn_predict)

        fig_learn,ax_learn = setup_figure("","$k$","$y$",(-1,2001),(-2,6.5))
        k = range(learn_data[:,1].size)
        add_plot(ax_learn,k,learn_data[:,1],'b-',"Obiekt")
        add_plot(ax_learn,k,y_learn_predict,'r--',"Model")


        y_verify_predict = eval_dyn_model(w,verify_data,N,1,recurssive)
        error_verify_predict = sum_of_squared_error(verify_data[:,1],y_verify_predict)

        fig_verify,ax_verify = setup_figure("","$k$","$y$",(-1,2001),(-2,6.5))
        k = range(verify_data[:,1].size)

        add_plot(ax_verify,k,verify_data[:,1],'b-',"Obiekt")
        add_plot(ax_verify,k,y_verify_predict,'r--',"Model")

        fig_learn.show()
        fig_verify.show()
        f.write(f"With N {N},for train data error: {error_learn_predict}\n")
        f.write(f"With N {N},for verify data error: {error_verify_predict}\n")

        ax_learn.legend()
        ax_verify.legend()

        if save:
            save_fig(fig_learn,f"2b_learning_{N}_{recurssive}.png")
            save_fig(fig_verify,f"2b_verify_{N}_{recurssive}.png")

        plt.close(fig_learn)
        plt.close(fig_verify)
    
    f.close()

def exercise_2c(learn_data:np.ndarray,verify_data:np.ndarray,save:bool=True,recurssive:bool=False):
    f = open(f"2c_dynamic_{recurssive}.log",'w')

    for N in [1,2,3]:
        for deg in [2,3,4,5]:
            w = dynamic_least_squares(learn_data,N,deg)
            f.write(f"For N {N}, degree {deg}, w {w}\n")
            y_learn_predict = eval_dyn_model(w,learn_data,N,deg,recurssive)
            error_learn_predict = sum_of_squared_error(learn_data[:,1],y_learn_predict)

            fig_learn,ax_learn = setup_figure("","$k$","$y$",(-1,2001),(-2,6.5))
            k = range(learn_data[:,1].size)
            add_plot(ax_learn,k,learn_data[:,1],'b-',"Obiekt")
            add_plot(ax_learn,k,y_learn_predict,'r--',"Model")


            y_verify_predict = eval_dyn_model(w,verify_data,N,deg,recurssive)
            error_verify_predict = sum_of_squared_error(verify_data[:,1],y_verify_predict)

            fig_verify,ax_verify = setup_figure("","$k$","$y$",(-1,2001),(-2,6.5))
            k = range(verify_data[:,1].size)

            add_plot(ax_verify,k,verify_data[:,1],'b-',"Obiekt")
            add_plot(ax_verify,k,y_verify_predict,'r--',"Model")

            fig_learn.show()
            fig_verify.show()
            f.write(f"With N {N}, degree {deg}, for train data error: {error_learn_predict}\n")
            f.write(f"With N {N}, degree {deg}, for verify data error: {error_verify_predict}\n")

            ax_learn.legend()
            ax_verify.legend()

            if save:
                save_fig(fig_learn,f"2c_learning_{N}_{deg}_{recurssive}.png")
                save_fig(fig_verify,f"2c_verify_{N}_{deg}_{recurssive}.png")

            plt.close(fig_learn)
            plt.close(fig_verify)
    
    f.close()