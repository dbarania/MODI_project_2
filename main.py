from utils import *
from static import *
from dynamic import *
if __name__=="__main__":
    dane_stat = load_data('dane/danestat69.txt')
    dane_stat_ucz, dane_stat_wer = divide_data(dane_stat)

    exercise_1a(dane_stat_ucz,dane_stat_wer)
    
    exercise_1b(dane_stat_ucz, dane_stat_wer)
    
    exercise_1c(dane_stat_ucz,dane_stat_wer)

    dane_dyn_ucz = load_data('dane/danedynucz69.txt')
    dane_dyn_wer = load_data('dane/danedynwer69.txt')
    
    exercise_2a(dane_dyn_ucz,dane_dyn_wer)

    exercise_2b(dane_dyn_ucz,dane_dyn_wer)
    exercise_2b(dane_dyn_ucz,dane_dyn_wer,recurssive=True)

    exercise_2c(dane_dyn_ucz,dane_dyn_wer)
    exercise_2c(dane_dyn_ucz,dane_dyn_wer,recurssive=True)
