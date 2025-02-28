import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

def resize(w_box, h_box, pil_image):
    w, h = pil_image.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


class Binary:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    @staticmethod
    def iterate_convergence(x, tolerance, max_iterations=100):
        def CM_(C0_f, C0_g, Ce_f, Ce_g, Ce_f_M):
            Ce_g_increase = (C0_g / (1000 - C0_f)) * (C0_f - Ce_f_M)
            Ce_g_M = Ce_g - Ce_g_increase
            Ce_f_increase = (C0_f / (1000 - C0_g)) * (C0_g - Ce_g_M)
            Ce_f_M = Ce_f - Ce_f_increase
            return Ce_f_M, Ce_g_M

        Ce_f_M, Ce_g_M = x[4], x[5]
        convergence_history = [(Ce_f_M, Ce_g_M)]
        iterations = 0

        while iterations < max_iterations:
            Ce_f_M_prev, Ce_g_M_prev = Ce_f_M, Ce_g_M
            Ce_f_M, Ce_g_M = CM_(x[2], x[3], x[4], x[5], Ce_f_M)
            iterations += 1

            diff_ef = abs(Ce_f_M - Ce_f_M_prev)
            diff_eg = abs(Ce_g_M - Ce_g_M_prev)

            if diff_ef < tolerance and diff_eg < tolerance or iterations >= max_iterations:
                return Ce_f_M, Ce_g_M, iterations, convergence_history

            convergence_history.append((Ce_f_M, Ce_g_M))

        raise RuntimeError("Convergence not achieved within the maximum number of iterations.")

    @staticmethod
    def calculate_convergence(row, tolerance):
        result_Ce_f, result_Ce_g, iterations, convergence_history = Binary.iterate_convergence(row, tolerance)
        return pd.Series({'Ce_f_M': result_Ce_f, 'Ce_g_M': result_Ce_g, 'iterations': iterations, 'history': convergence_history})

    @staticmethod
    def calculate_Qe(row):
        Qe_f = (row[2] - row[4]) * row[0] / row[1]
        Qe_g = (row[3] - row[5]) * row[0] / row[1]
        return pd.Series({'Qe_f': Qe_f, 'Qe_g': Qe_g})

    @staticmethod
    def calculate_Qe_M(row):
        Qe_f_M = (row[2] - row[6]) * row[0] / row[1]
        Qe_g_M = (row[3] - row[7]) * row[0] / row[1]
        return pd.Series({'Qe_f_M': Qe_f_M, 'Qe_g_M': Qe_g_M})

    @staticmethod
    def calculate_Selectivity(row):
        a = (row[11] * row[5]) / (row[12] * row[4])
        return pd.Series({'α': a})

    def run(self):
        df = pd.read_excel(self.input_file)
        tolerance = 1e-5

        convergence_results = df.apply(Binary.calculate_convergence, axis=1, args=(tolerance,))
        df['Ce_f_M'] = convergence_results['Ce_f_M']
        df['Ce_g_M'] = convergence_results['Ce_g_M']
        df['迭代次数'] = convergence_results['iterations']

        Qe_result = df.apply(Binary.calculate_Qe, axis=1)
        Qe_result_M = df.apply(Binary.calculate_Qe_M, axis=1)
        df['Qe_f'] = Qe_result['Qe_f']
        df['Qe_g'] = Qe_result['Qe_g']
        df['Qe_f_M'] = Qe_result_M['Qe_f_M']
        df['Qe_g_M'] = Qe_result_M['Qe_g_M']

        a_result = df.apply(Binary.calculate_Selectivity, axis=1)
        df['α'] = a_result
        df['history'] = convergence_results['history']

        df.to_excel(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")



class Ternary:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.ini_concen = None
        self.processed_data = None
        self.results = {}

    def data_pretreatment(self):
        df = pd.read_excel(self.input_file, index_col=0, header=0)
        ini_concen = df.iloc[:3, 0].to_numpy()
        eq_concen_df = df.iloc[:5, [1]]
        processed_data = {}
        for col in eq_concen_df.columns:
            processed_data[col] = {
                "eq_concen": np.array(eq_concen_df[col].iloc[:3]),
                "M": np.array(eq_concen_df[col].iloc[3:4]),
                "m": np.array(eq_concen_df[col].iloc[4:5])
            }
        self.ini_concen = ini_concen
        self.processed_data = processed_data

    def calculate_eq_concen(self,ini_concen, eq_concen, eq_concen_0):
        skip_indices = []
        delta_1 = []
        increase_1 = []
        eq_concen_1 = []

        delta_2 = []
        increase_2 = []
        eq_concen_2 = []

        delta_3 = []
        increase_3 = []
        eq_concen_3 = []

        negative_increase_detected = False

        for a, b in zip(ini_concen, eq_concen_0):  # 各组分吸附前后浓度变化值
            delta_1.append(a - b)
        filtered_delta_1 = [value for i, value in enumerate(delta_1) if i not in skip_indices]  # 筛选出除跳过的索引外的delta值
        corr_value_1 = max(filtered_delta_1)  # 筛选出的delta中的最大值
        skip_index_1 = delta_1.index(corr_value_1)  # 找到最大值在原list中的索引
        skip_indices.append(skip_index_1)
        for i, value in enumerate(ini_concen):
            if i == skip_index_1:
                increase_1.append(0)
            else:
                increase_1.append(ini_concen[i] / (1000 - ini_concen[skip_index_1]) * corr_value_1)
        if any(value < 0 for value in increase_1):
            negative_increase_detected = True
            return eq_concen_0,negative_increase_detected
        else:
            for a, b in zip(eq_concen, increase_1): # 这个位置始终用实验得到的平衡数值(定值，不参与迭代）减去increase_1
                eq_concen_1.append(a - b)

        for a, b in zip(ini_concen, eq_concen_1):
            delta_2.append(a - b)
        filtered_delta_2 = [value for i, value in enumerate(delta_2) if i not in skip_indices]
        corr_value_2 = max(filtered_delta_2)
        skip_index_2 = delta_2.index(corr_value_2)
        skip_indices.append(skip_index_2)
        for i, value in enumerate(ini_concen):
            if i == skip_index_2:
                increase_2.append(0)
            else:
                increase_2.append(ini_concen[i] / (1000 - ini_concen[skip_index_2]) * corr_value_2)
        if any(value < 0 for value in increase_2):
            negative_increase_detected = True
            return eq_concen_1,negative_increase_detected
        else:
            for a, b in zip(eq_concen_1, increase_2):
                eq_concen_2.append(a - b)

        for a, b in zip(ini_concen, eq_concen_2):
            delta_3.append(a - b)
        filtered_delta_3 = [value for i, value in enumerate(delta_3) if i not in skip_indices]  # 跳过的index不参比
        corr_value_3 = max(filtered_delta_3)
        skip_index_3 = delta_3.index(corr_value_3)
        skip_indices.append(skip_index_3)
        for i, value in enumerate(ini_concen):
            if i == skip_index_3:
                increase_3.append(0)
            else:
                increase_3.append(ini_concen[i] / (1000 - ini_concen[skip_index_3]) * corr_value_3)
        if any(value < 0 for value in increase_3):
            negative_increase_detected = True
            return eq_concen_2,negative_increase_detected
        else:
            for a, b in zip(eq_concen_2, increase_3):
                eq_concen_3.append(a - b)  

        return eq_concen_3,negative_increase_detected

    def iterate_convergence(self, ini_concen_, eq_concen_, tolerance=1e-6, max_iterations=100):
        ini_concen, eq_concen = ini_concen_, eq_concen_
        eq_concen_0 = eq_concen 
        convergence_history = [] 
        iterations = 0
        eq_concen_3 = []
        negative_increase_final = False
        Convergy = False
        while iterations < max_iterations:
            eq_concen_3, negative_increase= self.calculate_eq_concen(ini_concen, eq_concen, eq_concen_0)
            if negative_increase:
                negative_increase_final = True
                messagebox.showinfo("Iteration Stopped", "Negative increase detected, iteration stopped.")
                break
            if np.all(np.abs(np.array(eq_concen_3) - np.array(eq_concen_0)) < tolerance):
                break
            eq_concen_0 = eq_concen_3
            convergence_history.append(eq_concen_0)
            iterations += 1

        return eq_concen_3, iterations, convergence_history,negative_increase_final

    @staticmethod
    def cal_qe(ini_concen, eq_concen_3, M, m):
        ini_concen_arr = np.array(ini_concen)
        eq_concen_3_arr = np.array(eq_concen_3)
        return (ini_concen_arr - eq_concen_3_arr) * M / m

    def run(self):
        self.data_pretreatment()

        for key in self.processed_data.keys():
            eq_concen = self.processed_data[key]['eq_concen']
            M = self.processed_data[key]['M']
            m = self.processed_data[key]['m']

            eq_concen_3, iterations, convergence_history,negative_increase = self.iterate_convergence(self.ini_concen,eq_concen)
            
            if negative_increase: #判断是否出现负值
                messagebox.showinfo("Iteration Stopped", f"Negative increase detected for {key}, using last valid result.")

            qe = self.cal_qe(self.ini_concen, eq_concen_3, M, m)
            
            self.results[key] = {
                'eq_concen_3': eq_concen_3,
                'iterations': iterations,
                'convergence_history': convergence_history,
                'qe': qe,
                'negative_increase': negative_increase
            }

    def save_results(self):
        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            for key, data in self.results.items():
                df = pd.DataFrame({
                    'eq_concen_3': [data['eq_concen_3']],
                    'iterations': [data['iterations']],
                    'convergence_history': [str(data['convergence_history'])],
                    'qe': [data['qe']],
                    'negative_increase': [data['negative_increase']]
                })
                df.to_excel(writer, sheet_name=key, index=False)
        print(f"Results saved to {self.output_file}")




class Quaternary:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.ini_concen = None
        self.processed_data = None
        self.results = {}

    def data_pretreatment(self):
        df = pd.read_excel(self.input_file, index_col=0, header=0)
        ini_concen = df.iloc[:4, 0].to_numpy()
        eq_concen_df = df.iloc[:6, [1]]
        processed_data = {}
        for col in eq_concen_df.columns:
            processed_data[col] = {
                "eq_concen": np.array(eq_concen_df[col].iloc[:4]),
                "M": np.array(eq_concen_df[col].iloc[4:5]),
                "m": np.array(eq_concen_df[col].iloc[5:6])
            }
        self.ini_concen = ini_concen
        self.processed_data = processed_data

    def calculate_eq_concen(self,ini_concen, eq_concen, eq_concen_0):
        skip_indices = []
        delta_1 = []
        increase_1 = []
        eq_concen_1 = []

        delta_2 = []
        increase_2 = []
        eq_concen_2 = []

        delta_3 = []
        increase_3 = []
        eq_concen_3 = []

        delta_4 = []
        increase_4 = []
        eq_concen_4 = []

        negative_increase_detected = False

        for a, b in zip(ini_concen, eq_concen_0):  # 各组分吸附前后浓度变化值
            delta_1.append(a - b)
        filtered_delta_1 = [value for i, value in enumerate(delta_1) if i not in skip_indices]  # 筛选出除跳过的索引外的delta值
        corr_value_1 = max(filtered_delta_1)  # 筛选出的delta中的最大值
        skip_index_1 = delta_1.index(corr_value_1)  # 找到最大值在原list中的索引
        skip_indices.append(skip_index_1)
        for i, value in enumerate(ini_concen):
            if i == skip_index_1:
                increase_1.append(0)
            else:
                increase_1.append(ini_concen[i] / (1000 - ini_concen[skip_index_1]) * corr_value_1)
        if any(value < 0 for value in increase_1):
            negative_increase_detected = True
            return eq_concen_0,negative_increase_detected
        else:
            for a, b in zip(eq_concen, increase_1): # 这个位置始终用实验得到的平衡数值(定值，不参与迭代）减去increase_1
                eq_concen_1.append(a - b)

        for a, b in zip(ini_concen, eq_concen_1):
            delta_2.append(a - b)
        filtered_delta_2 = [value for i, value in enumerate(delta_2) if i not in skip_indices]
        corr_value_2 = max(filtered_delta_2)
        skip_index_2 = delta_2.index(corr_value_2)
        skip_indices.append(skip_index_2)
        for i, value in enumerate(ini_concen):
            if i == skip_index_2:
                increase_2.append(0)
            else:
                increase_2.append(ini_concen[i] / (1000 - ini_concen[skip_index_2]) * corr_value_2)
        if any(value < 0 for value in increase_2):
            negative_increase_detected = True
            return eq_concen_1,negative_increase_detected
        else:
            for a, b in zip(eq_concen_1, increase_2): 
                eq_concen_2.append(a - b)

        for a, b in zip(ini_concen, eq_concen_2):
            delta_3.append(a - b)
        filtered_delta_3 = [value for i, value in enumerate(delta_3) if i not in skip_indices]  # 跳过的index不参比
        corr_value_3 = max(filtered_delta_3)
        skip_index_3 = delta_3.index(corr_value_3)
        skip_indices.append(skip_index_3)
        for i, value in enumerate(ini_concen):
            if i == skip_index_3:
                increase_3.append(0)
            else:
                increase_3.append(ini_concen[i] / (1000 - ini_concen[skip_index_3]) * corr_value_3)
        if any(value < 0 for value in increase_3):
            negative_increase_detected = True
            return eq_concen_3,negative_increase_detected
        else:
            for a, b in zip(eq_concen_2, increase_3): 
                eq_concen_3.append(a - b)

        for a, b in zip(ini_concen, eq_concen_3):
            delta_4.append(a - b)
        filtered_delta_4 = [value for i, value in enumerate(delta_4) if i not in skip_indices]  # 跳过的index不参比
        corr_value_4 = max(filtered_delta_4)
        skip_index_4 = delta_4.index(corr_value_4)
        skip_indices.append(skip_index_4)
        for i, value in enumerate(ini_concen):
            if i == skip_index_4:
                increase_4.append(0)
            else:
                increase_4.append(ini_concen[i] / (1000 - ini_concen[skip_index_4]) * corr_value_4)
        if any(value < 0 for value in increase_4):
            negative_increase_detected = True
            return eq_concen_4,negative_increase_detected
        else:
            for a, b in zip(eq_concen_3, increase_4): 
                eq_concen_4.append(a - b)
            
        return eq_concen_4,negative_increase_detected

    def iterate_convergence(self, ini_concen_, eq_concen_, tolerance=1e-6, max_iterations=100):
        ini_concen, eq_concen = ini_concen_, eq_concen_
        eq_concen_0 = eq_concen 
        convergence_history = [] 
        iterations = 0
        eq_concen_4 = []
        negative_increase_final = False
        Convergy = False
        while iterations < max_iterations:
            eq_concen_4, nagative_increase = self.calculate_eq_concen(ini_concen, eq_concen, eq_concen_0)
            if nagative_increase:
                negative_increaase_final = True
                messagebox.showinfo("Iteration Stopped", "negative increase detected, iteration stopped")
                break
            if np.all(np.abs(np.array(eq_concen_4) - np.array(eq_concen_0)) < tolerance):
                break
            eq_concen_0 = eq_concen_4
            convergence_history.append(eq_concen_0)
            iterations += 1

        return eq_concen_4, iterations, convergence_history, negative_increase_final

    @staticmethod
    def cal_qe(ini_concen, eq_concen_4, M, m):
        ini_concen_arr = np.array(ini_concen)
        eq_concen_4_arr = np.array(eq_concen_4)
        return (ini_concen_arr - eq_concen_4_arr) * M / m

    def run(self):
        self.data_pretreatment()

        for key in self.processed_data.keys():
            eq_concen = self.processed_data[key]['eq_concen']
            M = self.processed_data[key]['M']
            m = self.processed_data[key]['m']

            eq_concen_4, iterations, convergence_history, negative_increase = self.iterate_convergence(self.ini_concen,eq_concen)

            if negative_increase:
                messagebox.showinfo("Iteration stopped")

            qe = self.cal_qe(self.ini_concen, eq_concen_4, M, m)

            self.results[key] = {
                'eq_concen_4': eq_concen_4,
                'iterations': iterations,
                'convergence_history': convergence_history,
                'qe': qe,
                'negative_increase':negative_increase

            }

    def save_results(self):
        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            for key, data in self.results.items():
                df = pd.DataFrame({
                    'eq_concen_4': [data['eq_concen_4']],
                    'iterations': [data['iterations']],
                    'convergence_history': [str(data['convergence_history'])],
                    'qe': [data['qe']],
                    'negative_increase':[data['negative_increase']]
                })
                df.to_excel(writer, sheet_name=key, index=False)
        print(f"Results saved to {self.output_file}")



















# GUI Integration
def select_file_and_execute_binary():
    io = filedialog.askopenfilename(title='Select your file')
    if io:
        execute_window = tk.Toplevel(window)
        execute_window.title('Binary')
        execute_window.geometry('550x200')

        canvas = tk.Canvas(execute_window, bg='white', height=150, width=500)
        pil_image = Image.open(r'glucose.gif')
        pil_image_resized = resize(150, 150, pil_image)
        tk_image = ImageTk.PhotoImage(pil_image_resized)
        image = canvas.create_image(250, 0, anchor='n', image=tk_image)
        canvas.pack()

        def run_binary():
            output_file = "binary_results.xlsx"
            binary = Binary(io, output_file)
            binary.run()
            messagebox.showinfo(message='Computation complete and results saved to binary_results.xlsx.')

        btn_execute = tk.Button(execute_window, text='Run', command=run_binary)
        btn_execute.place(x=210, y=160)

        execute_window.mainloop()

def select_file_and_execute_ternary():
    io = filedialog.askopenfilename(title='Select your file')
    if io:
        execute_window = tk.Toplevel(window)
        execute_window.title('ternary')
        execute_window.geometry('550x200')

        canvas = tk.Canvas(execute_window, bg='white', height=150, width=500)
        pil_image = Image.open(r'mof2.gif')
        pil_image_resized = resize(150, 150, pil_image)
        tk_image = ImageTk.PhotoImage(pil_image_resized)
        image = canvas.create_image(250, 0, anchor='n', image=tk_image)
        canvas.pack()

        def run_ternary():
            output_file = "ternary_results.xlsx"
            ternary = Ternary(io, output_file)
            ternary.run()
            ternary.save_results()
            messagebox.showinfo(message='Computation complete and results saved to results.xlsx.')

        btn_normal = tk.Button(execute_window, text='run', command=run_ternary)
        btn_normal.place(x=210, y=160)

        execute_window.mainloop()

# Main GUI
def select_file_and_execute_quaternary():
    io = filedialog.askopenfilename(title='Select your file')
    if io:
        execute_window = tk.Toplevel(window)
        execute_window.title('quaternary')
        execute_window.geometry('550x200')

        canvas = tk.Canvas(execute_window, bg='white', height=150, width=500)
        pil_image = Image.open(r'mof2.gif')
        pil_image_resized = resize(150, 150, pil_image)
        tk_image = ImageTk.PhotoImage(pil_image_resized)
        image = canvas.create_image(250, 0, anchor='n', image=tk_image)
        canvas.pack()

        def run_quaternary():
            output_file = "quaternary_results.xlsx"
            quaternary = Quaternary(io, output_file)
            quaternary.run()
            quaternary.save_results()
            messagebox.showinfo(message='Computation complete and results saved to results.xlsx.')

        btn_normal = tk.Button(execute_window, text='run', command=run_quaternary)
        btn_normal.place(x=210, y=160)

        execute_window.mainloop()

window = tk.Tk()
window.title('select a function')
window.geometry('550x250')

canvas = tk.Canvas(window, bg='white', height=150, width=500)
pil_image = Image.open(r'mof1.gif')
pil_image_resized = resize(150, 150, pil_image)
tk_image = ImageTk.PhotoImage(pil_image_resized)
image = canvas.create_image(250, 0, anchor='n', image=tk_image)
canvas.pack()

combo = ttk.Combobox(window)
combo['values'] = ('Please select a function', 'Binary', 'Ternary','Quaternary')
combo.current(0)
combo.place(x=200, y=170)

def on_confirm():
    selection = combo.get()
    if selection == 'Binary':
        select_file_and_execute_binary()
        return
    if selection == 'Ternary':
        select_file_and_execute_ternary()
        return    
    if selection == 'Quaternary':
        select_file_and_execute_quaternary()
        return
    messagebox.showinfo('Prompt', 'Please select a valid function！')

btn_confirm = tk.Button(window, text='confirm', command=on_confirm)
btn_confirm.place(x=250, y=210)

window.mainloop()