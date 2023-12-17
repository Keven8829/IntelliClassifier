import tkinter as tk
from tkinter.filedialog import *
from PIL import Image,ImageTk
from tkinter import ttk
from main import *
from emnist_recognition import ImageRecognitionModel
import os
import main_ui

class Zty:
    def __init__(self):
        self.model = ImageRecognitionModel(img_height, img_width)
        self.switch = True
        
    def built_window(self, *args):
        self.model.load_model(filename_index=0)
        self.window=tk.Tk()
        self.window.title("Image Recognition") 
        width,height=1145,650
        width_max,height_max=self.window.maxsize()
        s_center="%dx%d+%d+%d"%(width,height,(width_max-width)/2,(height_max-height)/2)
        self.window.geometry(s_center)
        self.window.resizable(0,0)
        
        self.call_function()
        self.window.mainloop()

    def built_frame(self):
        self.frame=tk.Frame(self.window,bg="#E4DFD7",width=1145,height=650)
        self.frame.pack()
    
    def built_frame_button(self):
        self.frame_button=tk.LabelFrame(self.frame,text="Buttons",font=("Times New Roman", 15),foreground="white",bd=8,width=270,height=600,relief="ridge",bg="#AFB79C")
        self.frame_button.place(x=150,y=325,anchor="center")

    def built_frame_show(self):
        self.frame_show=tk.LabelFrame(self.frame,text="Show",font=("Times New Roman", 15),foreground="white",bd=8,width=270,height=600,relief="ridge",bg="#ABB8D1")
        self.frame_show.place(x=985,y=325,anchor="center")

    def built_frame_log(self):
        self.frame_log=tk.LabelFrame(self.frame,text="Log",font=("Times New Roman", 12),foreground="white",bd=5,width=230,height=425,relief="ridge",bg="#ABB8D1")
        self.frame_log.place(x=985,y=265,anchor="center")

    def built_frame_result(self):
        self.frame_result=tk.LabelFrame(self.frame,text="Result",font=("Times New Roman", 12),foreground="white",bd=5,width=230,height=126,relief="ridge",bg="#ABB8D1")
        self.frame_result.place(x=985,y=542,anchor="center")

    def built_frame_canvas(self):
        self.frame_cavans=tk.LabelFrame(self.frame,text="Picture",font=("Times New Roman", 15),foreground="white",bd=8,width=515,height=515,relief="ridge",bg="#D3A3A0")
        self.frame_cavans.place(x=572.5,y=280,anchor="center")

    def built_frame_combobox(self):
        self.frame_combobox=tk.LabelFrame(self.frame,text="Optimizer",foreground="white",font=("Times New Roman", 15),bd=8,width=515,height=80,relief="ridge",bg="#E7AE7c")
        self.frame_combobox.place(x=572.5,y=580,anchor="center")

    def built_canvas(self):
        self.image_canvas=tk.Canvas(self.frame,width=470,height=470,bg="#B68784")
        self.image_canvas.place(x=572.5,y=285,anchor="center")

    def open_folder(self):
        self.filename=askopenfilename()
        self.load_opened_image()
        image_name = os.path.basename(self.filename)
        if self.switch:
            self.first_line_log("Opened File: {}".format(image_name))
            self.switch = False
        else:
            self.log("Opened File: {}".format(image_name))

    def load_opened_image(self):
        global original_image
        self.pil_image=Image.open(self.filename)
        self.pil_image.thumbnail((470,470))
        original_image=ImageTk.PhotoImage(self.pil_image)
        self.image_canvas.config(width=470,height=470)
        self.image_canvas.create_image(235,235,anchor="center",image=original_image)

    def built_button_open(self):
        self.button_open=tk.Button(self.frame_button,text="Open Folder",bd=6,relief="ridge",width=20,height=5,bg="#96A284",fg="white",command=self.open_folder,font=("Times New Roman", 12))
        self.button_open.place(x=130,y=75,anchor="center")

    def built_button_detect(self):
        self.button_detect=tk.Button(self.frame_button,text="Detect",bd=6,relief="ridge",width=20,height=5,bg="#96A284",fg="white",command=self.check_button_detect,font=("Times New Roman", 12))
        self.button_detect.place(x=130,y=213,anchor="center")

    def check_button_detect(self):
        if self.switch:
            self.first_line_log("Clicked Detect")
            self.switch = False
        else:
            self.log(f"Clicked Detect")
        self.run_model()
        
    def built_button_statistic(self):
        self.button_statistic=tk.Button(self.frame_button,text="Statistic",bd=6,relief="ridge",width=20,height=5,bg="#96A284",fg="white",command=self.check_button_statistic,font=("Times New Roman", 12))
        self.button_statistic.place(x=130,y=352,anchor="center")

    def check_button_statistic(self):
        self.get_model=self.selection_model.get()
        if self.get_model == "Adam":
            data = 0
        elif self.get_model == "Adadelta":
            data = 1
        elif self.get_model == "Adagrad":
            data = 2
        else:
            data = 0
        self.model.statistic_history_data(data)
        if self.switch:
            self.first_line_log("Clicked Statistic")
            self.switch = False
        else:
            self.log(f"Clicked Statistic")

    def check_button_return(self):
        self.window.destroy()
        main_ui.RecognitionApp()

    def built_button_return(self):
        self.button_return=tk.Button(self.frame_button,text="Return",bd=6,relief="ridge",width=20,height=5,bg="#96A284",fg="white",command=self.check_button_return,font=("Times New Roman", 12))
        self.button_return.place(x=130,y=490,anchor="center")

    def show_model(self,*args):
        self.get_model=self.selection_model.get()
        return self.get_model
    
    def built_combobox_model(self):
        self.var_model = tk.StringVar()
        self.selection_model = ttk.Combobox(self.frame_combobox,width=36, height=10,textvariable=self.var_model, state="readonly",font=("Times New Roman", 16))
        self.selection_model["values"] = ("Adam", "Adagrad", "Adadelta")
        self.selection_model.place(x=240, y=23, anchor="center")
        self.selection_model.current(0)
        self.selection_model['state'] = 'readonly'

        # Set the initial placeholder
        self.var_model.set('--Optimizer--')
        self.selection_model['foreground'] = 'black'

        # Binding events to manage placeholder
        self.var_model.trace_add('write', lambda *args: "--Optimizer--")
        self.selection_model.bind('<<ComboboxSelected>>', self.show_model)

    def built_label_log(self):
        self.label_log = tk.Text(self.frame_log,bd=4,relief="ridge",width=22,height=18,bg="#808CA2",fg="white", font=("Times New Roman", 14))
        self.label_log.place(x=6,y=0,anchor="nw")

    def built_label_result(self):
        self.label_result=tk.Label(self.frame_result,bd=4,relief="ridge",width=20,height=4,bg="#808CA2",fg="white", font=("Times New Roman", 14))
        self.label_result.place(x=6,y=0,anchor="nw")
        
    def run_model(self):
        self.get_model=self.selection_model.get()
        if self.get_model == "Adam":
            data = 0
        elif self.get_model == "Adadelta":
            data = 1
        elif self.get_model == "Adagrad":
            data = 2
        else:
            data = 0
        self.model.load_model(data)
        img_path = rf'{self.filename}'
        img = Test_Image_Loader(img_height, img_width, img_path)
        result, accuracy= self.model.model_prediction(img)
        self.result(f"Prediction: {result}\nAccuracy: {accuracy}%")

    def log(self, message):
        self.label_log.config(state=tk.NORMAL)
        current_text = self.label_log.get("1.0", tk.END)
        new_text = current_text + message 
        self.label_log.delete("1.0", tk.END)
        self.label_log.insert(tk.END, new_text)
        self.label_log.config(state=tk.DISABLED)
    
    def first_line_log(self, message):
        self.label_log.config(state=tk.NORMAL)
        self.label_log.delete("1.0", tk.END)
        self.label_log.insert(tk.END, message)
        self.label_log.config(state=tk.DISABLED)

    def result(self, message):
        self.label_result.config(text=message)

    def make_log_readonly(self):
        self.label_log.config(state=tk.DISABLED)

    def call_function(self):
        self.built_frame()
        self.built_frame_button()
        self.built_frame_show()
        self.built_frame_log()
        self.built_frame_result()
        self.built_frame_canvas()
        self.built_frame_combobox()
        self.built_canvas()
        self.built_label_log()
        self.built_button_detect()
        self.built_button_statistic()
        self.built_button_open() 
        self.built_button_return()
        self.built_combobox_model()
        self.built_label_result()
        self.make_log_readonly()

if __name__ == "__main__":
    zty = Zty()
    zty.built_window()