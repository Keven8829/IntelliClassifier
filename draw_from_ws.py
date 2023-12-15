import tkinter as tk
import customtkinter as ctk
import PIL
from tkinter import colorchooser as cc, messagebox as mb
from PIL import ImageTk, Image, ImageDraw
from emnist_recognition import load_and_predict_image
from Application import Classify_or_Recognise_Anything  # 导入 RecognitionApp 类

def drawing_window():
    class Drawing():
        def __init__(self, root):
            self.root = root
            self.root.title("Number and Alphabet detection (Drawing site)")
            self.root.geometry("1300x650+10+10")
            self.root.configure(background="white")
            self.root.resizable(False, False)
            self.label = None
            self.accuracy = None
            self.pen_colour = "black"
            self.message = ""
            
            self.back_button = None

        # Frames
        def frames1(self):
            self.frame1 = tk.Frame(self.root, width=800, height=200)
            self.frame1.grid(row=0, column=0)

            self.frame1_within = tk.LabelFrame(self.frame1, width=800, height=200, bg="white", bd=7, padx=6, pady=10)
            self.frame1_within.grid(row=0, column=0, sticky="nw")

            self.frame1_within.columnconfigure(0, minsize=250)
            self.frame1_within.columnconfigure(1, minsize=250)
            self.frame1_within.columnconfigure(2, minsize=250)
            self.frame1_within.rowconfigure(0, minsize=100)

            # Colour selecting frame
            self.colour_frame = tk.LabelFrame(self.frame1_within, text=" Pen Colour ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
            self.colour_frame.grid(row=0, column=0)
            # Colour buttons
            colours = ["red", "orange", "yellow", "green", "lightblue", "blue", "violet", "purple", "indigo", "black"]
            i=j=0
            for colour in colours: 
                tk.Button(self.colour_frame, bg=colour, bd=3, relief="ridge", width=4, command=lambda col1=colour:self.select_colour(col1)).grid(row=i, column=j)
                i+=1
                if i==2:
                    i=0
                    j+=1
            # Custom colour button
            self.custom_colour_button = tk.Button(self.frame1_within, text="Custom Colour", font=("Times New Roman", 12), bd=6, relief="ridge", bg="white", command=self.select_custom_colour, width=21, height=2)
            self.custom_colour_button.grid(row=1, column=0)

            # Button frame1
            self.button_frame1 = tk.LabelFrame(self.frame1_within, text=" Tools ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
            self.button_frame1.grid(row=0, column=1)
            # Eraser button
            self.erasor_button = tk.Button(self.button_frame1, text="Eraser", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.eraser, width=10, height=2)
            self.erasor_button.grid(row=0, column=0) 
            # Clear button
            self.clear_button = tk.Button(self.button_frame1, text="Clear", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.clearing, width=10, height=2)
            self.clear_button.grid(row=0, column=1) 
            # Button frame2
            self.button_frame2 = tk.LabelFrame(self.frame1_within, bd=5, relief="ridge", bg="white")
            self.button_frame2.grid(row=1, column=1)
            # Save button
            self.save = tk.Button(self.button_frame2, text="Save", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.saving, width=10, height=2)
            self.save.grid(row=0, column=0)
            # Statistics button
            self.statistic_button = tk.Button(self.button_frame2, text="Statistics", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.statistics, width=10, height=2)
            self.statistic_button.grid(row=0, column=1) 
        
            # Pen scale frame
            self.scale_frame = tk.LabelFrame(self.frame1_within, text=" Size of pen/eraser ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
            self.scale_frame.grid(row=0, column=2)
            # Pen scale
            self.pen_scale = tk.Scale(self.scale_frame, orient="horizontal", from_=1, to=25, length=250)
            self.pen_scale.set(1)
            self.pen_scale.grid(row=0)

            # Detect button frame
            self.detect_button_frame = tk.LabelFrame(self.frame1_within, bd=5, relief="ridge", bg="white")
            self.detect_button_frame.grid(row=1, column=2)
            # Detect button
            self.detect_button = tk.Button(self.detect_button_frame, text="Detect", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.predict, width=25, height=2)
            self.detect_button.grid(row=0, column=0)
            
            self.back_button = tk.Button(self.frame1_within, text="Back", font=("Times New Roman", 12), bd=6, relief="ridge", bg="white", command=self.back_to_main, width=21, height=2)
            self.back_button.grid(row=2, column=0)
            
        def predict(self): 
            self.label, self.accuracy = load_and_predict_image('image.png')
            self.message = f"Predicted Label: {self.label}, Accuracy: {self.accuracy:.2f}"
            print(self.label, self.accuracy)
            self.update_log_text()

        def update_log_text(self):
            # Update the text of self.log_text
            self.log_text.config(text=self.message)


        def frames2(self):
            # Canvas frame
            self.frame2 = tk.Frame(self.root, width=800, height=448)
            self.frame2.grid(row=1, column=0)

        def frames3(self):
            self.frame3 = tk.Frame(self.root, width=300, height=200)
            self.frame3.grid(row=0, column=1)

            self.frame3_within = tk.LabelFrame(self.frame3, bg="white", bd=7, padx=6, pady=10, width=300, height=200)
            self.frame3_within.grid(row=0, column=0, sticky="nw")

        def frames3_1(self):
            self.frame3_1 = tk.Frame(self.root, width=300, height=450)
            self.frame3_1.grid(row=1, column=1)

            self.frame3_1_within = tk.LabelFrame(self.frame3_1, bg="white", bd=7, padx=6, pady=10)
            self.frame3_1_within.grid(row=0, column=0, sticky="nw")
        

        # Functions
        def canvas_panel(self): # Canvas
            self.canvas = tk.Canvas(self.frame2, bg="white", bd=5, relief="groove", height=431, width=431)
            self.image1=PIL.Image.new("RGB", (431, 431), (255, 255, 255))
            self.draw=ImageDraw.Draw(self.image1)
            self.canvas.pack()
            self.canvas.bind("<B1-Motion>", self.painting)

        def painting(self, event): # Painting on canvas
            x1, y1 = (event.x-3), (event.y-3)
            x2, y2 = (event.x+3), (event.y+3)
            self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_colour, outline=self.pen_colour, width=self.pen_scale.get())
            self.draw.line((x1, y1, x2, y2), fill=self.pen_colour, width=self.pen_scale.get())

        def clearing(self): # For clearing canvas
            self.canvas.delete("all")
            self.image1=PIL.Image.new("RGB", (431, 431), (255, 255, 255))
            self.draw=ImageDraw.Draw(self.image1)

        def select_colour(self, col1): # Pen colour from default options
            self.pen_colour = col1
    
        def select_custom_colour(self): # Pen custom colour
            colour = cc.askcolor()
            self.pen_colour = colour[1]

        def eraser(self): # Eraser 
            self.pen_colour = "white"

        def optimiser(self):
            # Optimiser frame
            self.optimiser_frame = tk.LabelFrame(self.frame3_within, text=" Optimiser ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
            self.optimiser_frame.grid(row=0, column=0)
            # Optimiser setting
            optimiser_options = ctk.CTkOptionMenu(master=self.optimiser_frame, values=["adam", "adaGrad", "adadelta"], font=("Times New Roman", 20), button_color=("blue"), button_hover_color=("purple"), dropdown_font=("Times New Roman", 15), dropdown_hover_color=("Light blue"), width=265, height=56)
            optimiser_options.grid(row=0, column=0)
            optimiser_options.set("Optimiser")

        def loss(self):
            # Loss frame
            self.loss_frame = tk.LabelFrame(self.frame3_within, text=" Loss ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
            self.loss_frame.grid(row=1, column=0)
            # Loss setting
            loss_options = ctk.CTkOptionMenu(master=self.loss_frame, values=["categorical crossentropy", "sparse categorical crossentropy"], font=("Times New Roman", 20), button_color=("blue"), button_hover_color=("purple"), dropdown_font=("Times New Roman", 15), dropdown_hover_color=("Light blue"), width=265, height=56)
            loss_options.grid(row=0, column=0)
            loss_options.set("Loss")

        def saving(self):
            filename="image.png"
            self.image1.save(filename)

        def statistics(self):
            self.root = tk.Tk()
            self.root.mainloop()

        def log(self): # Log
            self.log_frame = tk.LabelFrame(self.frame3_1_within, text=" Log ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white", width=280, height=330)
            self.log_frame.grid(row=0, column=0)
            self.log_text = tk.Label(self.log_frame, text="text", width=37, height=20)
            self.log_text.config(text=self.message)
            self.log_text.pack()

        def result(self): # Result
            self.result_frame = tk.LabelFrame(self.frame3_1_within, text=" Result ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white", width=280, height=80)
            self.result_frame.grid(row=1, column=0)
            self.result_text = tk.Label(self.result_frame, text=" ", width=37, height=2)
            self.result_text.pack()
            
        def back_to_main(self):
            self.root.withdraw()  # 隐藏当前窗口
            # 创建新的 main 页面
            app = Classify_or_Recognise_Anything(self.root)  # 使用现有的 root 对象
            self.root.deiconify()  # 显示新窗口

    # 使用现有的 root 对象
    root = tk.Tk()
    activation = Drawing(root)
    activation.frames1()
    activation.frames2()
    activation.frames3()
    activation.frames3_1()
    activation.canvas_panel()
    activation.optimiser()
    activation.loss()
    activation.log()
    activation.result()
    root.mainloop()

if __name__ == "__main__":
    draw = drawing_window()