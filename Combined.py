import tkinter as tk
import customtkinter as ctk
import PIL
from tkinter import colorchooser as cc, filedialog as fd, messagebox as mb
from tkinter.ttk import Scale
from PIL import ImageTk, Image, ImageDraw
import Application

class Drawing():
    def __init__(self):
        self.windows = tk.Tk()
        self.windows.title("Number and Alphabet detection (Drawing site)")
        self.windows.geometry("1080x635+10+10")
        self.windows.configure(background="white")
        self.windows.resizable(False, False)
        self.pen_colour = "black"

    def frames1(self): # Top frame
        self.frame1 = tk.Frame(self.windows, width=1080, height=200)
        self.frame1.grid(row=0, column=0)

        self.frame1_within = tk.LabelFrame(self.frame1, width=1080, height=200, bg="lightyellow", bd=7, padx=6, pady=10)
        self.frame1_within.grid(row=0, column=0, sticky="nw")

        self.frame1_within.columnconfigure(0, minsize=250)
        self.frame1_within.columnconfigure(1, minsize=250)
        self.frame1_within.columnconfigure(2, minsize=250)
        self.frame1_within.columnconfigure(3, minsize=290)
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
        # Clear button
        self.clear_button = tk.Button(self.button_frame1, text="Clear", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.clearing, width=10, height=2)
        self.clear_button.grid(row=0, column=0) 
        # Save button
        self.save = tk.Button(self.button_frame1, text="Save", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.saving, width=10, height=2)
        self.save.grid(row=0, column=1)

        # Button frame2
        self.button_frame2 = tk.LabelFrame(self.frame1_within, bd=5, relief="ridge", bg="white")
        self.button_frame2.grid(row=1, column=1)
        # Statistics button
        self.statistic_button = tk.Button(self.button_frame2, text="Statistics", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.statistics, width=10, height=2)
        self.statistic_button.grid(row=0, column=0) 
        # Back button
        self.back_button = tk.Button(self.button_frame2, text="Back", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.backing, width=10, height=2)
        self.back_button.grid(row=0, column=1)

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
        self.detect_button = tk.Button(self.detect_button_frame, text="Detect", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.detection, width=25, height=2)
        self.detect_button.grid(row=0, column=0)

    def frames2(self): # Bottom frame
        self.frame2 = tk.Frame(self.windows, width=1080, height=435, bg="lightgrey")
        self.frame2.grid(row=1, column=0)

        self.frame2.columnconfigure(0, minsize=720)
        self.frame2.columnconfigure(1, minsize=360)
        
        self.frame2_within = tk.LabelFrame(self.frame2, bg="white", bd=7, padx=6, pady=10) # Bottom right frame
        self.frame2_within.grid(row=0, column=1)

    def canvas_panel(self): # Canvas
        self.canvas = tk.Canvas(self.frame2, bg="white", bd=5, relief="groove", height=420, width=420)
        self.image1=PIL.Image.new("RGB", (420, 420), (255, 255, 255))
        self.draw=ImageDraw.Draw(self.image1)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<B1-Motion>", self.painting)

    def painting(self, event): # Painting on canvas
        x1, y1 = (event.x-10), (event.y-10)
        x2, y2 = (event.x+10), (event.y+10)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_colour, outline=self.pen_colour, width=self.pen_scale.get())
        self.draw.line((x1, y1, x2, y2), fill=self.pen_colour, width=self.pen_scale.get())

    def select_colour(self, col1): # Pen colour from default options
        self.pen_colour = col1

    def select_custom_colour(self): # Pen custom colour
        colour = cc.askcolor()
        self.pen_colour = colour[1]

    def clearing(self): # For clearing canvas
        self.canvas.delete("all")
        self.image1=PIL.Image.new("RGB", (420, 420), (255, 255, 255))
        self.draw=ImageDraw.Draw(self.image1)
        
    def backing(self): # Back button
        self.windows.destroy()
        Application.Classify_or_Recognise_Anything()

    def saving(self): # Saving image
        filename="image.png"
        self.image1.save(filename)

    def statistics(self): #Statistic button
        stat_window = tk.Toplevel(self.windows)
        stat_window.title("Statistics")

    def detection(self):
        pass

    def optimiser(self):
        # Optimiser frame
        self.optimiser_frame = tk.LabelFrame(self.frame1_within, text=" Optimiser ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
        self.optimiser_frame.grid(row=0, column=3)
        # Optimiser setting
        optimiser_options = ctk.CTkOptionMenu(master=self.optimiser_frame, values=["adam", "adaGrad", "adadelta"], font=("Times New Roman", 20), button_color=("blue"), button_hover_color=("purple"), dropdown_font=("Times New Roman", 15), dropdown_hover_color=("Light blue"), width=250, height=50)
        optimiser_options.grid(row=0, column=0)
        optimiser_options.set("Optimiser")

    def log(self): # Log
        self.log_frame = tk.LabelFrame(self.frame2_within, text=" Log ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
        self.log_frame.grid(row=0, column=0)
        self.log_text = tk.Label(self.log_frame, text="test", font=("Times New Roman", 15), width=29, height=10, bg="lightgreen")
        self.log_text.pack()

    def result(self): # Result
        self.result_frame = tk.LabelFrame(self.frame2_within, text=" Result ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
        self.result_frame.grid(row=1, column=0)
        self.result_text = tk.Label(self.result_frame, text="test", font=("Times New Roman", 15), width=29, height=5, bg="lightblue")
        self.result_text.pack()

    def caller(self):
        self.frames1()
        self.frames2()
        self.canvas_panel()
        self.optimiser()
        self.log()
        self.result()
        self.windows.mainloop()

if __name__ == "__main__":
    activation = Drawing()
    activation.caller()