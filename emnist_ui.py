import os
import tkinter as tk
import customtkinter as ctk
import PIL
from tkinter import colorchooser as cc, filedialog as fd, messagebox as mb
from tkinter.ttk import Scale
from PIL import ImageTk, Image, ImageDraw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from emnist_recognition import load_and_predict_image, show_stats

#put back to root
class Drawing():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Number and Alphabet Recognition")
        self.root.geometry("1080x810+10+10")
        self.root.configure(background="white")
        self.root.resizable(False, False)
        self.pen_colour = "black"

        image_path = "image/background.jpeg"  
        self.set_bg_image(image_path)
        self.old_x = None
        self.old_y = None
        self.label = []
        self.accuracy = []
        self.optimizer_choice = tk.StringVar()
        self.optimizer_choice.set("Adam")

    def set_bg_image(self,image_path):
        bg_image_path = image_path  # Replace with the path to your background image
        bg_image = Image.open(bg_image_path)  # Open the image file
        bg_photo = ImageTk.PhotoImage(bg_image)  # Convert the image for tkinter
        
        bg_label = tk.Label(self.root, image=bg_photo)  # Create a label with the image
        bg_label.image = bg_photo  # Attach the image to the label to prevent garbage collection
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Use place to set the label as the background


    def frames1(self): # Top frame
        self.frame1 = tk.Frame(self.root, width=1080, height=200)
        self.frame1.grid(row=0, column=0)

        self.frame1_within = tk.LabelFrame(self.frame1, width=1080, height=200, bg="lightyellow", bd=7, padx=6, pady=10)
        self.frame1_within.grid(row=0, column=0, sticky="nw")

        self.frame1_within.columnconfigure(0, weight=1)
        self.frame1_within.columnconfigure(1, weight=1)
        self.frame1_within.columnconfigure(2, weight=1)
        self.frame1_within.columnconfigure(3, weight=1)
        self.frame1_within.rowconfigure(0, weight=1)

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
        self.scale_frame = tk.LabelFrame(self.frame1_within, text=" Size of pen ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
        self.scale_frame.grid(row=0, column=2)
        # Pen scale
        self.pen_scale = tk.Scale(self.scale_frame, orient="horizontal", from_=10, to=25, length=250)
        self.pen_scale.set(10)
        self.pen_scale.grid(row=0)

        # Detect button frame
        self.detect_button_frame = tk.LabelFrame(self.frame1_within, bd=5, relief="ridge", bg="white")
        self.detect_button_frame.grid(row=1, column=2)
        # Detect button
        self.detect_button = tk.Button(self.detect_button_frame, text="Detect", font=("Times New Roman", 12), bd=5, relief="ridge", bg="white", command=self.detection, width=25, height=2)
        self.detect_button.grid(row=0, column=0)

    def frames2(self): # Bottom frame
        self.frame2 = tk.Frame(self.root, width=1080, height=435, bg="lightgrey")
        self.frame2.grid(row=1, column=0)

        self.frame2.columnconfigure(0, weight=3)
        self.frame2.columnconfigure(1, weight=1)

        self.frame2_within = tk.LabelFrame(self.frame2, bg="white", bd=7, padx=6, pady=10) # Bottom right frame
        self.frame2_within.grid(row=0, column=1, sticky="nsew")
        self.frame2_within.columnconfigure(0, weight=1)
        self.frame2_within.rowconfigure(0, weight=1)
        self.frame2_within.rowconfigure(1, weight=1)

    def canvas_panel(self): # Canvas
        self.canvas = tk.Canvas(self.frame2, bg="white", bd=5, relief="groove", height=420, width=420)
        self.image1=PIL.Image.new("RGB", (420, 420), (255, 255, 255))
        self.draw=ImageDraw.Draw(self.image1)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<B1-Motion>", self.painting)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coor)

    def painting(self,event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x,self.old_y,event.x,event.y,width=self.pen_scale.get(),fill=self.pen_colour,capstyle='round',smooth=True)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=self.pen_colour, width=self.pen_scale.get())

        self.old_x = event.x
        self.old_y = event.y

    def reset_coor(self, event):
        self.old_x = None
        self.old_y = None

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
        self.root.destroy()

    def saving(self): # Saving image           
        filename="image/image.png"
        self.image1.save(filename)

    def statistics(self): #Statistic button
        stats_window = tk.Toplevel(self.root)
        stats_window.title(f"Statistics of Model with {self.optimizer_choice} Optimizer")

        # Create a frame to contain the Matplotlib plots
        matplotlib_frame = tk.ttk.Frame(stats_window)
        matplotlib_frame.pack(side=tk.TOP, padx=10, pady=10)

        fig_loss , fig_accuracy = show_stats(self.optimizer_choice)
        # Embed the loss plot in Tkinter
        canvas_loss = FigureCanvasTkAgg(fig_loss, master=matplotlib_frame)
        canvas_loss.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_loss.draw()

        # Embed the accuracy plot in Tkinter
        canvas_accuracy = FigureCanvasTkAgg(fig_accuracy, master=matplotlib_frame)
        canvas_accuracy.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_accuracy.draw()

    def detection(self):
        #get selected optimizer choice
        selected_optimizer = self.optimizer_choice.get()
        #Return label and accuracy
        self.label, self.accuracy = load_and_predict_image('image/image.png',selected_optimizer)
        # self.message = f"Predicted Label: {self.label}, Accuracy: {self.accuracy:.2f}"
        # print(self.label, self.accuracy)      

        # self.log_text.config(text=self.message)
        # self.result_text.config(text=self.message)
        self.chart()

    def optimiser(self):
        # Optimiser frame
        self.optimiser_frame = tk.LabelFrame(self.frame1_within, text=" Optimiser ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
        self.optimiser_frame.grid(row=0, column=5)
        # Optimiser setting
        optimiser_options = ctk.CTkOptionMenu(master=self.optimiser_frame, values=["RMSDrop", "Adam", "Adamax", "SGD", "Adadelta"], font=("Times New Roman", 20), button_color=("blue"), button_hover_color=("purple"), dropdown_font=("Times New Roman", 15), dropdown_hover_color=("Light blue"), width=200, height=40, variable=self.optimizer_choice)
        optimiser_options.grid(row=0, column=0, padx=10, pady=10)
        # optimiser_options.set("Optimiser")

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

    def chart(self): # Result
        self.result_frame = tk.LabelFrame(self.frame2_within, text=" Result ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
        self.result_frame.grid(row=1, column=0, sticky="nsew")
        
        # Create a canvas for the vertical bar plot
        result_canvas = tk.Canvas(self.result_frame, bg="white", width=300, height=300)
        result_canvas.pack(fill="both",expand=True)

        bar_height = 40  # Height of each bar
        bar_spacing = 20  # Spacing between bars

        if all(self.label) and all(self.accuracy):
            for i in range(len(self.label)):
                label = self.label[i]
                accuracy = self.accuracy[i]

                # Scale accuracy to fit in the canvas
                scaled_accuracy = int(accuracy * 100)

                # Draw a horizontal bar with separation
                result_canvas.create_rectangle(
                    60, i*(bar_height+bar_spacing), scaled_accuracy+60, (i+1)*(bar_height+bar_spacing),
                    fill="beige", outline="black"
                )

                # Add percentage text within the bar
                result_canvas.create_text(
                    70 + scaled_accuracy, i*(bar_height+bar_spacing) + bar_height/2,
                    text=f"{accuracy*100:.2f}%", anchor="w", font=("Times New Roman", 12)
                )

                # Add label text outside the bar
                result_canvas.create_text(
                    10, i*(bar_height+bar_spacing) + bar_height/2,
                    text=f"{label}", anchor="e", font=("Times New Roman", 12)
                )

    # def back_to_main(self):
    #     self.root.withdraw()  # 隐藏当前窗口
    #     # 创建新的 main 页面
    #     app = Classify_or_Recognise_Anything(self.root)  # 使用现有的 root 对象
    #     self.root.deiconify()  # 显示新窗口

    def build_window(self):
        # root = tk.Tk()
        # activation = Drawing()
        self.frames1()
        self.frames2()
        self.canvas_panel()
        self.optimiser()
        # activation.log()
        # activation.result()
        self.chart()
        self.root.mainloop()

if __name__ == "__main__":
    drawing = Drawing()
    drawing.build_window()