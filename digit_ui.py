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
def digit_ui():
    class Drawing():
        def __init__(self, root):
            self.root = root
            self.root.title("Number and Alphabet detection (Drawing site)")
            self.root.geometry("1080x635+10+10")
            self.root.configure(background="white")
            self.root.resizable(True, True)
            self.pen_colour = "black"

            image_path = "image/picture.png"  
            # self.load_and_display_image(image_path)

            #Keven edits
            self.old_x = None
            self.old_y = None
            self.optimizer_choice = tk.StringVar()
            self.optimizer_choice.set("Adam")

            #Background
        # def load_and_display_image(self, path):
        #     image = Image.open(path)
        #     photo = ImageTk.PhotoImage(image)
        #     image_label = tk.Label(self.root, image=photo, bg="white")
        #     image_label.photo = photo  # This line is crucial to prevent garbage collection
        #     image_label.grid(row=1, column=0, sticky="nsew")  # Adjust the row and column as needed


        def frames1(self): # Top frame
            self.frame1 = tk.Frame(self.root, width=1080, height=200)
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
            self.scale_frame = tk.LabelFrame(self.frame1_within, text=" Size of pen ", font=("Times New Roman", 15), bd=5, relief="ridge", bg="white")
            self.scale_frame.grid(row=0, column=2)
            # Pen scale
            self.pen_scale = tk.Scale(self.scale_frame, orient="horizontal", from_=6, to=25, length=250)
            self.pen_scale.set(6)
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
            self.canvas.bind("<ButtonRelease-1>", self.reset_coor)

        # def painting(self, event): # Painting on canvas
        #     x1, y1 = (event.x-10), (event.y-10)
        #     x2, y2 = (event.x+10), (event.y+10)
        #     self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_colour, outline=self.pen_colour, width=self.pen_scale.get())
        #     self.draw.line((x1, y1, x2, y2), fill=self.pen_colour, width=self.pen_scale.get())

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
            stats_window.title(f"Statistics of {self.optimizer_choice}")

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
            optimiser_options = ctk.CTkOptionMenu(master=self.optimiser_frame, values=["RMSDrop", "Adam", "Adamax", "SGD", "Adadelta"], font=("Times New Roman", 20), button_color=("blue"), button_hover_color=("purple"), dropdown_font=("Times New Roman", 15), dropdown_hover_color=("Light blue"), width=250, height=50, variable=self.optimizer_choice)
            optimiser_options.grid(row=0, column=0)
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
            self.result_frame.grid(row=1, column=0)
            
            # Create a canvas for the vertical bar plot
            result_canvas = tk.Canvas(self.result_frame, bg="white", width=300, height=300)
            result_canvas.pack()

            # Plot vertical bars for labels and accuracies
            bar_width = 40
            bar_spacing = 20
            x_position = bar_spacing
            
            for i in range(len(self.label)):
                label = self.label[i]
                accuracy = self.accuracy[i]

                # Scale accuracy to fit in the canvas
                scaled_accuracy = int(accuracy * 100)

                # Draw a vertical bar
                result_canvas.create_rectangle(
                    x_position, 200 - scaled_accuracy, 
                    x_position + bar_width, 200, 
                    fill="red", outline="black"
                )

                # Add label text below the bar
                result_canvas.create_text(
                    x_position + bar_width // 2, 200 + bar_spacing, 
                    text=label, anchor="n", font=("Times New Roman", 12)
                )

                # Move x_position for the next bar
                x_position += bar_width + bar_spacing


        # def back_to_main(self):
        #     self.root.withdraw()  # 隐藏当前窗口
        #     # 创建新的 main 页面
        #     app = Classify_or_Recognise_Anything(self.root)  # 使用现有的 root 对象
        #     self.root.deiconify()  # 显示新窗口

    root = tk.Tk()
    activation = Drawing(root)
    activation.frames1()
    activation.frames2()
    activation.canvas_panel()
    activation.optimiser()
    # activation.log()
    # activation.result()
    root.mainloop()

digit_ui()