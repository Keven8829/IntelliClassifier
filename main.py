import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tkinter import ttk
# from image_ui import Zty
from emnist_ui import Drawing


class IntelliClassifier:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IntelliClassifier")
        self.root.geometry("1080x810")
        self.root.configure(bg="white")

        image_path = "image/background.jpeg"  
        self.load_and_display_image(image_path)

        self.zty_instance = None  

        self.frame = tk.Frame(self.root, bg="beige", bd=10, relief=tk.GROOVE, borderwidth=2, border=2, padx=20, pady=20)

        # Create a Label widget for the title
        self.title_label = tk.Label(self.root, text="IntelliClassifier", bg="black", font=("Helvetica", 64, "bold"))
        # Center the title label using the 'place' geometry manager
        self.title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        self.button1 = tk.Button(self.frame, text="Image Recognition", command=self.image_recognition, width=20, height=2, bg="white", bd=5, relief=tk.RAISED, borderwidth=2)
        self.button2 = tk.Button(self.frame, text="Text Recognition", command=self.text_recognition, width=20, height=2, bg="white", bd=5, relief=tk.RAISED, borderwidth=2)
        self.button3 = tk.Button(self.frame, text="About Our Application", command=self.show_introduction, width=20, height=2, bg="white", bd=5, relief=tk.RAISED, borderwidth=2)
        self.button4 = tk.Button(self.frame, text="Exit", command=self.close_program, width=20, height=2, bg="white", bd=5, relief=tk.RAISED, borderwidth=2)

        self.button1.grid(row=0, column=0, padx=20, pady=20)
        self.button2.grid(row=1, column=0, padx=20, pady=20)
        self.button3.grid(row=2, column=0, padx=20, pady=20)
        self.button4.grid(row=3, column=0, padx=20, pady=20)

        self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.root.mainloop()

    def load_and_display_image(self, path):
        image = Image.open(path)
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(self.root, image=photo, bg="white")
        image_label.photo = photo  # This line is crucial to prevent garbage collection
        image_label.pack(side="top")

    def image_recognition(self):
        self.root.destroy()
        # zty_instance = ui.Zty()
        # zty_instance.built_window(self)

    def text_recognition(self):
        # # self.root.destroy()
        # self.root.withdraw()
        # popup_window = digit_ui(self.root)
        # popup_window.wait_window()
        # # drawing_window()
        # self.root.deiconify()

        # Create a Toplevel window for digit_ui
        popup_window = tk.Toplevel(self.root)
        popup_window.title("Text Recognition")
        digit_ui_instance = Drawing(popup_window)
        digit_ui_instance.wait_window()
        self.root.deiconify()

    def show_introduction(self):
        # Add the logic to display introduction here
        pass

    def close_program(self, event=None):
        self.root.destroy()


if __name__ == "__main__":
    app = IntelliClassifier()


