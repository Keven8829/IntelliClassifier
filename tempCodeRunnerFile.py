# Resize the image when the window is resized
            def resize_image(event):
                width = event.width
                height = event.height
                ratio = min(width / bg_image.width, height / bg_image.height)
                new_width = int(bg_image.width * ratio)
                new_height = int(bg_image.height * ratio)
                bg_image = bg_image.resize((new_width, new_height), Image.ANTIALIAS)
                bg_photo = ImageTk.PhotoImage(bg_image)
                bg_label.config(image=bg_photo)
                bg_label.image = bg_photo

            self.root.bind('<Configure>', resize_image)