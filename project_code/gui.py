import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from pathlib import Path

from checkpoint import Checkpoint
from DCGAN import Generator
from bert_encoder import BERTWrapper
from image_transform import get_inv_image_transform


class BirdImageGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch Bird Image Generator")
        self.root.resizable(False, False)

        self.root.configure(bg="white")

        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack()

        path_frame = tk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=5)

        tk.Label(path_frame, text="Model Path:", width=12, anchor="w").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value="./checkpoints/train1/saved_models/ep740.tar")
        self.model_path_entry = tk.Entry(path_frame, textvariable=self.model_path_var, width=40)
        self.model_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = tk.Button(path_frame, text="Browse", command=self.browse_model)
        browse_btn.pack(side=tk.LEFT, padx=5)

        desc_frame = tk.Frame(main_frame)
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        tk.Label(desc_frame, text="Description:", width=12, anchor="nw").pack(side=tk.LEFT, anchor="n")

        self.desc_text = tk.Text(desc_frame, width=40, height=5)
        self.desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.desc_text.insert("1.0", "the medium sized bird...")

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10, anchor="e")

        clear_btn = tk.Button(button_frame, text="Clear", width=10, command=self.clear_fields)
        clear_btn.pack(side=tk.LEFT, padx=5)

        generate_btn = tk.Button(button_frame, text="Generate Image", width=15, command=self.generate_image)
        generate_btn.pack(side=tk.LEFT, padx=5)

        self.image_windows = []

    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.tar"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)

    def clear_fields(self):
        self.desc_text.delete("1.0", tk.END)

    def generate_image(self):
        try:
            model_path = self.model_path_var.get().strip()
            if not model_path:
                messagebox.showerror("Error", "Please select a model path")
                return

            prompt = self.desc_text.get("1.0", tk.END).strip()
            if not prompt:
                messagebox.showerror("Error", "Please enter a description")
                return

            model_path = Path(model_path)
            checkpoint_path = model_path.parent.parent
            checkpoint = Checkpoint(checkpoint_path, 1)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = Generator()
            model.eval()
            model.to(device)
            epoch = int(model_path.stem[2:])
            model_config = checkpoint.load_generator(model, epoch, map_location=device)

            embedding = BERTWrapper(device=device)(prompt).unsqueeze(0)
            noise = torch.randn(1, model_config["noise_size"]).to(device)

            pil_image = get_inv_image_transform()(model(embedding, noise)[0])

            self.show_image_window(pil_image)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate image: {str(e)}")

    def show_image_window(self, pil_image):
        img_window = tk.Toplevel(self.root)
        img_window.title("Generated Bird Image")

        display_img = pil_image.resize((256, 256), Image.LANCZOS)

        photo_img = ImageTk.PhotoImage(display_img)

        img_label = tk.Label(img_window, image=photo_img)
        img_label.image = photo_img
        img_label.pack(padx=10, pady=10)

        save_btn = tk.Button(
            img_window,
            text="Save Image",
            command=lambda: self.save_image(pil_image)
        )
        save_btn.pack(pady=10)

        self.image_windows.append(img_window)
        img_window.protocol("WM_DELETE_WINDOW", lambda: self.close_image_window(img_window))

    def close_image_window(self, window):
        if window in self.image_windows:
            self.image_windows.remove(window)
        window.destroy()

    def save_image(self, pil_image):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Image As"
        )

        if file_path:
            try:
                pil_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BirdImageGenerator(root)
    root.mainloop()
