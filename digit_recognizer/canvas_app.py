import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#  Recongize the model architecture used during training
class DigitRecognizerNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerNN, self).__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        return self.output(x)

# Instantiate and load trained parameters
model = DigitRecognizerNN()
model.load_state_dict(torch.load('digit_model.pth', weights_only=True))
model.eval()

# Main Interface Layout with Side-by-Side Visualisation
class CanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        
        # 🟢 Handle clean program closure when clicking the 'X' window button
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.old_x = None
        self.old_y = None
        self.stroke_width = 16  
        
        # Main Container Frame for Side-by-Side Layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Left Panel: User Input Canvas Drawing Area
        self.left_frame = tk.LabelFrame(self.main_frame, text=" Draw Digit Here ")
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.canvas = tk.Canvas(self.left_frame, width=280, height=280, bg='black', cursor="pencil")
        self.canvas.pack(padx=5, pady=5)
        
        # Right Panel: Processed Model Input Display (28x28 Matrix View)
        self.right_frame = tk.LabelFrame(self.main_frame, text=" What the Model Sees (28x28) ")
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Initialise Matplotlib Figure for the 28x28 array rendering
        self.fig, self.ax = plt.subplots(figsize=(3, 3), dpi=100)
        self.fig.patch.set_facecolor('#F0F0F0')
        self.ax.axis('off')
        
        self.matrix_display = self.ax.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.plot_canvas.get_tk_widget().pack(padx=5, pady=5)
        
        # Persistent In-Memory Grayscale image copy tracking structure
        self.pil_img = Image.new("L", (280, 280), "black")
        self.draw_context = ImageDraw.Draw(self.pil_img) 
        
        # Action Control Panel Footer
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.btn_predict = tk.Button(self.bottom_frame, text="Predict Digit", command=self.run_prediction, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(self.bottom_frame, text="Clear Canvas", command=self.clear_canvas, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        self.label_result = tk.Label(self.bottom_frame, text="Prediction: Ready", font=("Helvetica", 14, "bold"))
        self.label_result.pack(side=tk.RIGHT, padx=10)
        
        # Drawing Mouse Event Hooks Configuration
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def on_mouse_press(self, event):
        self.old_x = event.x
        self.old_y = event.y
        r = self.stroke_width // 2
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="white", outline="white")
        self.draw_context.ellipse([event.x-r, event.y-r, event.x+r, event.y+r], fill="white")

    def on_mouse_drag(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=self.stroke_width, fill="white", 
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)
            
            self.draw_context.line([self.old_x, self.old_y, event.x, event.y], 
                                   fill="white", width=self.stroke_width)
            
        self.old_x = event.x
        self.old_y = event.y

    def on_mouse_release(self, event):
        self.old_x = None
        self.old_y = None

    def run_prediction(self):
        img = self.pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        
        self.matrix_display.set_data(img_array)
        self.plot_canvas.draw()
        
        flat_array = img_array.flatten()
        input_tensor = torch.tensor(flat_array, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item() * 100
            
        self.label_result.config(text=f"Prediction: {prediction} ({confidence:.1f}%)")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_context.rectangle([0, 0, 280, 280], fill="black")
        
        self.matrix_display.set_data(np.zeros((28, 28)))
        self.plot_canvas.draw()
        
        self.label_result.config(text="Prediction: Ready")

    # 🟢 Safe shutdown handler method
    def on_closing(self):
        """Stops the mainloop and completely closes the Python program instance."""
        plt.close(self.fig)  # Close active Matplotlib figure resources
        self.root.quit()     # Stops the root.mainloop()
        self.root.destroy()  # Destroys all widgets and closes window

if __name__ == "__main__":
    root = tk.Tk()
    app = CanvasApp(root)
    root.mainloop()
