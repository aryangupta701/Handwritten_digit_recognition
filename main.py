from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps
import numpy as np
import torch
from torch import nn

class DigitRecognition(nn.Module):
    
  def __init__(self, input_size, hidden_layers, num_classes):
    super(DigitRecognition, self).__init__()
    #first layer 
    self.input = nn.Linear(in_features = input_size, out_features = hidden_layers )
    self.relu_1 = nn.ReLU()

    self.hidden_1 = nn.Linear(in_features=hidden_layers, out_features = hidden_layers)
    self.relu_2 = nn.ReLU()

    self.hidden_2 = nn.Linear(in_features=hidden_layers, out_features = hidden_layers)
    self.relu_3 = nn.ReLU()

    self.hidden_3 = nn.Linear(in_features=hidden_layers, out_features = hidden_layers)
    self.relu_4 = nn.ReLU()

    self.hidden_4 = nn.Linear(in_features=hidden_layers, out_features = hidden_layers)
    self.relu_5 = nn.ReLU()

    self.output = nn.Linear(in_features=hidden_layers, out_features=num_classes)

 
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    model = self.input(x)
    model = self.relu_1(model)

    model = self.hidden_1(model)
    model = self.relu_2(model)

    model = self.hidden_2(model)
    model = self.relu_3(model)

    model = self.hidden_3(model)
    model = self.relu_4(model)

    model = self.hidden_4(model)
    model = self.relu_5(model)

    model = self.output(model)
    return model

input_size = 784
num_classes = 10
hidden_layers = 100
PATH = 'model.pth'
model = DigitRecognition(input_size, hidden_layers, num_classes)
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))


def predict_digit(img):
    #resize image to 28×28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = torch.from_numpy(img)
    # img.reshape(1,784)
    img = img.reshape(1,784)
    img = img.type(torch.float32)
    print(img.dtype)
    #predicting the class)
    model.eval()
    with torch.inference_mode():
        res = model(img)
    print(res)
    return torch.argmax(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)

        digit = predict_digit(im)
        self.label.configure(text= str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
app.mainloop()