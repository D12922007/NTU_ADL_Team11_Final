import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from diffusers import StableDiffusionModelEditingPipeline
from googletrans import Translator
translator = Translator()
global idx_img

#####################################################
################### OpenAI Model ####################
#####################################################


from openai import OpenAI
import pandas as pd
import yaml
import uuid
from tqdm import tqdm
from context import *
import random
import chinese_converter
import json

client = OpenAI(api_key="")
MODEL = "gpt-3.5-turbo-16k"
# FIXME: Add diversity
INSTRUCTION = {"question_answering": question_answering_context,
               "preach": preach_context,
               "pray": pray_context,
               "consult": consult_context}
INSTRUCTION_NO_INPUT = {"pray": "請根據聖經為使用者的處境禱告。", "consult": "請根據聖經幫助使用者面對他的處境。"}

def inference(prompt):
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=MODEL,
    )
    return chat_completion.choices[0].message.content
#print("INSTRUCTION_NO_INPUT",INSTRUCTION_NO_INPUT)
#output = inference(str(INSTRUCTION_NO_INPUT))
#print("output", output)



#####################################################
################### Display All #####################
#####################################################

from tkinter import *
import tkinter as tk
from PIL import Image,ImageTk


root = Tk()
root.geometry("975x700")

def close_win():
   root.destroy()
    
str_var = tk.StringVar()
str_var.set('')

    

# Text Bar
global idx_img
idx_img = 0
show_text_0 = Text(root, height=2, width=100)
show_text_0.place(x = 20, y = 155+50)

def show():
    global newWindow
    global idx_img
    idx_img=idx_img+1
    model_ckpt = "./pretrained_model"
    pipe = StableDiffusionModelEditingPipeline.from_pretrained(model_ckpt)
    pipe = pipe.to("cuda")
    show_text = show_text_0.get(1.0, 'end-1c')
    #str_var.set(show_text.get(1.0, 'end-1c'))
    #str_var.set(show_text)
    
    output = inference(str(show_text))
    print("output ",len(output))
    for i  in range (len(output)//72):
        output = output[:i*72+72]+'\n' + output[i*72+72:]
    str_var.set(output)
    try:
        #prompt = translator.translate(show_text.get(1.0, 'end-1c'),dest='en').text
        prompt = translator.translate(show_text,dest='en').text
        print("prompt", prompt)
    except:
        prompt = "Taipei City"
        print("No Prompt")
    image = pipe(prompt).images[0]
    image.save("fusing_result_{}.png".format(idx_img))
    print("Saved Image")
    
    # Toplevel object which will 
    # be treated as a new window
    newWindow = Toplevel(root)
    # sets the title of the
    # Toplevel widget
    newWindow.title("Generation Result")
    # sets the geometry of toplevel
    newWindow.geometry("512x512")
    # A Label widget to show in toplevel
    #Label(newWindow, text ="This is a new window").pack()
    fusing_image = Image.open("fusing_result_{}.png".format(idx_img))
    fusing_image=fusing_image.resize((512, 512))
    fusing_image = ImageTk.PhotoImage(fusing_image)
    Label(newWindow,image = fusing_image).place(x = 0, y = 0) #.pack(side = RIGHT, pady=5)

def clear():
    global newWindow
    str_var.set('')
    show_text.delete(1.0,'end')
    newWindow.destroy()
    # 執行 clear 函式時，清空內0容

#Load the image

#human_image = Image.open(display_image)
#fusing_image = Image.open("fusing_result_{}.png".format(idx_img))
#human_image=human_image.resize((410, 443))
#fusing_image=fusing_image.resize((256, 256))
#human_image = ImageTk.PhotoImage(human_image)
#fusing_image = ImageTk.PhotoImage(fusing_image)

image = Image.open('click.png')
close_img = Image.open('close.jpg')
clear_img = Image.open('clear.png')
NTU_img = Image.open('NTU.png')
INM_img = Image.open('CSIE.png')
NEWALAB_img = Image.open('NEWS_LAB.png')
#Resize the Image
image = image.resize((25,25), Image.Resampling.LANCZOS) # Image.ANTIALIAS # Image.Resampling.LANCZOS
close_image = close_img.resize((25,25), Image.Resampling.LANCZOS) # Image.ANTIALIAS # Image.Resampling.LANCZOS
clear_image = clear_img.resize((25,25), Image.Resampling.LANCZOS) # Image.ANTIALIAS # Image.Resampling.LANCZOS
NTU_img = NTU_img.resize((479,100), Image.Resampling.LANCZOS) # Image.ANTIALIAS # Image.Resampling.LANCZOS
INM_img = INM_img.resize((497,100), Image.Resampling.LANCZOS) # Image.ANTIALIAS # Image.Resampling.LANCZOS
NEWALAB_img = NEWALAB_img.resize((232,50), Image.Resampling.LANCZOS) # Image.ANTIALIAS # Image.Resampling.LANCZOS
#Convert the image to PhotoImage
img= ImageTk.PhotoImage(image)
close_img= ImageTk.PhotoImage(close_image)
clear_img= ImageTk.PhotoImage(clear_image)
NTU_img= ImageTk.PhotoImage(NTU_img)
INM_img= ImageTk.PhotoImage(INM_img)
NEWALAB_img= ImageTk.PhotoImage(NEWALAB_img)

#Button(root,text='show', command=show).place(x = 410, y = 155)    # 放入顯示按鈕
#Button(root,text='clear', command=clear).place(x = 500, y = 155)  # 放入清空按鈕
Label(root, textvariable=str_var).place(x = 28, y = 280)

# label widget
Label(root, image=NTU_img, compound= LEFT).place(anchor = NW)
Label(root, image=INM_img, compound= LEFT).place(relx = 1, x =-2, anchor = NE)
Label(root, text = "※ End-to-end Text to Image Generation Model Pipeline:", font=('Aerial 15 bold')).place(x = 28, y = 120) #.pack(side = TOP,pady=120)
Label(root, text = "Questions:", font=('Aerial 15 bold')).place(x = 28, y = 175) #.pack(side = TOP,pady=120)
Label(root, text = "Answers:", font=('Aerial 15 bold')).place(x = 28, y = 250) #.pack(side = TOP,pady=120)
#Label(root, text = "Figure. Text to Image Generation Result 1",font=('Aerial 13 bold')).place(x = 30, y = 560)#.pack(side = TOP)#.place(anchor = NW)
#Label(root, text = "Figure. Text to Image Generation Result 2",font=('Aerial 13 bold')).place(x = 450, y = 560)#.pack(padx=20)#.place(relx = 1, x =-2, y = 2, anchor = NE)

Button(root, text="Click to Show Result",font= ('Helvetica 13 bold'),image=img, compound= LEFT, command=show).place(x = 120, y = 560)#.pack(side = BOTTOM)
Button(root, text="Click to Clear",font= ('Helvetica 13 bold'),image=clear_img, compound= LEFT, command=clear).place(x = 630, y = 560)#.pack(side = BOTTOM)
#Create a label with the image
Button(root, text="Click to Close Window",font= ('Helvetica 13 bold'),image=close_img, compound= LEFT, command=close_win).place(x = 360, y = 620)#.pack(side = BOTTOM)
#Create a Label
#Label(root, text="Click the button to close the window",font=('Aerial 15 bold')).pack(side = BOTTOM) # pady=20
Label(root, image=NEWALAB_img, compound= CENTER).place(x = 372, y = 560)#.pack(side = BOTTOM,pady=5)

#Label(root,image = human_image).place(x = 28, y = 155) #.pack(side = LEFT, pady=5)
root.mainloop()

#####################################################
################### Display All #####################
#####################################################


