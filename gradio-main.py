
import gradio as gr

from helper import *

import os
import matplotlib.pyplot as plt
import seaborn as sns


iface = gr.Interface(fn=pred_gradio, inputs=gr.inputs.Image(type="filepath"), outputs=gr.outputs.Label())

iface.launch(share=True)