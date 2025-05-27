import pandas as pd
import numpy as np
from scipy.signal import medfilt

def post_process_predictions(preds, frame_hop=0.01, min_duration=0.2, audiofile_name="File1"):
    # Apply median filtering to smooth out prediction noise
    smoothed = medfilt(preds, kernel_size=5)

    segments = []
    start = 0
    current_class = smoothed[0]

 
    for i in range(1, len(smoothed)):
        if smoothed[i] != current_class:
            end = i
            duration = (end - start) * frame_hop
            
            if duration >= min_duration:
                segments.append((start * frame_hop, end * frame_hop, current_class))
            start = i
            current_class = smoothed[i]

    # Handle the final segment
    if start < len(smoothed):
        end = len(smoothed)
        duration = (end - start) * frame_hop
        if duration >= min_duration:
            segments.append((start * frame_hop, end * frame_hop, current_class))

    # Gap correction
    corrected = []
    for i, seg in enumerate(segments):
        if i == 0:
            corrected.append(list(seg)) 
        else:
            prev = corrected[-1]
            if round(seg[0], 3) > round(prev[1], 3):  
                seg = list(seg)  
                seg[0] = prev[1]  
