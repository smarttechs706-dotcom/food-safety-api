from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # Add this import
import torch
from PIL import Image
import io
import numpy as np
import pickle
import os
import torchvision.transforms as transforms

app = FastAPI(title="Food Safety Prediction API")

# Add CORS middleware - ADD THESE LINES
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

device = torch.device("cpu")

# ... rest of your code stays the same
