from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, LayoutLMv3TokenizerFast
from PIL import Image
import torch
from io import BytesIO
import numpy as np
import os
from huggingface_hub import login

import subprocess
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
import tempfile
import os

# Use an environment variable for the token


# Login at the module level

@csrf_exempt
def index(request):
    if request.method == 'POST':
        # Check if an image file is in the request
        if 'image' in request.FILES:
            # Process the image using LayoutLMv3
            image = request.FILES['image']
            
            # Open the image using PIL
            img = Image.open(BytesIO(image.read())).convert("RGB")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
                img.save(temp_image, format='JPEG')
                temp_image_path = temp_image.name
            
            detectron2_python = '/home/ubuntu/detectron2_env/bin/python'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            detectron2_processor = os.path.join(current_dir, 'detectron2_processor.py')

            try:
                # Call the Detectron2 script in the separate environment
                result = subprocess.run([
                    detectron2_python,
                    detectron2_processor,
                    temp_image_path
                ], capture_output=True, text=True, check=True)

                # Parse the JSON output
                detectron2_output = json.loads(result.stdout)

                return JsonResponse({
                    "result": detectron2_output,
                    "message": "Image processed successfully. The result contains words, their predicted classes, and bounding boxes."
                })
            except subprocess.CalledProcessError as e:
                return JsonResponse({'error': f'Detectron2 processing failed: {e.stderr}'}, status=500)
            except IOError:
                return JsonResponse({'error': 'Unable to open the image file.'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)
            finally:
                # Clean up the temporary file
                if 'temp_image_path' in locals():
                    os.unlink(temp_image_path)

        else:
            return JsonResponse({'error': 'No image file found in the request.'}, status=400)

    return HttpResponse("Please send a POST request with an image file.")