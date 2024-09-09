import cv2
import numpy as np
import json
import sys
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.structures import Boxes
import layoutparser as lp
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os
def preprocess_image(image):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	
	# Noise reduction
	denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
	
	# Contrast enhancement
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	enhanced = clahe.apply(denoised)
	
	# Thresholding
	_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# Convert back to RGB (3 channels) as layoutparser expects RGB input
	preprocessed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
	
	return preprocessed
def layout_to_dict(layout):
	return [
		{
			"id": block.id,
			"type": block.type,
			"score": block.score,
			"bbox": block.coordinates
		}
		for block in layout
	]

def process_image(image_path):
	# Load image
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = preprocess_image(image)
	model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
								 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
								 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
	layout = model.detect(image)

	list_blocks = lp.Layout([b for b in layout if b.type=='List'])
	table_blocks = lp.Layout([b for b in layout if b.type=='Table'])
	figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])

	text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
	text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks) and
							 not any(b.is_in(b_list) for b_list in list_blocks) and
							 not any(b.is_in(b_table) for b_table in table_blocks)])

	h, w = image.shape[:2]

	text_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(text_blocks)])
	list_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(list_blocks)])

	result_image = lp.draw_box(image, layout, box_width=3, show_element_id=True)










# Convert result_image to numpy array if it's not already
	if not isinstance(result_image, np.ndarray):
		result_image = np.array(result_image)

	# Save the result image to a temporary file
	with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
		cv2.imwrite(temp_image.name, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
		temp_image_path = temp_image.name
















	result = {
		"layout": layout_to_dict(layout),
		"text_blocks": layout_to_dict(text_blocks),
		"image_path": temp_image_path
	}







	return json.dumps(result)

if __name__ == "__main__":
	image_path = sys.argv[1]
	print(process_image(image_path))