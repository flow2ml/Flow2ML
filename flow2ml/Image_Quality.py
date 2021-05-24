import imquality.brisque as brisque
import PIL.Image
from docx import Document
import os

class Image_Quality:
	'''
		Class to calculate image quality score for all images that are stored in ProcessedData folder
	'''

	def __init__(self):
		'''
		Initializes various attributes regarding to the object.
		Args : 
			None.
		'''
		self.image_scores = {}

	def generate_img_scores(self, path):
		''' 
		Generates the image quality scores for all images in the given folder using BRISQUE and stores it in a dictionary.
		Args : 
			path : (string) path to the processedData folder.
		'''
		for i in os.listdir(path):
			img_path = os.path.join(path, i)
			img = PIL.Image.open(img_path)
			img_score = round(brisque.score(img), 2)
			self.image_scores[i] = img_score

	def create_scores_doc(self, path):
		''' 
		Calculates image scores for the processedData folder and creates a document containing this information.
		Args : 
			path : (string) path to the processedData folder.
		'''
		self.generate_img_scores(path)
		try:
			doc = Document()
			doc.add_heading('Image Quality Report')
			# loop over all images in dictionary and add their names and scores to the document.
			for image in self.image_scores:
				doc.add_paragraph(f"Image {image}: Quality Score {self.image_scores[image]}")
			doc.save(os.path.join(path, "image_quality_report.docx"))
		except Exception as e:
			print("Unable to create image quality document due to {e}")