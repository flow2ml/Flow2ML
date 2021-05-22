import unittest
import cv2
import os
import shutil
import numpy as np
import coverage
from flow2ml import Flow

# create mock data for 5 classes with 5 images each
def create_data():
	if os.path.exists('./dataset_dir/'):
		shutil.rmtree('./dataset_dir/')
	for i in range(5):
		os.makedirs(f'./dataset_dir/data_dir/class{i}')
		for j in range(5):
			img = np.random.randint(0, 256, size = [150, 150, 3], dtype = np.uint8)
			cv2.imwrite(f'./dataset_dir/data_dir/class{i}/class{i}_{j}.png', img)

class testFlow2ML(unittest.TestCase):

	create_data()
	flow = Flow('dataset_dir', 'data_dir')
	filters = ["median", "gaussian", "laplacian", "sobelx", "sobely"]
	flow.applyFilters(filters)

	img_dimensions = (150, 150, 1)
	test_val_split = 0.4
	(train_x, train_y, val_x, val_y) = flow.getDataset(img_dimensions, test_val_split)

	def test_data(self):
		self.assertTrue(os.path.exists('./dataset_dir/'), "Data not generated")

	def test_dataset_dir(self):
		self.assertTrue(self.flow.dataset_dir == 'dataset_dir', "Dataset directory not found")

	def test_data_dir(self):
		self.assertTrue(self.flow.data_dir == 'data_dir', "Data directory not found")
	
	def test_classes(self):
		self.assertTrue(self.flow.classes == ['class0', 'class1', 'class2', 'class3', 'class4'], "Classes not properly generated")

	def test_num_classes(self):
		self.assertTrue(self.flow.num_classes == 5, "Classes not properly generated")

	def test_checkpoints(self):
		self.assertFalse(os.path.exists('.ipynb_checkpoints'), ".ipynb checkpoints present")

	def test_median_applied(self):
		for i in self.flow.classes:
			self.assertTrue(len(os.listdir(f'./dataset_dir/data_dir/{i}/MedianImages')) == 5, f"Median filter not properly applied for {i}")

	def test_laplacian_applied(self):
		for i in self.flow.classes:
			self.assertTrue(len(os.listdir(f'./dataset_dir/data_dir/{i}/LaplacianImages')) == 5, f"Laplacian filter not properly applied for {i}")
	
	def test_sobelx_applied(self):
		for i in self.flow.classes:
			self.assertTrue(len(os.listdir(f'./dataset_dir/data_dir/{i}/SobelxImages')) == 5, f"Sobelx filter not properly applied for {i}")
	
	def test_sobely_applied(self):
		for i in self.flow.classes:
			self.assertTrue(len(os.listdir(f'./dataset_dir/data_dir/{i}/SobelyImages')) == 5, f"Sobely filter not properly applied for {i}")

	def test_gaussian_applied(self):
		for i in self.flow.classes:
			self.assertTrue(len(os.listdir(f'./dataset_dir/data_dir/{i}/GaussianImages')) == 5, f"Gaussian filter not properly applied for {i}")

	def test_processed_data_created(self):
		for i in self.flow.classes:
			for j in self.filters:
				for k in os.listdir(f'./dataset_dir/data_dir/{i}/{j.capitalize() + "Images"}'):
					self.assertTrue(os.path.exists(f'./dataset_dir/processedData/{k}'), "Images not moved to processedData folder")

	def test_training_split(self):
		self.assertTrue((self.train_x.shape[0] == self.train_y.shape[0] == 75) and (self.val_x.shape[0] == self.val_y.shape[0] == 50), "Data is not split properly according to ratio given")

	def test_img_resize(self):
		self.assertTrue(self.train_x.shape[1:] == self.val_x.shape[1:] == self.img_dimensions == (150, 150, 1), "Data is not resized properly according to dimensions given")

	def test_training_classes(self):
		self.assertTrue(self.train_y.shape[-1] == self.val_y.shape[-1] == self.flow.num_classes == 5, "Data is not split properly according to classes given")

if __name__ == '__main__':
	unittest.main()