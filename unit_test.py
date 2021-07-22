import unittest
import torch
import tarzan.core as tz


ANNOT_FILE = 'data/mnist_jpg/labels.csv'
IMG_DIR = 'data/mnist_jpg/raw'
N_CLASSES = 10
BATCH_SIZE = 32


# Assuming that labels are for classification (NOT segmentation)
class TestData(unittest.TestCase):
    def setUp(self):
        # Tests that Data loads images without error
        self.dataset = tz.Data()
        self.dataset.load_image(ANNOT_FILE, IMG_DIR)
    
    def test_load(self):
        image, y = self.dataset.data[0]
        self.assertEqual(torch.Size((1, 28, 28)), image.shape)
        self.assertEqual(torch.Size([]), y.shape)

    def test_split(self):
        self.dataset.split(0.5)
        self.assertEqual(len(self.dataset.train), 30000)
        self.assertEqual(len(self.dataset.test), 30000)
        
        image, y = self.dataset.train[0]
        self.assertEqual(torch.Size((1, 28, 28)), image.shape)
        self.assertEqual(torch.Size([]), y.shape)

        image, y = self.dataset.test[0]
        self.assertEqual(torch.Size((1, 28, 28)), image.shape)
        self.assertEqual(torch.Size([]), y.shape)

    def test_both_dataloaders(self):
        self.dataset.split()

        train_dataloader = self.dataset.train_dataloader()        
        for batch in train_dataloader:
            image, y = batch
            self.assertEqual(torch.Size([BATCH_SIZE, 1, 28, 28]), image.shape)
            self.assertEqual(torch.Size([BATCH_SIZE]), y.shape)
            break

        test_dataloader = self.dataset.test_dataloader()
        for batch in test_dataloader:
            image, y = batch
            self.assertEqual(torch.Size([BATCH_SIZE, 1, 28, 28]), image.shape)
            self.assertEqual(torch.Size([BATCH_SIZE]), y.shape)
            break

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset = tz.Data()
        self.dataset.load_image(ANNOT_FILE, IMG_DIR)
        self.dataset.split(0.8)

        test_dataloader = self.dataset.train_dataloader()

        # Tests that Classifier instantiates
        self.clf = tz.Classifier("mlp")

    def test_training_step(self):
        train_dataloader = self.dataset.train_dataloader()
        # Test neural network architecture / model 
        for batch in train_dataloader:
            # Catch error if input does not match architecture of model
            loss = self.clf.training_step(batch)
            self.assertEqual(torch.Size([]), loss.shape)
            break

    def test_predict_proba_step(self):
        test_dataloader = self.dataset.test_dataloader()
        # Test neural network architecture / model 
        for batch in test_dataloader:
            # Catch error if input does not match architecture of model
            y_hat = self.clf.predict_proba_step(batch)

            self.assertEqual(torch.Size([BATCH_SIZE, N_CLASSES]), y_hat.shape)
            break

    def test_predict_step(self):
        test_dataloader = self.dataset.test_dataloader()
        # Test neural network architecture / model 
        for batch in test_dataloader:
            # Catch error if input does not match architecture of model
            predicted_labels = self.clf.predict_step(batch)
            self.assertEqual(torch.Size([BATCH_SIZE]), predicted_labels.shape)
            break

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.dataset = tz.Data()
        self.dataset.load_image(ANNOT_FILE, IMG_DIR)
        self.dataset.split(0.8)
        
        self.clf = tz.Classifier("mlp")
        self.clf.set_optimiser("adam")
        self.clf.set_scheduler("exponential")

        self.trainer = tz.Trainer()

    # def test_update_params(self):
        # Test that model parameters changes on step

    def test_predict(self):
        test_dataloader = self.dataset.test_dataloader()
        self.trainer.model = self.clf
        list_pred = self.trainer.predict(test_dataloader)
        pred = list_pred[0]
        self.assertEqual(torch.Size([BATCH_SIZE]), pred.shape)

    def test_count_classes(self):
        test_dataloader = self.dataset.test_dataloader()
        self.trainer.model = self.clf
        n_classes = self.trainer.count_classes(test_dataloader)
        print(f'Classes: {n_classes}')
        self.assertLessEqual(n_classes, N_CLASSES)

    def test_test(self):
        test_dataloader = self.dataset.test_dataloader()
        self.trainer.model = self.clf
        acc = self.trainer.test(test_dataloader)
        self.assertLessEqual(acc, 100)


if __name__ == '__main__':
    unittest.main()
