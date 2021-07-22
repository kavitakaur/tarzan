import tarzan.core as tz
import torch
from torchvision import transforms
import tkinter as tk


class Window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Machine Learning')

        main_frame = tk.Frame(self)
        main_frame.grid(row=0, column=0) # sticky="nswe")

        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0) # sticky="nswe")

        self.right_frame = tk.Frame(main_frame)
        self.right_frame.grid(row=0, column=1) # sticky="nswe")

        # Section Header for loading data

        tk.Label(left_frame, text="PART 1: LOAD DATA", font=("", 18), fg='blue').grid(row=0)

        # User Interface to load data - 2 input fields and 1 button

        tk.Label(left_frame,
            text="Input directory of annotation file. For e.g., data/mnist_jpg/labels.csv"
        ).grid(row=1)
        self.e1 = tk.Entry(left_frame)
        self.e1.grid(row=2) 

        tk.Label(
            left_frame,
            text="Input directory of image file"
        ).grid(row=3) 
        self.e2 = tk.Entry(left_frame)
        self.e2.grid(row=4)

        # Get Data button
        btn1 = tk.Button(
            left_frame, text='Get Data', fg='purple', bg='grey',
            command=self.get_data
        )
        btn1.grid(row=5, columnspan=2)

        # Creating an output panel so users can view outcome of the function call

        self.canvas = tk.Canvas(self.right_frame, bg='grey') # height=500, width=500, 
        self.canvas.grid(column=1)

        output = tk.Label(self.right_frame, text="Output:").grid(row=0)

        #TODO: Alert for failed function call


        # Section Header for loading DL model and running ML tasks

        tk.Label(left_frame,
            text="PART 2: TRAIN DATA", font=("", 18), fg='blue'
        ).grid(row=6)

        # User Interface to load DL model and run ML tasks - 3 input fields and 1 button

        #TODO: Make dropdown menu for options
        tk.Label(
            left_frame,
            text="Input DL Model - Choose either 'mlp', 'cnn', 'fnn', 'unet' or 'vggnet'"
        ).grid(row=7)
        self.e3 = tk.Entry(left_frame)
        self.e3.grid(row=8) 
        
        #TODO: Make dropdown menu for options
        tk.Label(
            left_frame,
            text="Input optimiser - Choose either 'adam' or 'sgd'"
        ).grid(row=9)
        self.e4 = tk.Entry(left_frame)
        self.e4.grid(row=10)

        #TODO: Make dropdown menu for options
        tk.Label(
            left_frame, 
            text="Input learning rate scheduler - Choose either 'step', 'multistep' or 'exponential'"
        ).grid(row=11)
        self.e5 = tk.Entry(left_frame)
        self.e5.grid(row=12)

        # 'Start Training' button
        btn2 = tk.Button(
            left_frame, text='Start Training', fg='purple', bg='grey',
            command=self.ML_tasks
        )
        btn2.grid(row=13, columnspan=2)

        # Display results for user view has been moved under the respective functions


    def get_data(self):
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        annot_path = self.e1.get()
        mnist_path = self.e2.get()
        self.dataset = tz.Data()
        
        self.dataset.load_image(
            annot_path,
            mnist_path,
            transformations
        )
        #DatasetFromImage.transform
        self.dataset.split()
        # self.dataset.save(save_path)
        # self.dataset.load(save_path)
        self.train_dataloader = self.dataset.train_dataloader()
        self.test_dataloader = self.dataset.test_dataloader()

        # Display for user - display the type so user knows data loaded
        data_type = tk.Text(self.canvas, height=5, width=30)
        data_type.grid(row=1, column=1)
        data_type.insert(tk.END, type(self.train_dataloader))
        data_type.insert(tk.END, type(self.test_dataloader))
        
        # Display for user - prompt to continue
        completion = tk.Text(self.canvas, height=5, width=30)
        completion.grid(row=2, column=1)
        completion.insert(tk.END, "Data loaded - Move on to Part 2.")

        #TODO: Alert for failed function call or error messages
    

    def ML_tasks(self):
        model = self.e3.get()
        optimiser = self.e4.get()
        scheduler = self.e5.get()

        # instantiate the neural network
        self.clf = tz.Classifier(model)
        self.clf.set_optimiser(optimiser)
        self.clf.set_scheduler(scheduler)

        self.trainer = tz.Trainer()
        self.trainer.fit(self.clf, self.train_dataloader, epochs=1)
        #acc = self.trainer.test(self.test_dataloader)
        
        predictions = self.trainer.predict(self.test_dataloader)
        class_count = self.trainer.count_classes(self.test_dataloader)
        
        # Display for user - display the result of predict and count_classes tasks

        result = tk.Text(self.canvas)  # height=5, width=30)
        result.grid(row=3, column=1) # rowspan=7)
        result.insert(tk.END, predictions)
        
        result2 = tk.Text(self.canvas, height=5, width=30)  # height=5, width=30)
        result2.grid(row=4, column=1)
        result2.insert(tk.END, class_count)

        #TODO: Alert for failed function call or error messages


if __name__ == '__main__':
    window = Window()
    window.mainloop()
