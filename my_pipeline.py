import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from scsims import SIMS

# handles data preprocessing like filtering, normalization, scaling, and selecting hvgs
class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.adata = None

    def preprocess_data(self):
        # load the dataset and process it
        self.adata = sc.read_h5ad(self.data_path)
        self.adata.var_names_make_unique()  # make variable names unique
        sc.pp.filter_cells(self.adata, min_genes=100)  # filter cells with few genes
        sc.pp.filter_genes(self.adata, min_cells=3)  # filter genes expressed in few cells
        sc.pp.normalize_total(self.adata)  # normalize total counts
        sc.pp.log1p(self.adata)  # log-transform the data
        self.adata.X = self.adata.X.toarray() if not isinstance(self.adata.X, np.ndarray) else self.adata.X  # ensure dense
        sc.pp.scale(self.adata)  # scale the data
        sc.pp.highly_variable_genes(self.adata, n_top_genes=2000)  # select top hvgs
        self.adata = self.adata[:, self.adata.var["highly_variable"]]  # subset to hvgs
        return self.adata

# trains the model using the processed data and logs progress
class TrainerClass:
    def __init__(self, traindata, valdata, columna):
        self.traindata = traindata
        self.valdata = valdata
        self.columna = columna
        self.model = None
        self.best_model_path = None

    def train_model(self, weight_decay, lr):
        # setup logger for wandb
        wandb_logger = WandbLogger(project="generic_sims", name=f"Training_{self.columna}_wd_{weight_decay}_lr_{lr}")

        # initialize the sims model
        sims = SIMS(
            data=self.traindata,
            class_label=self.columna,
            num_workers=os.cpu_count(),
            batch_size=2048,
        )
        sims.setup_model(
            weights=sims.weights,
            optim_params={"lr": lr, "weight_decay": weight_decay},
        )

        # setup callbacks for training
        early_stopping = EarlyStopping(monitor="val_micro_accuracy", patience=50, mode="max")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_micro_accuracy", filename=f"best_model_wd_{weight_decay}_lr_{lr}", save_top_k=1, mode="max"
        )

        # setup trainer with callbacks and logger
        sims.setup_trainer(
            early_stopping_patience=50,
            max_epochs=1000,
            devices=1,
            accelerator="gpu",
            logger=wandb_logger,
            callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        )

        # train the model
        sims.train(weights=sims.weights)
        self.model = sims
        self.best_model_path = checkpoint_callback.best_model_path

    def get_best_model_path(self):
        # return path to the best model
        return self.best_model_path

# makes predictions and handles evaluation metrics like confusion matrix
class Predictor:
    def __init__(self, model, valdata, columna):
        self.model = model
        self.valdata = valdata
        self.columna = columna

    def predict(self):
        # predict using the trained model
        predictions = self.model.predict(self.valdata)
        return predictions

    def visualize_confusion_matrix(self, predictions):
        # create a confusion matrix for true vs predicted labels
        true_labels = self.valdata.obs[self.columna]
        class_names = true_labels.unique()
        cm = confusion_matrix(true_labels.values, predictions.values, labels=class_names)
        cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # plot the confusion matrix
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm_percentage, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix (Percentage)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

# manages the pipeline, combining preprocessing, training, and predictions
class GenericPipeline:
    def __init__(self, data_path, columna, weight_decay_values, lr_values):
        self.data_path = data_path
        self.columna = columna
        self.weight_decay_values = weight_decay_values
        self.lr_values = lr_values

    def run_pipeline(self):
        # preprocess the data
        print("Starting data preprocessing...")
        preprocessor = Preprocessor(self.data_path)
        adata = preprocessor.preprocess_data()
        print("Data preprocessing completed.")

        # split data into training and validation sets
        print("Splitting data...")
        indices = np.arange(len(adata))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, stratify=adata.obs[self.columna], random_state=42
        )
        traindata = adata[train_indices]
        valdata = adata[val_indices]
        print("Data split completed.")

        # grid search over weight decay and learning rate
        print("Starting grid search...")
        for weight_decay in self.weight_decay_values:
            for lr in self.lr_values:
                print(f"Training with weight_decay={weight_decay}, lr={lr}...")
                trainer = TrainerClass(traindata, valdata, self.columna)
                trainer.train_model(weight_decay, lr)
                print(f"Completed training for weight_decay={weight_decay}, lr={lr}. Best model saved at: {trainer.get_best_model_path()}")

        # generate predictions and evaluate using the best model (example)
        print("Generating predictions...")
        best_trainer = TrainerClass(traindata, valdata, self.columna)
        best_trainer.train_model(self.weight_decay_values[0], self.lr_values[0])  # just an example to use the first params
        predictor = Predictor(best_trainer.model, valdata, self.columna)
        predictions = predictor.predict()
        predictor.visualize_confusion_matrix(predictions["pred_1"])

# parse command-line arguments for customization
def parse_args():
    parser = argparse.ArgumentParser(description="Generic Training Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--columna", type=str, default="label", help="Column to classify")
    parser.add_argument("--weight_decay_values", type=float, nargs='+', default=[0.001, 0.01], help="List of weight decay values")
    parser.add_argument("--lr_values", type=float, nargs='+', default=[0.01, 0.001], help="List of learning rate values")
    return parser.parse_args()

# main entry point to run the pipeline
def main():
    args = parse_args()
    pipeline = GenericPipeline(
        data_path=args.data_path,
        columna=args.columna,
        weight_decay_values=args.weight_decay_values,
        lr_values=args.lr_values,
    )
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
