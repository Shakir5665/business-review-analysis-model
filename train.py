"""
Training Script

Runs complete model training.

Project:
AI-Powered Business Risk Analysis
and Recommendation System\
"""


from src.dataset.dataset_loader import DatasetLoader

from src.tokenization.tokenizer import ReviewTokenizer

from src.encoding.sentiment_encoder import SentimentEncoder

from src.encoding.aspect_encoder import AspectEncoder

from src.dataloader.business_dataset import BusinessRiskDataset

from src.dataloader.data_loader import BusinessDataLoader

from src.models.business_risk_model import BusinessRiskModel

from src.training.optimizer import OptimizerFactory

from src.training.scheduler import SchedulerFactory

from src.training.trainer import Trainer

from src.utils.device import get_device

from configs.training_config import NUM_EPOCHS

from src.training.class_weights import ClassWeightCalculator

from src.training.losses import LossComputer


# --------------------------------------------------
# Device
# --------------------------------------------------

device = get_device()

# --------------------------------------------------
# Dataset
# --------------------------------------------------

loader = DatasetLoader()

train_df = loader.load_train()

validation_df = loader.load_validation()

tokenizer = ReviewTokenizer()

sentiment_encoder = SentimentEncoder()

aspect_encoder = AspectEncoder()

train_dataset = BusinessRiskDataset(

    dataframe=train_df,

    tokenizer=tokenizer,

    sentiment_encoder=sentiment_encoder,

    aspect_encoder=aspect_encoder

)

validation_dataset = BusinessRiskDataset(

    dataframe=validation_df,

    tokenizer=tokenizer,

    sentiment_encoder=sentiment_encoder,

    aspect_encoder=aspect_encoder

)

# --------------------------------------------------
# Compute Class Weights
# --------------------------------------------------

class_weight_calculator = ClassWeightCalculator()

sentiment_weights = class_weight_calculator.sentiment_weights(
    train_df
).to(device)

aspect_pos_weights = class_weight_calculator.aspect_pos_weights(
    train_df
).to(device)


# --------------------------------------------------
# Loss Function
# --------------------------------------------------

loss_computer = LossComputer(

    sentiment_weights=sentiment_weights,

    aspect_pos_weights=aspect_pos_weights

)



# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

data_loader = BusinessDataLoader()

train_loader = data_loader.create_train_loader(

    train_dataset

)

validation_loader = data_loader.create_validation_loader(

    validation_dataset

)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = BusinessRiskModel(

    num_aspect_classes=3

)

model.to(device)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = OptimizerFactory.create(

    model

)

# --------------------------------------------------
# Scheduler
# --------------------------------------------------

total_training_steps = (

    len(train_loader)

    * NUM_EPOCHS

)

scheduler = SchedulerFactory.create(

    optimizer,

    total_training_steps

)

print("\nSentiment Weights")
print(sentiment_weights)

print("\nAspect Positive Weights")
print(aspect_pos_weights)

# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Trainer(

    model=model,

    train_loader=train_loader,

    validation_loader=validation_loader,

    optimizer=optimizer,

    scheduler=scheduler,

    loss_computer=loss_computer,

    device=device


)


# --------------------------------------------------
# Resume Training
# --------------------------------------------------

checkpoint = trainer.checkpoint_manager.load_latest(

    model=model,

    optimizer=optimizer,

    scheduler=scheduler

)

if checkpoint is not None:

    trainer.start_epoch = checkpoint["epoch"] + 1
    trainer.load_history()

    if "validation_loss" in checkpoint:
        trainer.best_validation_loss = checkpoint["validation_loss"]

    print()

    print("=" * 60)

    print(

        f"Resuming Training From Epoch "

        f"{trainer.start_epoch}"

    )

    print("=" * 60)

    
# --------------------------------------------------
# Start Training
# --------------------------------------------------

trainer.train()